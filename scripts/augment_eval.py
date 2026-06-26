"""Experiment — augmentation for deployment: accuracy + robustness.

augment.py (photometric+geometric, q follows) was never used. Two deployment
levers, training-free, on frozen exemplar retrieval:
  aug-gallery : add K augmented views of each gallery pin as extra exemplars
                (covers intra-class variation -> accuracy).
  TTA         : average each test pin over K augmented views (stable query).
And the deployment question that matters most: ROBUSTNESS to domain shift —
re-embed the test under a strong photometric corruption (different camera/lighting)
and see how much top1 drops, with vs without an augmented gallery.

10-seed mean±std, paired. Logged via explog.

    .venv/bin/python scripts/augment_eval.py
"""

from __future__ import annotations

import collections
import datetime
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

import explog  # noqa: E402
from eval_appearance import split_indices, _MEAN, _STD, _git_sha  # noqa: E402
from learned_pooler import gauss_pool  # noqa: E402
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.data.augment import augment  # noqa: E402
from scalpel.data.parse import Triple  # noqa: E402
from scalpel.perception import DinoBackbone  # noqa: E402

K = 4


def split_idx(tris, test_frac, seed):
    """Specimen-level split on Triple objects (.src/.page)."""
    g = collections.defaultdict(list)
    for i, t in enumerate(tris):
        g[f"{t.src}#{t.page}"].append(i)
    keys = sorted(g)
    np.random.default_rng(seed).shuffle(keys)
    nt = max(1, int(round(len(keys) * test_frac)))
    tk = set(keys[:nt])
    tr = [i for k in keys if k not in tk for i in g[k]]
    te = [i for k in keys if k in tk for i in g[k]]
    return tr, te


def load_triples(jsonl, base, min_count=2):
    rows = [json.loads(l) for l in open(jsonl, encoding="utf-8") if l.strip()]
    cnt = collections.Counter(r["label"] for r in rows)
    rows = [r for r in rows if cnt[r["label"]] >= min_count]
    tris = []
    for r in rows:
        img = np.asarray(Image.open(base / r["image"]).convert("RGB"))
        tris.append(Triple(img, tuple(r["q"]), r["label"], int(r["page"]), r["src"]))
    return tris


def corrupt(img, rng):
    """Strong, deterministic domain shift (camera/lighting) for the robustness test."""
    x = img.astype(np.float32)
    x *= 0.6                                     # darker
    x = (x - 128) * 1.35 + 128                   # higher contrast
    x *= np.array([1.15, 0.95, 0.85])[None, None, :]   # warm colour cast
    x += rng.normal(0, 8, x.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


@torch.no_grad()
def embed_items(items, backbone, centers, S, device, chunk=32):
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)
    Z = []
    for s in range(0, len(items), chunk):
        batch = items[s:s + chunk]
        arrs, qs = [], []
        for img, q in batch:
            h, w = img.shape[:2]
            arrs.append(np.asarray(Image.fromarray(img).resize((S, S)), np.float32) / 255.0)
            qs.append([q[0] * S / w, q[1] * S / h])
        x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).to(device)
        grid, _ = backbone((x - mean) / std)
        z = gauss_pool(grid, centers, torch.tensor(qs, device=device))
        Z.append(z.cpu())
    return torch.cat(Z).numpy()


def exemplar(galZ, galY, teZ, teY):
    labels = sorted(set(galY)); lidx = {l: i for i, l in enumerate(labels)}
    cols = collections.defaultdict(list)
    for j, l in enumerate(galY):
        cols[lidx[l]].append(j)
    cv = [k for k, l in enumerate(teY) if l in lidx]
    if not cv:
        return 0.0, 0.0
    sims = teZ[cv] @ galZ.T
    sc = np.full((len(cv), len(labels)), -2.0, np.float32)
    for c, ix in cols.items():
        sc[:, c] = sims[:, ix].max(1)
    o = np.argsort(-sc, axis=1)
    n1 = sum(int(labels[o[r, 0]] == teY[k]) for r, k in enumerate(cv))
    n5 = sum(int(teY[k] in [labels[o[r, t]] for t in range(5)]) for r, k in enumerate(cv))
    return 100 * n1 / len(cv), 100 * n5 / len(cv)


def main():
    base = Path("data/triples")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = PipelineCfg(); S = cfg.backbone.image_size
    tris = load_triples("data/triples/triples.jsonl", base)
    Y = [t.label for t in tris]
    ncls = len(set(Y))
    print(f"core {len(tris)}/{ncls} | device={device}")
    bb = DinoBackbone(cfg.backbone); bb.ensure_loaded(); bb.to(device)
    centers = bb.patch_centers(device)

    print("embedding originals..."); Zo = embed_items([(t.image, t.q) for t in tris], bb, centers, S, device)
    print(f"embedding {K} augmented views each...")
    aug = [augment(t, K, seed=i) for i, t in enumerate(tris)]
    Za = embed_items([(a.image, a.q) for sub in aug for a in sub], bb, centers, S, device).reshape(len(tris), K, -1)
    print("embedding corrupted test copies...")
    rng = np.random.default_rng(0)
    Zc = embed_items([(corrupt(t.image, rng), t.q) for t in tris], bb, centers, S, device)

    def norm(a):
        return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    Zo, Zc = norm(Zo), norm(Zc)
    Za = norm(Za)
    Ztta = norm(np.concatenate([Zo[:, None], Za], axis=1).mean(1))   # mean over orig+aug

    cfgs = ["base", "aug-gal", "tta", "both"]
    acc = {c: ([], []) for c in cfgs}
    rob = {"base": [], "aug-gal": []}      # top1 on CORRUPTED test
    for seed in range(10):
        tr, te = split_idx(tris, 0.3, seed)
        ytr = [Y[i] for i in tr]; yte = [Y[i] for i in te]
        galA_Z = np.concatenate([Zo[tr]] + [Za[tr, k] for k in range(K)])
        galA_Y = ytr * (K + 1)
        acc["base"][0].append(exemplar(Zo[tr], ytr, Zo[te], yte)[0]); acc["base"][1].append(exemplar(Zo[tr], ytr, Zo[te], yte)[1])
        a = exemplar(galA_Z, galA_Y, Zo[te], yte); acc["aug-gal"][0].append(a[0]); acc["aug-gal"][1].append(a[1])
        t = exemplar(Zo[tr], ytr, Ztta[te], yte); acc["tta"][0].append(t[0]); acc["tta"][1].append(t[1])
        b = exemplar(galA_Z, galA_Y, Ztta[te], yte); acc["both"][0].append(b[0]); acc["both"][1].append(b[1])
        rob["base"].append(exemplar(Zo[tr], ytr, Zc[te], yte)[0])
        rob["aug-gal"].append(exemplar(galA_Z, galA_Y, Zc[te], yte)[0])
        print(f"  seed {seed}: base {acc['base'][0][-1]:.0f} aug-gal {a[0]:.0f} tta {t[0]:.0f} both {b[0]:.0f} "
              f"| corrupt base {rob['base'][-1]:.0f} aug {rob['aug-gal'][-1]:.0f}")

    ms = lambda v: (round(st.mean(v), 1), round(st.pstdev(v), 1))
    rows = [(c, *ms(acc[c][0]), *ms(acc[c][1])) for c in cfgs]
    base1 = rows[0][1]
    # paired deltas vs base
    def paired(c):
        d = [a - b for a, b in zip(acc[c][0], acc["base"][0])]
        return round(st.mean(d), 1), sum(x > 0 for x in d)
    rb, ra = ms(rob["base"]), ms(rob["aug-gal"])
    rob_gain = round(ra[0] - rb[0], 1)
    print("\n== clean: " + " ".join(f"{c} {ms(acc[c][0])[0]}" for c in cfgs)
          + f" | CORRUPT base {rb[0]} -> aug-gal {ra[0]} (Δ{rob_gain}) ==")

    d = explog.next_dir("augmentation")
    explog.bar(d / "fig_aug.png", cfgs, [r[1] for r in rows],
               "Augmentation: top1 (clean, 10-seed)", "%", ymax=100, errors=[r[2] for r in rows])
    explog.bar(d / "fig_robust.png", ["base", "aug-gal"], [rb[0], ra[0]],
               "Robustness: top1 on CORRUPTED test", "%", ymax=100, errors=[rb[1], ra[1]])
    ptab = "\n".join(f"| {c} | {r1}±{s1}% | {r5}±{s5}% | {('+' if paired(c)[0]>=0 else '')}{paired(c)[0]} ({paired(c)[1]}/10) |"
                     for (c, r1, s1, r5, s5) in rows)
    report = f"""# 증강 — 정확도 + 견고성 (augmentation)

- 날짜: {datetime.date.today().isoformat()}
- 커밋: `data-pivot @ {_git_sha()}`
- 스크립트: `scripts/augment_eval.py`  (frozen exemplar, 10-seed)

## 목적
배포 견고성: augment.py(조명·기하, q 따라감, K={K})를 **갤러리 확장 + 테스트 TTA**로 쓰고,
**강한 손상(어둡게+대비+색캐스트+노이즈=다른 카메라/조명)에 대한 robustness**를 측정.

## 정확도 (clean test, paired vs base)
| 설정 | top1 | top5 | Δtop1(paired) |
|---|---|---|---|
{ptab}

![aug](fig_aug.png)

## 견고성 (손상 test에서 top1)
| 갤러리 | top1(손상) |
|---|---|
| base (원본만) | {rb[0]}±{rb[1]}% |
| **aug-gallery** | **{ra[0]}±{ra[1]}%** |

손상 시 base는 clean {base1}%→{rb[0]}%로 추락, augmented gallery는 {ra[0]}% (Δ{'+' if rob_gain>=0 else ''}{rob_gain}%p) → **배포 환경 변화에 대한 강건성** 지표.

![robust](fig_robust.png)

## 해석 / 다음
- clean 정확도 이득은 작아도(증강은 새 시신을 안 만듦), **손상 robustness가 오르면 배포 가치 큼**.
- 다음: 핀 노이즈 강건성, val 기반 운영점 고정(정확도 보장), open-set 기권.
"""
    explog.write(d, report, {
        "title": "증강 (정확도+견고성)", "date": datetime.date.today().isoformat(),
        "headline": f"clean base {base1}→aug-gal {ms(acc['aug-gal'][0])[0]}; CORRUPT base {rb[0]}→aug {ra[0]} (Δ{rob_gain})",
        "clean": {c: ms(acc[c][0]) for c in cfgs}, "corrupt_base": rb, "corrupt_aug": ra})
    print(f"wrote -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
