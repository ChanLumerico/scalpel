"""Experiment 050 — M-rep1 (LoRA): reshape the BACKBONE last block, not just a head on frozen.

049 trained a projection head on the *frozen* pooled embedding — it can only re-project the 1536-d
vector, not recover a cue σ40-pooling already discarded. LoRA is the distinct lever the handout named:
low-rank adapters on the last DINO block change the *patch tokens before pooling*, so the pooled vector
itself carries different information. This is the most expressive (and most overfit-prone, §2) option.

Cost control: cache the INPUT to the last block once (frozen partial forward); LoRA then trains only the
last block on cached tokens (no full-backbone forward per step). Leak-safety: LoRA trains on **dev only**,
so the **sealed test is a clean leak-safe estimate** (test never seen). dev-CV (LoRA trained on all dev)
is the optimistic upper bound — if LoRA can't beat frozen even there, the lever is dead (cheap kill).

  baseline   frozen global+L256 (045: dev 33.5 / sealed 36.1)
  treatment  [LoRA-reshaped global ; frozen L256]  and  LoRA-global alone
LoRA on last block {qkv,proj,fc1,fc2}, rank 8, trained with class cross-entropy (max discriminative).

    .venv/bin/python scripts/lora_reshape.py
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
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import explog  # noqa: E402
from eval_appearance import _MEAN, _STD, _git_sha  # noqa: E402
from eval_merged import load, exemplar_eval, block_split, BASE  # noqa: E402
from split_devtest import get_split  # noqa: E402
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.perception import DinoBackbone, GaussianPool  # noqa: E402

SEEDS = 10
EPOCHS = 12
RANK = 8
ALPHA = 16


class LoRALinear(nn.Module):
    def __init__(self, orig, r=RANK, alpha=ALPHA):
        super().__init__()
        self.orig = orig
        self.A = nn.Parameter(torch.randn(r, orig.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(orig.out_features, r))
        self.s = alpha / r

    def forward(self, x):
        return self.orig(x) + self.s * (x @ self.A.t() @ self.B.t())


def unit(Z):
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)


def ms(v):
    v = [x for x in v if x == x]
    return (round(st.mean(v), 1), round(st.pstdev(v), 1)) if v else (float("nan"), 0.0)


@torch.no_grad()
def cache_penult(rows, bb, S, device):
    """Cache the input token-seq to the last block, per unique image (disk-persisted float16)."""
    imgs = sorted(set(r["image"] for r in rows))
    arrf = BASE / "_penult_cache.npy"; idxf = BASE / "_penult_imgs.json"
    if arrf.exists() and idxf.exists() and json.loads(idxf.read_text()) == imgs:
        print("   loaded cached penult")
        A = np.load(arrf, mmap_mode="r")
        return {im: A[k] for k, im in enumerate(imgs)}
    from PIL import Image
    cap = {}
    h = bb._model.blocks[-1].register_forward_pre_hook(lambda m, a: cap.__setitem__("x", a[0]))
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)
    store = {}
    for n, img in enumerate(imgs, 1):
        arr = np.asarray(Image.open(BASE / img).convert("RGB").resize((S, S)), np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        bb._model.forward_features((x - mean) / std)
        store[img] = cap["x"][0].half().cpu().numpy()
        if n % 150 == 0:
            print(f"   penult cache {n}/{len(imgs)}")
    h.remove()
    np.save(arrf, np.stack([store[im] for im in imgs]))
    idxf.write_text(json.dumps(imgs))
    del store
    A = np.load(arrf, mmap_mode="r")          # stream from disk, keep RAM low
    return {im: A[k] for k, im in enumerate(imgs)}


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = PipelineCfg(); S = cfg.backbone.image_size; g = cfg.backbone.grid_size
    rows = load(); Y = [r["label"] for r in rows]
    # q must be in resized-S coordinates (penult grid is at S); store per-image scale
    from PIL import Image
    wh = {}
    split = get_split()
    cnt = collections.Counter(Y)
    core = [i for i in range(len(rows)) if cnt[Y[i]] >= 2]
    dev = [i for i in core if split[rows[i]["image"]] == "dev"]
    test = [i for i in core if split[rows[i]["image"]] == "test"]
    block_id = [json.loads((BASE / "_blocks.json").read_text())[r["image"]] for r in rows]
    zg = unit(np.load(BASE / "_dino_cache.npy").astype(np.float32))
    zl = unit(np.load(BASE / "_local256_cache.npy").astype(np.float32))
    Zf = unit(np.concatenate([zg, zl], 1)).astype(np.float32)
    print(f"core {len(core)} | dev {len(dev)} / test {len(test)} | {device}")

    bb = DinoBackbone(cfg.backbone); bb.ensure_loaded(); bb.to(device)
    pool = GaussianPool(cfg.point).to(device); centers = bb.patch_centers(device)
    # rescale q into S coords (image resized w,h -> S)
    for i in core:
        im = rows[i]["image"]
        if im not in wh:
            wh[im] = Image.open(BASE / im).size  # (w,h)
    qS = {i: (rows[i]["q"][0] * S / wh[rows[i]["image"]][0], rows[i]["q"][1] * S / wh[rows[i]["image"]][1]) for i in core}
    rows2 = [dict(r) for r in rows]
    for i in core:
        rows2[i]["q"] = [qS[i][0], qS[i][1]]

    print("caching penult tokens (last-block input)...")
    penult = cache_penult([rows2[i] for i in core], bb, S, device)

    # inject LoRA into the last block
    blk = bb._model.blocks[-1]
    blk.attn.qkv = LoRALinear(blk.attn.qkv); blk.attn.proj = LoRALinear(blk.attn.proj)
    blk.mlp.fc1 = LoRALinear(blk.mlp.fc1); blk.mlp.fc2 = LoRALinear(blk.mlp.fc2)
    blk.to(device); norm = bb._model.norm.to(device)
    lora_params = [p for n, p in blk.named_parameters() if p.requires_grad and (".A" in n or ".B" in n)]
    classes = sorted(set(Y[i] for i in dev)); cidx = {c: j for j, c in enumerate(classes)}
    clf = nn.Linear(zg.shape[1], len(classes)).to(device)
    opt = torch.optim.Adam(lora_params + list(clf.parameters()), lr=2e-3, weight_decay=1e-4)

    # train LoRA on dev (class CE) — optimistic for dev-CV, clean for sealed test
    devc = [i for i in dev]
    print(f"training LoRA last-block ({sum(p.numel() for p in lora_params)/1e3:.0f}K params) on {len(devc)} dev triples...")
    for ep in range(EPOCHS):
        np.random.default_rng(ep).shuffle(devc)
        tot = 0.0
        for s in range(0, len(devc), 16):                 # batch 16: attn activations ~2.9GB (no thrash)
            chunk = devc[s:s + 16]
            toks = torch.from_numpy(np.stack([np.asarray(penult[rows2[i]["image"]]) for i in chunk])).float().to(device)
            q = torch.tensor([rows2[i]["q"] for i in chunk], dtype=torch.float32, device=device)
            x = norm(blk(toks))[:, 1:, :]
            grid = x.reshape(x.shape[0], g, g, x.shape[-1])
            z = F.normalize(pool(grid, centers, q), dim=1)
            logits = clf(z)
            tgt = torch.tensor([cidx[Y[i]] for i in chunk], device=device)
            loss = F.cross_entropy(logits, tgt)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        if ep % 3 == 0 or ep == EPOCHS - 1:
            print(f"   ep {ep}: CE {tot/ (len(devc)//16+1):.3f}")

    # embed ALL core with the trained LoRA
    blk.eval()
    with torch.no_grad():
        Zl = [None] * len(rows)
        for s in range(0, len(core), 64):
            chunk = core[s:s + 64]
            toks = torch.from_numpy(np.stack([penult[rows2[i]["image"]] for i in chunk])).float().to(device)
            q = torch.tensor([rows2[i]["q"] for i in chunk], dtype=torch.float32, device=device)
            x = norm(blk(toks))[:, 1:, :]
            grid = x.reshape(x.shape[0], g, g, x.shape[-1])
            z = F.normalize(pool(grid, centers, q), dim=1).cpu().numpy()
            for j, i in enumerate(chunk):
                Zl[i] = z[j]
    zlora = np.zeros_like(zg)
    for i in core:
        zlora[i] = Zl[i]
    zlora = unit(zlora)
    Zlf = unit(np.concatenate([zlora, zl], 1))      # LoRA-global ⊕ frozen L256
    print("LoRA embeddings ready.")

    splits = [block_split(dev, block_id, s) for s in range(SEEDS)]

    def devcv(Z):
        return [exemplar_eval(Z, Y, tr, te)[0] for tr, te in splits]

    variants = {"frozen global+L256 (045)": Zf, "LoRA-global only": unit(zlora),
                "LoRA-global+L256": Zlf}
    res = {k: devcv(v) for k, v in variants.items()}
    base = res["frozen global+L256 (045)"]

    def paired(a):
        d = [x - y for x, y in zip(a, base)]
        return round(st.mean(d), 2), int(sum(x > 0 for x in d))

    print(f"\n== dev-CV 10-seed (LoRA trained on ALL dev = OPTIMISTIC) ==")
    table = {}
    for k in variants:
        dlt = paired(res[k]); table[k] = (ms(res[k]), dlt)
        print(f"  {k:26} top1 {ms(res[k])[0]}±{ms(res[k])[1]}  Δ {dlt[0]:+} ({dlt[1]}/10)")

    # sealed test (CLEAN: LoRA trained on dev, test unseen)
    def sealed(Z):
        return exemplar_eval(Z, Y, dev, test)[0]
    t_frozen, t_lora, t_lf = sealed(Zf), sealed(unit(zlora)), sealed(Zlf)
    best = max(["LoRA-global only", "LoRA-global+L256"], key=lambda k: st.mean(res[k]))
    d_best = table[best][1]
    # honest verdict keys on the SEALED test (clean) not the optimistic dev-CV
    adopt = t_lf > t_frozen and table["LoRA-global+L256"][1][0] > 0
    print(f"\n  ★ SEALED TEST (clean): frozen {round(t_frozen,1)} | LoRA-only {round(t_lora,1)} | "
          f"LoRA+L256 {round(t_lf,1)} → {'🟢 LoRA 이김' if adopt else '🔴 LoRA도 frozen 못 넘음'}")

    d = explog.EXP / "050-lora-reshape"; d.mkdir(parents=True, exist_ok=True)
    ks = list(variants)
    explog.bar(d / "fig1_devcv.png", [k.replace(" ", "\n") for k in ks], [table[k][0][0] for k in ks],
               "050 LoRA last-block reshape: dev-CV (LoRA trained on all dev = optimistic)", "top1 %",
               ymax=42, errors=[table[k][0][1] for k in ks])
    explog.bar(d / "fig2_sealed.png", ["frozen\nglobal+L256", "LoRA-global\nonly", "LoRA-global\n+L256"],
               [t_frozen, t_lora, t_lf], "050 SEALED test (clean: LoRA on dev, test unseen)", "top1 %", ymax=45)

    rowmd = "\n".join(f"| {k} | {table[k][0][0]}±{table[k][0][1]} | {table[k][1][0]:+} | {table[k][1][1]}/10 |" for k in ks)
    verdict = (f"🟢 **LoRA가 frozen을 이김** (봉인 {round(t_lf,1)} > {round(t_frozen,1)}) → 백본 reshape 유효, 채택."
               if adopt else
               f"🔴 **LoRA(백본 reshape)도 frozen exemplar를 못 넘는다** — 봉인 LoRA+L256 {round(t_lf,1)} ≤ frozen {round(t_frozen,1)}, "
               f"dev-CV 낙관적 상한조차 Δ{table['LoRA-global+L256'][1][0]:+}. head(049)에 이어 백본 적응도 무가산 → "
               f"M-rep1 전체 음성. 잔여 모호성은 학습으로 못 푸는 데이터/내재 한계(§2). 남은 레버 = 데이터.")
    report = f"""# 050 — M-rep1(LoRA): 마지막 블록 백본 reshape (head가 아닌 토큰 적응)

- 날짜: {datetime.date.today().isoformat()} · 커밋 `main @ {_git_sha()}` · `scripts/lora_reshape.py`
- clean 502 (dev {len(dev)}/test {len(test)} 봉인). LoRA(last block {{qkv,proj,fc1,fc2}}, rank {RANK})를
  **dev에서만** class-CE로 학습 → **봉인 test = 깨끗한 누수안전 평가**(test 미관측). dev-CV는 낙관적 상한.
- 049(frozen head)와 차이: LoRA는 풀링 *전* 패치 토큰을 바꿔 frozen이 못 담는 축 생성 가능.

## 결과
**dev-CV (LoRA가 dev 전체 학습 = 낙관적 상한)**
| variant | dev-CV top1 | Δ vs frozen | wins |
|---|---|---|---|
{rowmd}

**봉인 TEST (깨끗): frozen {round(t_frozen,1)} | LoRA-only {round(t_lora,1)} | LoRA+L256 {round(t_lf,1)}**

![devcv](fig1_devcv.png)
![sealed](fig2_sealed.png)

## 판정
{verdict}

## 핵심
- head(049)는 frozen 풀링벡터 재투영만, LoRA(050)는 토큰 자체 적응 — {'그럼에도 무가산' if not adopt else '가산'}.
- {'M-rep1(학습형 표현) 전체 음성 = 049 head + 050 LoRA 둘 다 frozen 못 넘음. 표현 축은 해상도(045) 외 완전 소진 → 데이터.' if not adopt else 'LoRA 채택 — 백본 적응이 표현 축을 추가 전진시킴.'}
"""
    explog.write(d, report, {
        "title": "M-rep1(LoRA): 마지막 블록 백본 reshape", "date": datetime.date.today().isoformat(),
        "headline": f"LoRA last-block: dev-CV LoRA+L256 {ms(res['LoRA-global+L256'])[0]} (Δ vs frozen "
                    f"{table['LoRA-global+L256'][1][0]:+}, 낙관) | 봉인 frozen {round(t_frozen,1)} vs LoRA+L256 "
                    f"{round(t_lf,1)} {'🟢채택' if adopt else '🔴무가산 → M-rep1 전체 음성, 데이터가 레버'}",
        "rank": RANK, "epochs": EPOCHS, "adopt": bool(adopt),
        "devcv": {k: {"top1": table[k][0], "delta_vs_frozen": table[k][1]} for k in ks},
        "sealed": {"frozen": round(t_frozen, 1), "lora_only": round(t_lora, 1), "lora_L256": round(t_lf, 1)}})
    print(f"\nwrote -> {d}  (2 figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
