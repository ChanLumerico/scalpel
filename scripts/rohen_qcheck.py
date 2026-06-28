"""Experiment 062b / M-rohen0 — Rohen q-quality self-consistency (no human labour).

062 raw-q effect was weak (top1 flat). Is that q-noise (off pins → wrong structure embedded) or a domain
gap (right pins, but Rohen structures just don't match our exemplars)? Diagnose WITHOUT verification:
query each matched Rohen exemplar against OUR dev gallery and ask whether our model recognises it as its
own labelled class. Compare that self-match rate to OUR OWN in-domain rate (test→dev) — the recognizability
ceiling. Rohen ≈ ours → pins fine, flat top1 is domain-hardness (verify won't help). Rohen ≪ ours → q-noise
or domain gap (verify *might* help, esp. if thin vessel/nerve pins are the worst).

    .venv/bin/python scripts/rohen_qcheck.py
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from eval_merged import load, BASE  # noqa: E402
from split_devtest import get_split  # noqa: E402
from rohen_effect import unit, norm, embed_rohen  # noqa: E402
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.perception import DinoBackbone, GaussianPool  # noqa: E402

TIS = {"artery": "artery", "vein": "vein", "nerve": "nerve", "muscle": "muscle", "bone": "bone"}


def tissue(lab):
    for w in reversed(lab.split()):
        if w in TIS:
            return TIS[w]
    return "other"


def readout(Zq, Zg, Yg, truth, k=5):
    """CSLS top1/top5 of queries (truth) vs gallery (Zg,Yg)."""
    labs = sorted(set(Yg)); li = {l: j for j, l in enumerate(labs)}
    cov = [i for i in range(len(truth)) if truth[i] in li]
    sqg = Zq[cov] @ Zg.T; gg = Zg @ Zg.T; np.fill_diagonal(gg, -9)
    rg = np.sort(gg, 1)[:, -k:].mean(1); rq = np.sort(sqg, 1)[:, -k:].mean(1)
    S = 2 * sqg - rq[:, None] - rg[None, :]
    cols = collections.defaultdict(list)
    for j, y in enumerate(Yg):
        cols[li[y]].append(j)
    sc = np.full((len(cov), len(labs)), -9.0, np.float32)
    for c, ix in cols.items():
        sc[:, c] = S[:, ix].max(1)
    order = np.argsort(-sc, 1)
    t1 = [labs[order[r, 0]] == truth[cov[r]] for r in range(len(cov))]
    t5 = [truth[cov[r]] in [labs[order[r, t]] for t in range(5)] for r in range(len(cov))]
    return t1, t5, cov


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = PipelineCfg(); S = cfg.backbone.image_size
    rows = load(); Y = [r["label"] for r in rows]
    split = get_split(); cnt = collections.Counter(Y)
    core = [i for i in range(len(rows)) if cnt[Y[i]] >= 2]
    dev = [i for i in core if split[rows[i]["image"]] == "dev"]
    test = [i for i in core if split[rows[i]["image"]] == "test"]
    zg = unit(np.load(BASE / "_dino_cache.npy").astype(np.float32))
    zl = unit(np.load(BASE / "_local256_cache.npy").astype(np.float32))
    Zours = unit(np.concatenate([zg, zl], 1))
    our_norm = {}
    for c in set(Y):
        our_norm.setdefault(norm(c), c)
    cands = json.loads(Path("data/rohen/candidates.json").read_text())
    for c in cands:
        nn = norm(c["name"]); c["label"] = our_norm.get(nn); c["matched"] = nn in our_norm

    cache = BASE.parent / "rohen" / "_emb.npz"
    if cache.exists():
        d = np.load(cache); Zrohen = d["Z"]; print("  cached rohen emb")
    else:
        bb = DinoBackbone(cfg.backbone); bb.ensure_loaded(); bb.to(device)
        pool = GaussianPool(cfg.point).to(device); centers = bb.patch_centers(device)
        print("embedding Rohen...")
        rzg, rzl = embed_rohen(cands, bb, pool, centers, S, device)
        Zrohen = unit(np.concatenate([rzg, rzl], 1)); np.savez(cache, Z=Zrohen)

    mi = [i for i, c in enumerate(cands) if c["matched"]]
    Zg = Zours[dev]; Yg = [Y[i] for i in dev]
    # Rohen matched → our gallery
    rtruth = [cands[i]["label"] for i in mi]
    rt1, rt5, rcov = readout(Zrohen[mi], Zg, Yg, rtruth)
    # our own in-domain (test → dev)
    otruth = [Y[i] for i in test]
    ot1, ot5, ocov = readout(Zours[test], Zg, Yg, otruth)

    r_top1 = 100 * np.mean(rt1); r_top5 = 100 * np.mean(rt5)
    o_top1 = 100 * np.mean(ot1); o_top5 = 100 * np.mean(ot5)
    # per-tissue Rohen self-match (thin vessel/nerve are q-sensitive)
    by = collections.defaultdict(lambda: [0, 0])
    for r, i in enumerate(rcov):
        t = tissue(cands[mi[i]]["label"]); by[t][0] += 1; by[t][1] += rt1[r]
    pt = {t: (v[1], v[0], round(100 * v[1] / v[0], 0)) for t, v in by.items() if v[0] >= 4}

    print(f"\n==== Rohen q-quality self-consistency (matched {len(mi)}) ====")
    print(f"  Rohen exemplar → our gallery:  top1 {r_top1:.1f}  top5 {r_top5:.1f}")
    print(f"  OUR in-domain (test → dev):    top1 {o_top1:.1f}  top5 {o_top5:.1f}  (recognizability ceiling)")
    print(f"  ratio Rohen/ours: top1 {r_top1/o_top1:.2f}  top5 {r_top5/o_top5:.2f}")
    print(f"  per-tissue Rohen top1: " + " ".join(f"{t}:{c}/{n}={p:.0f}%" for t, (c, n, p) in sorted(pt.items())))

    if r_top1 >= 0.8 * o_top1:
        verdict = (f"🟢 q-품질 양호 — Rohen self-match top1 {r_top1:.1f} ≈ 우리 in-domain {o_top1:.1f}. "
                   f"q는 대체로 맞음 → 062 top1-평탄은 *도메인 hardness*(매칭 90개·sub-gap)이지 q-노이즈 아님. "
                   f"손수 검증해도 큰 이득 없음 → Rohen은 coverage용.")
    elif r_top1 >= 0.45 * o_top1:
        verdict = (f"🟡 부분 q-노이즈 — Rohen {r_top1:.1f} < 우리 {o_top1:.1f} ({r_top1/o_top1:.0%}). "
                   f"틀린 q가 일부 깎음. 손수 검증으로 노이즈 제거 시 +0.5~1 회복 가능하나 BlueLink급은 아님. "
                   f"per-tissue서 vessel/nerve가 특히 낮으면 thin-structure q-노이즈 확증.")
    else:
        verdict = (f"🔴 심한 q-노이즈 또는 큰 도메인갭 — Rohen {r_top1:.1f} ≪ 우리 {o_top1:.1f} ({r_top1/o_top1:.0%}). "
                   f"Rohen exemplar가 자기 클래스로도 인식 안 됨 → 손수 검증(q 정제)이 효과의 전제. 다만 도메인갭이면 정제해도 한계.")
    print(f"\n  → {verdict}")
    Path("data/rohen/qcheck.json").write_text(json.dumps({
        "rohen_top1": round(r_top1, 1), "rohen_top5": round(r_top5, 1),
        "our_top1": round(o_top1, 1), "our_top5": round(o_top5, 1),
        "ratio_top1": round(r_top1 / o_top1, 2), "n_matched": len(mi),
        "per_tissue_top1": {t: p for t, (c, n, p) in pt.items()}}, ensure_ascii=False, indent=2))
    print("wrote -> data/rohen/qcheck.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
