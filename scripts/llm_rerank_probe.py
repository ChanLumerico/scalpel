"""Experiment 059 — M-llm0: bare Vision-LLM gate (Qwen2.5-VL-7B-4bit via MLX), no Atlas.

Question: does Qwen2.5-VL see the pinned structure in a dissection photo at all? It re-ranks the
DINO top-k candidates → top1. Gate BEFORE any expensive Atlas knowledge-grounding.

rev2 design (from the XAI poster):
  • candidates = DINO global+L256+CSLS top-K (the current best). recall@K is the LLM CEILING.
  • K10 (poster panel 16: most misses rank >5 → top-5 too shallow; top-10 lifts recall).
  • V_both: global image with a red marker at q  +  the L256 high-res crop (poster panel 7).
  • BLIND control (no image): does the LLM SEE the photo or just guess from candidate names? (027 trap)
    vision_gain = with-image − blind MUST be > 0, else the LLM isn't using the image.

Sealed test (seed 20260628), greedy decode, local/free. Pre-registered gate (see RESEARCH).
    .venv/bin/python scripts/llm_rerank_probe.py --limit 40        # pilot
    .venv/bin/python scripts/llm_rerank_probe.py --limit 0         # full sealed
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from eval_merged import load, BASE  # noqa: E402
from split_devtest import get_split  # noqa: E402
from multiscale_local import crop_pad  # noqa: E402

MODEL = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
SCRATCH = Path("/private/tmp/claude-501/-Users-chanlee-Desktop-Programming-scalpel/320890f6-79e3-48e4-86bc-86ffdb842a81/scratchpad/llm_imgs")
BASELINE = 38.3

SYSTEM = ("You are an expert anatomist analyzing cadaveric dissection photographs. You identify the "
          "anatomical structure indicated by a marker. Choose exactly one answer from the candidate list. "
          "Reason from visible evidence: the structure's location in the specimen, its shape/texture/colour "
          "at the marker (vessel lumen+wall, nerve cable+fascicle, muscle fibre, bone), and its spatial "
          "relationship to neighbours. Never propose a structure not in the list. If unsure, still pick the "
          "single most likely candidate.")


def unit(Z):
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)


def csls_rank(Z, Y, dev, test):
    """Per covered sealed query: (idx, truth, ranked_class_list)."""
    labset = set(Y[i] for i in dev); labs = sorted(labset); li = {l: j for j, l in enumerate(labs)}
    cov = [q for q in test if Y[q] in labset]
    sqg = Z[cov] @ Z[dev].T; gg = Z[dev] @ Z[dev].T; np.fill_diagonal(gg, -9)
    k = 5; rg = np.sort(gg, 1)[:, -k:].mean(1); rq = np.sort(sqg, 1)[:, -k:].mean(1)
    S = 2 * sqg - rq[:, None] - rg[None, :]
    cols = collections.defaultdict(list)
    for j, i in enumerate(dev):
        cols[li[Y[i]]].append(j)
    sc = np.full((len(cov), len(labs)), -9.0, np.float32)
    for c, ix in cols.items():
        sc[:, c] = S[:, ix].max(1)
    order = np.argsort(-sc, 1)
    out = []
    for r, q in enumerate(cov):
        ranked = [labs[order[r, t]] for t in range(len(labs))]
        out.append((q, Y[q], ranked))
    return out


def make_images(idx, rows):
    SCRATCH.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(Image.open(BASE / rows[idx]["image"]).convert("RGB"))
    qx, qy = rows[idx]["q"]
    H, W = arr.shape[:2]
    sc = 896 / max(H, W) if max(H, W) > 896 else 1.0
    disp = cv2.resize(arr, (int(W * sc), int(H * sc)))
    cx, cy = int(qx * sc), int(qy * sc)
    cv2.circle(disp, (cx, cy), max(6, int(min(disp.shape[:2]) * 0.012)), (255, 0, 0), -1)
    cv2.circle(disp, (cx, cy), max(22, int(min(disp.shape[:2]) * 0.045)), (255, 0, 0), 2)
    gp = SCRATCH / f"g{idx}.png"; Image.fromarray(disp).save(gp)
    crop = cv2.resize(crop_pad(arr, qx, qy, 256), (384, 384))
    cpath = SCRATCH / f"c{idx}.png"; Image.fromarray(crop).save(cpath)
    return str(gp), str(cpath)


def build_prompt(cands, blind=False):
    clist = "\n".join(f"{i+1}. {c}" for i, c in enumerate(cands))
    if blind:
        return (SYSTEM + "\n\nFrom this candidate list of anatomical structures (ranked by an appearance "
                "retrieval system, most-similar first), which SINGLE one is most likely the answer?\n\n"
                f"{clist}\n\nOutput your final answer on the LAST line EXACTLY as:\nANSWER: <number> <structure name>")
    return (SYSTEM + "\n\nThis is a cadaveric dissection photograph. A red dot (thin red circle) marks ONE "
            "structure. The second image is a high-resolution zoom centered on that same marked location.\n\n"
            "Identify the structure at the red dot. Choose EXACTLY ONE from these candidates (ranked by an "
            f"appearance-based retrieval system, most-similar first):\n\n{clist}\n\n"
            "Reason briefly: LOCATION (where in the specimen), APPEARANCE at the dot (vessel lumen/wall? "
            "nerve cable/fascicle? muscle fibre? bone? colour?), NEIGHBOURS (which candidate fits).\n"
            "Output your final answer on the LAST line EXACTLY as:\nANSWER: <number> <structure name>")


def parse(out, K):
    txt = out if isinstance(out, str) else getattr(out, "text", str(out))
    for line in reversed(txt.strip().splitlines()):
        m = re.search(r"ANSWER:\s*(\d+)", line)
        if m:
            n = int(m.group(1))
            return (n if 1 <= n <= K else None), txt
    m = re.search(r"\b(\d+)\b", txt.strip().splitlines()[-1]) if txt.strip() else None
    return ((int(m.group(1)) if m and 1 <= int(m.group(1)) <= K else None), txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--K", type=int, default=10)
    args = ap.parse_args()
    K = args.K

    rows = load(); Y = [r["label"] for r in rows]
    split = get_split(); cnt = collections.Counter(Y)
    core = [i for i in range(len(rows)) if cnt[Y[i]] >= 2]
    dev = [i for i in core if split[rows[i]["image"]] == "dev"]
    test = [i for i in core if split[rows[i]["image"]] == "test"]
    zg = unit(np.load(BASE / "_dino_cache.npy").astype(np.float32))
    zl = unit(np.load(BASE / "_local256_cache.npy").astype(np.float32))
    Z = unit(np.concatenate([zg, zl], 1))
    ranked = csls_rank(Z, Y, dev, test)
    if args.limit:
        ranked = ranked[:args.limit]
    n = len(ranked)
    recall = {k: 100 * np.mean([truth in r[:k] for _, truth, r in ranked]) for k in [1, 3, 5, 10]}
    print(f"sealed queries: {n} | recall@1/3/5/10 = "
          f"{recall[1]:.1f}/{recall[3]:.1f}/{recall[5]:.1f}/{recall[10]:.1f} (LLM ceiling @K{K}={recall[K]:.1f})")

    print("loading Qwen2.5-VL-7B-4bit (MLX)...")
    from mlx_vlm import load as mlx_load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    model, processor = mlx_load(MODEL)
    config = model.config

    def ask(prompt, images):
        formatted = apply_chat_template(processor, config, prompt, num_images=len(images))
        out = generate(model, processor, formatted, image=images if images else None,
                       max_tokens=400, temperature=0.0, verbose=False)
        return out

    img_ok = blind_ok = pfail = oolist = 0
    rows_log = []
    for t, (q, truth, rk) in enumerate(ranked, 1):
        cands = rk[:K]
        gp, cp = make_images(q, rows)
        # with image (V_both)
        n_img, raw = parse(ask(build_prompt(cands), [gp, cp]), K)
        pick = cands[n_img - 1] if n_img else None
        if n_img is None:
            pfail += 1
        ic = int(pick == truth) if pick else 0
        img_ok += ic
        # blind
        nb, _ = parse(ask(build_prompt(cands, blind=True), []), K)
        pb = cands[nb - 1] if nb else None
        bc = int(pb == truth) if pb else 0
        blind_ok += bc
        rows_log.append({"q": q, "truth": truth, "img_pick": pick, "img_correct": ic,
                         "blind_pick": pb, "blind_correct": bc, "truth_in_cands": truth in cands})
        if t % 10 == 0 or t == n:
            print(f"  [{t}/{n}] img {100*img_ok/t:.1f} | blind {100*blind_ok/t:.1f} | parse_fail {pfail}")

    qwen = 100 * img_ok / n; blind = 100 * blind_ok / n
    vision_gain = round(qwen - blind, 1); rerank_gain = round(qwen - BASELINE, 1)
    ceiling_frac = round(qwen / recall[K], 2) if recall[K] else 0
    print(f"\n==== M-llm0 GATE (V_both, K{K}, sealed n={n}) ====")
    print(f"  recall@{K} (ceiling)   {recall[K]:.1f}")
    print(f"  Qwen top1 (with image) {qwen:.1f}   (ceiling_frac {ceiling_frac})")
    print(f"  Qwen top1 (blind)      {blind:.1f}")
    print(f"  vision_gain (img−blind) {vision_gain:+}   {'🟢 sees image' if vision_gain > 2 else '🔴 NOT using image (language prior)'}")
    print(f"  rerank_gain (vs {BASELINE}) {rerank_gain:+}")
    print(f"  parse_fail {100*pfail/n:.0f}%")
    out = Path("experiments/059-llm-rerank"); out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({
        "title": "M-llm0 Qwen2.5-VL gate", "n_sealed": n, "K": K, "model": MODEL,
        "recall_at": recall, "qwen_top1": round(qwen, 1), "blind_top1": round(blind, 1),
        "vision_gain": vision_gain, "rerank_gain": rerank_gain, "ceiling_frac": ceiling_frac,
        "parse_fail_pct": round(100 * pfail / n, 1), "baseline": BASELINE,
        "verdict_sees_image": vision_gain > 2,
        "log": rows_log}, ensure_ascii=False, indent=2))
    print(f"\nwrote -> {out}/metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
