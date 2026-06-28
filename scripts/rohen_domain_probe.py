"""Experiment 060 / M-rohen0 STEP 0 — Rohen Atlas domain gate (training-free, no extraction).

Before building a multi-day extraction pipeline for the Rohen *Color Atlas of Anatomy*, gate the
honest risk (027 precedent): Rohen is PROFESSIONAL publication photography; our QuizLink/BlueLink is
EDUCATIONAL. If Rohen photos sit in a separate island of DINO space, adding them to the gallery won't
match our queries (OOD) → extraction is wasted sweat.

Question: are Rohen cadaver photos CLOSE to our merged_final photos in embedding space?
Method: extract ~50 Rohen main photos (largest image/page >40KB, excluding section/MRI/CT/illustration
pages by text keywords), embed both Rohen and our images with the SAME global representation (frozen
vitb14@518 CLS token — no pin needed, fair image-level comparison). Three distances + t-SNE overlay.

    .venv/bin/python scripts/rohen_domain_probe.py
"""

from __future__ import annotations

import collections
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from pypdf import PdfReader  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

import explog  # noqa: E402
from eval_appearance import _MEAN, _STD, _git_sha  # noqa: E402
from eval_merged import load, BASE  # noqa: E402
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.perception import DinoBackbone  # noqa: E402

PDF = "data/color_atlas_of_anatomy.pdf"
EXCLUDE = ("section", "mri", "ct ", "radiograph", "horizontal section", "sagittal section",
           "coronal section", "schematic", "diagram")
N_ROHEN = 60
N_OURS = 220


def unit(Z):
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)


def extract_rohen(n_target):
    r = PdfReader(PDF)
    out = []
    for pg in range(20, min(540, len(r.pages))):
        try:
            page = r.pages[pg]
            txt = (page.extract_text() or "").lower()
            if any(k in txt for k in EXCLUDE):
                continue
            imgs = sorted(page.images, key=lambda im: len(im.data), reverse=True)
            if not imgs or len(imgs[0].data) < 40000:
                continue
            im = imgs[0].image.convert("RGB")
            if min(im.size) < 200:           # too small = icon/legend
                continue
            out.append((pg, im))
        except Exception:
            continue
    # sample evenly across the page range (region diversity)
    if len(out) > n_target:
        idx = np.linspace(0, len(out) - 1, n_target).astype(int)
        out = [out[i] for i in idx]
    return out


@torch.no_grad()
def embed_cls(images, bb, device):
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)
    S = 518
    Z = []
    for im in images:
        arr = np.asarray(im.convert("RGB").resize((S, S)), np.float32) / 255.0
        x = (torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device) - mean) / std
        _, cls = bb(x)
        Z.append(torch.nn.functional.normalize(cls[0], dim=0).cpu().numpy())
    return np.stack(Z).astype(np.float32)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = PipelineCfg()
    rows = load()
    our_imgs = sorted(set(r["image"] for r in rows))
    rng = np.random.default_rng(0); rng.shuffle(our_imgs)
    our_imgs = our_imgs[:N_OURS]
    print("extracting Rohen photos (excluding section/MRI/illustration)...")
    rohen = extract_rohen(N_ROHEN)
    print(f"  Rohen photos: {len(rohen)} (pages {rohen[0][0]}..{rohen[-1][0]})")

    bb = DinoBackbone(cfg.backbone); bb.ensure_loaded(); bb.to(device)
    print("embedding (CLS, vitb14@518)...")
    z_rohen = unit(embed_cls([im for _, im in rohen], bb, device))
    z_ours = unit(embed_cls([Image.open(BASE / im) for im in our_imgs], bb, device))

    # ---- three distances ----
    def mean_offdiag(A, B, same):
        S = A @ B.T
        if same:
            n = S.shape[0]; S = S[~np.eye(n, dtype=bool)].reshape(n, n - 1)
        return float(S.mean())
    d_ours = mean_offdiag(z_ours, z_ours, True)
    d_rohen = mean_offdiag(z_rohen, z_rohen, True)
    d_cross = mean_offdiag(z_rohen, z_ours, False)

    # ---- nearest-neighbour domain purity ----
    allz = np.concatenate([z_ours, z_rohen]); src = np.array([0] * len(z_ours) + [1] * len(z_rohen))
    Sall = allz @ allz.T; np.fill_diagonal(Sall, -9)
    base_rate = len(z_ours) / len(allz)            # P(a random neighbour is ours)
    # for each Rohen photo: fraction of its 20-NN that are OURS (vs base_rate)
    ro_idx = np.where(src == 1)[0]
    frac_ours_nn = []
    nn_is_ours = 0
    for i in ro_idx:
        nn = np.argsort(-Sall[i])[:20]
        frac_ours_nn.append(np.mean(src[nn] == 0))
        nn_is_ours += int(src[np.argmax(Sall[i])] == 0)
    frac_ours_nn = float(np.mean(frac_ours_nn))
    nn_purity = nn_is_ours / len(ro_idx)           # fraction of Rohen whose #1 NN is ours

    # gate
    ratio = d_cross / d_ours
    if ratio >= 0.92 and frac_ours_nn >= 0.6 * base_rate:
        gate, tag = "GO", "🟢"
        verdict = (f"🟢 도메인 일치 — Rohen↔ours {d_cross:.3f} ≈ ours↔ours {d_ours:.3f} (비 {ratio:.2f}), "
                   f"Rohen의 20-NN 중 ours 비율 {frac_ours_nn:.2f}(base {base_rate:.2f}). STEP 1 추출 진행.")
    elif ratio >= 0.85:
        gate, tag = "NARROW", "🟡"
        verdict = (f"🟡 갭 있으나 활용 가능 — Rohen↔ours {d_cross:.3f} vs ours {d_ours:.3f} (비 {ratio:.2f}), "
                   f"부분 겹침(NN-ours {frac_ours_nn:.2f}/base {base_rate:.2f}). STEP 1 소규모 먼저.")
    else:
        gate, tag = "STOP", "🔴"
        verdict = (f"🔴 OOD (027 재방문) — Rohen↔ours {d_cross:.3f} ≪ ours↔ours {d_ours:.3f} (비 {ratio:.2f}), "
                   f"Rohen이 별개 섬(NN-ours {frac_ours_nn:.2f} ≪ base {base_rate:.2f}). 추출 중단 — gallery 추가해도 매칭 안 됨.")
    print(f"\n== domain distances (CLS cosine) ==")
    print(f"  ours↔ours   {d_ours:.3f}  (in-domain reference)")
    print(f"  Rohen↔Rohen {d_rohen:.3f}")
    print(f"  Rohen↔ours  {d_cross:.3f}  (ratio to ours {ratio:.2f})")
    print(f"  Rohen 20-NN ours-fraction {frac_ours_nn:.2f} (base rate {base_rate:.2f}) | #1-NN-is-ours {nn_purity:.2f}")
    print(f"\n==> {verdict}")

    # ---- t-SNE overlay ----
    XY = TSNE(n_components=2, metric="cosine", init="pca", perplexity=20, random_state=0).fit_transform(allz)
    d = explog.EXP / "060-rohen-domain"; d.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.scatter(XY[src == 0, 0], XY[src == 0, 1], s=22, c="#1f77b4", alpha=0.6, edgecolors="#222",
               linewidths=0.3, label=f"ours (merged_final, n={len(z_ours)})")
    ax.scatter(XY[src == 1, 0], XY[src == 1, 1], s=42, c="#d62728", alpha=0.85, marker="^",
               edgecolors="#222", linewidths=0.4, label=f"Rohen Atlas (n={len(z_rohen)})")
    ax.legend(fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"060 Rohen domain gate (CLS t-SNE) — {tag} {gate}\n"
                 f"Rohen↔ours {d_cross:.2f} vs ours↔ours {d_ours:.2f} (ratio {ratio:.2f}); "
                 f"Rohen NN-ours {frac_ours_nn:.2f}/base {base_rate:.2f}")
    fig.tight_layout(); fig.savefig(d / "fig1_tsne.png", dpi=130); plt.close(fig)

    explog.bar(d / "fig2_dist.png", ["ours↔ours\n(reference)", "Rohen↔Rohen", "Rohen↔ours\n(cross)"],
               [round(d_ours, 3), round(d_rohen, 3), round(d_cross, 3)],
               "060 domain CLS-cosine distances", "mean cosine", ymax=max(d_ours, d_rohen) * 1.2, fmt="{:.3f}")

    report = f"""# 060 / M-rohen0 STEP 0 — Rohen Atlas 도메인 게이트 (학습 0, 추출 없이)

- 날짜: {__import__('datetime').date.today().isoformat()} · 커밋 `main @ {_git_sha()}` · `scripts/rohen_domain_probe.py`
- 질문: Rohen 카데바 사진이 우리 merged_final과 *임베딩 공간에서 가까운가* (gallery 추가가 작동할 도메인인가).
- 방법: Rohen 메인 사진 {len(rohen)}장(섹션/MRI/일러스트 텍스트 제외) + 우리 {len(z_ours)}장, **둘 다 vitb14@518 CLS**(핀 무관 공정 비교).

## 거리 (CLS cosine)
| 쌍 | 평균 cos |
|---|---|
| ours↔ours (in-domain 기준) | {d_ours:.3f} |
| Rohen↔Rohen | {d_rohen:.3f} |
| **Rohen↔ours (cross)** | **{d_cross:.3f}** (ours 대비 비 {ratio:.2f}) |

- Rohen 20-NN 중 ours 비율 **{frac_ours_nn:.2f}** (무작위 base rate {base_rate:.2f}); Rohen #1-NN이 ours인 비율 {nn_purity:.2f}.

![tsne](fig1_tsne.png)
![dist](fig2_dist.png)

## 판정 (사전등록 게이트)
{tag} **{gate}** — {verdict}

## 다음
- {'🟢 STEP 1 추출 파일럿 (소규모 (I,q,y) → gallery 추가 → sealed Δtop1).' if gate=='GO' else ('🟡 STEP 1 소규모 먼저, 효과 보고 확대.' if gate=='NARROW' else '🔴 추출 중단 — Rohen은 OOD, sweat 헛수고 방지. 데이터는 다른 교육용 원천(BlueLink류)에서.')}
- ⚠️ 주의: Rohen은 raw(번호/선 포함), 우리는 cleaned — 미세 confound이나 도메인(전문촬영 vs 교육용)이 지배.
"""
    explog.write(d, report, {
        "title": "Rohen 도메인 게이트 (STEP 0)", "date": __import__('datetime').date.today().isoformat(),
        "headline": f"Rohen↔ours {d_cross:.3f} vs ours↔ours {d_ours:.3f} (ratio {ratio:.2f}); "
                    f"NN-ours {frac_ours_nn:.2f}/base {base_rate:.2f} → {gate}",
        "n_rohen": len(rohen), "n_ours": len(z_ours), "gate": gate,
        "d_ours_ours": round(d_ours, 3), "d_rohen_rohen": round(d_rohen, 3), "d_cross": round(d_cross, 3),
        "ratio_cross_to_ours": round(ratio, 3), "frac_ours_in_rohen_20nn": round(frac_ours_nn, 3),
        "base_rate": round(base_rate, 3), "nn1_is_ours_frac": round(nn_purity, 3)})
    print(f"\nwrote -> {d}  (2 figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
