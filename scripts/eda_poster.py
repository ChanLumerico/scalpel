"""Comprehensive EDA poster for the QuizLink (I,q,y) dataset (stats only, no imagery).

    .venv/bin/python scripts/eda_poster.py   ->  outputs/eda_poster.png (high-res)
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

ROWS = [json.loads(l) for l in open("data/triples/triples.jsonl", encoding="utf-8") if l.strip()]
Y = [r["label"] for r in ROWS]
cnt = collections.Counter(Y)
per_img = collections.Counter(r["image"] for r in ROWS)
per_src = collections.Counter(r["src"] for r in ROWS)
inst = np.array(sorted(cnt.values(), reverse=True))
core = sum(1 for v in cnt.values() if v >= 2)
singles = sum(1 for v in cnt.values() if v == 1)

TISSUE = [("muscle", "muscle"), ("nerve", "nerve"), ("artery", "artery"), ("vein", "vein"),
          ("bone", r"bone|skull|vertebra|rib|mandible|maxilla|tibia|fibula|femur|sacrum|coccyx|"
                   r"scapula|clavicle|sternum|humerus|patella|ilium|pubis|ischium"),
          ("ligament/tendon", "ligament|tendon"), ("gland", "gland"),
          ("sinus/duct/cavity", "sinus|duct|cavity|canal|foramen|fossa"),
          ("joint/cartilage", "joint|cartilage")]


def tissue(lab):
    for name, pat in TISSUE:
        if re.search(pat, lab):
            return name
    return "other"


tis = collections.Counter(tissue(l) for l in Y)

# pin positions normalized to the photo frame
qx = np.array([r["q"][0] / max(1, r["w"]) for r in ROWS])
qy = np.array([r["q"][1] / max(1, r["h"]) for r in ROWS])

plt.rcParams.update({"font.size": 11, "axes.titlesize": 14, "axes.titleweight": "bold"})
fig = plt.figure(figsize=(24, 15), dpi=200)
gs = GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.26,
              top=0.88, bottom=0.06, left=0.05, right=0.97)
fig.suptitle("SCALPEL — QuizLink point-conditioned anatomy dataset: EDA",
             fontsize=26, fontweight="bold", y=0.965)
fig.text(0.5, 0.915,
         f"{len(ROWS)} triples  ·  {len(cnt)} structures  ·  {len(per_img)} images  ·  "
         f"{len(per_src)} specimen PDFs   |   evaluable core (≥2): {sum(cnt[k] for k in cnt if cnt[k]>=2)} "
         f"triples / {core} classes   |   singletons: {singles} ({100*singles/len(cnt):.0f}% of classes)",
         ha="center", fontsize=14, color="#333")

# 1. instances-per-class (long tail), log y
ax = fig.add_subplot(gs[0, 0])
ax.bar(range(len(inst)), inst, width=1.0, color="#3b7dd8")
ax.set_yscale("log"); ax.set_title("(1) Long tail: instances per class")
ax.set_xlabel("class rank"); ax.set_ylabel("# instances (log)")
ax.axhline(2, ls="--", color="crimson", lw=1)
ax.text(len(inst)*0.4, 2.2, f"≥2 = evaluable core ({core} classes)", color="crimson", fontsize=10)

# 2. histogram of instances/class
ax = fig.add_subplot(gs[0, 1])
h = collections.Counter(cnt.values())
ks = sorted(h)[:12]
ax.bar([str(k) for k in ks], [h[k] for k in ks], color="#d8743b")
ax.set_title("(2) How many classes have N instances")
ax.set_xlabel("instances per class"); ax.set_ylabel("# classes")
for k in ks:
    ax.text(ks.index(k), h[k], str(h[k]), ha="center", va="bottom", fontsize=9)

# 3. labels (leaders) per image
ax = fig.add_subplot(gs[0, 2])
lp = collections.Counter(per_img.values())
ks = sorted(lp)
ax.bar([str(k) for k in ks], [lp[k] for k in ks], color="#5aa469")
ax.set_title(f"(3) Labels per image (mean {np.mean(list(per_img.values())):.2f}, max {max(per_img.values())})")
ax.set_xlabel("# pins on one photo"); ax.set_ylabel("# images")

# 4. triples per specimen PDF
ax = fig.add_subplot(gs[1, 0])
vals = sorted(per_src.values(), reverse=True)
ax.bar(range(len(vals)), vals, color="#7b5ea7")
ax.set_title(f"(4) Triples per specimen PDF (31 total)")
ax.set_xlabel("PDF rank"); ax.set_ylabel("# triples")
ax.axhline(np.mean(vals), ls="--", color="k", lw=1)
ax.text(len(vals)*0.5, np.mean(vals)+1, f"mean {np.mean(vals):.0f}", fontsize=10)

# 5. tissue-type composition
ax = fig.add_subplot(gs[1, 1])
order = tis.most_common()
ax.barh([t for t, _ in order][::-1], [c for _, c in order][::-1], color="#cc5577")
ax.set_title("(5) Tissue-type composition (by label keyword)")
ax.set_xlabel("# triples")
for i, (t, c) in enumerate(order[::-1]):
    ax.text(c, i, f" {c} ({100*c/len(ROWS):.0f}%)", va="center", fontsize=9)

# 6. pin spatial heatmap
ax = fig.add_subplot(gs[1, 2])
hb = ax.hist2d(qx, qy, bins=30, range=[[0, 1], [0, 1]], cmap="inferno")
ax.invert_yaxis(); ax.set_title("(6) Pin location density (normalized photo frame)")
ax.set_xlabel("x / width"); ax.set_ylabel("y / height")
fig.colorbar(hb[3], ax=ax, fraction=0.046, pad=0.04)

# 7. top-22 structures
ax = fig.add_subplot(gs[2, 0])
top = cnt.most_common(22)
ax.barh([t for t, _ in top][::-1], [c for _, c in top][::-1], color="#3b7dd8")
ax.set_title("(7) Most frequent structures (top 22)")
ax.set_xlabel("# instances"); ax.tick_params(axis="y", labelsize=8)

# 8. cumulative coverage by class rank
ax = fig.add_subplot(gs[2, 1])
cum = np.cumsum(inst) / inst.sum() * 100
ax.plot(range(1, len(cum)+1), cum, color="#d8743b", lw=2)
ax.set_title("(8) Cumulative coverage by class rank")
ax.set_xlabel("top-N classes"); ax.set_ylabel("% of all triples")
ax.grid(alpha=0.3)
for frac in (50, 80):
    n = int(np.searchsorted(cum, frac)) + 1
    ax.axhline(frac, ls=":", color="gray", lw=1); ax.axvline(n, ls=":", color="gray", lw=1)
    ax.text(n+3, frac-6, f"{frac}% by top {n}", fontsize=9)

# 9. core vs singleton (donut) + key facts
ax = fig.add_subplot(gs[2, 2])
ax.pie([core, singles], labels=[f"≥2 core\n{core} cls", f"singleton\n{singles} cls"],
       colors=["#3b7dd8", "#cccccc"], autopct="%1.0f%%", startangle=90,
       wedgeprops=dict(width=0.42), textprops=dict(fontsize=11))
ax.set_title("(9) Classes: evaluable core vs singletons")

out = Path("outputs"); out.mkdir(exist_ok=True)
fig.savefig(out / "eda_poster.png", dpi=200, bbox_inches="tight", facecolor="white")
print("saved outputs/eda_poster.png", fig.get_size_inches())
