"""Experiment logging — one folder per experiment with a Korean report + figures.

    experiments/NNN-<name>/
      report.md          한글 리포트 (목적 / 설정 / 결과 / 해석 / 다음)
      metrics.json       raw 수치 (인덱스 재생성용 headline 포함)
      *.png              차트 (이미지 없음 -> 커밋 가능)
      *.private.png      카데바 이미지가 들어간 figure -> .gitignore (donor dignity §6)

Figure 라벨은 폰트 깨짐 방지를 위해 ASCII만 사용하고, 한글은 report.md 에만 둔다.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"


def next_dir(name: str) -> Path:
    """Create the next experiments/NNN-name/ folder and return it."""
    EXP.mkdir(exist_ok=True)
    nums = [int(m.group(1)) for p in EXP.glob("[0-9]*-*")
            if (m := re.match(r"(\d+)-", p.name))]
    n = (max(nums) + 1) if nums else 1
    d = EXP / f"{n:03d}-{name}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def bar(path, labels, values, title, ylabel, ymax=None, fmt="{:.1f}", errors=None):
    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 1.1), 4))
    bars = ax.bar([str(x) for x in labels], values, color="#3b7dd8",
                  yerr=errors, capsize=5)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                ha="center", va="bottom", fontsize=9)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ymax:
        ax.set_ylim(0, ymax)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def grouped_bar(path, groups, series, title, ylabel, ymax=None):
    fig, ax = plt.subplots(figsize=(max(5, len(groups) * 1.4), 4))
    n = len(series)
    w = 0.8 / n
    x = np.arange(len(groups))
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in groups])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ymax:
        ax.set_ylim(0, ymax)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def barh_pairs(path, pairs, title, xlabel="count"):
    """Horizontal bar of (label, count) pairs (e.g. top confusions). ASCII labels."""
    if not pairs:
        return
    labels = [p[0] for p in pairs][::-1]
    vals = [p[1] for p in pairs][::-1]
    fig, ax = plt.subplots(figsize=(8, max(3, len(pairs) * 0.5)))
    ax.barh(labels, vals, color="#d8743b")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def lineplot(path, curves, title, xlabel, ylabel, xlim=None, ylim=None,
             diagonal=False, hline=None):
    """curves = list of (label, xs, ys[, ystd]). diagonal -> y=x ideal line."""
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    if diagonal:
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="ideal (calibrated)")
    if hline is not None:
        ax.axhline(hline[0], ls=":", color="crimson", lw=1, label=hline[1])
    for c in curves:
        label, xs, ys = c[0], c[1], c[2]
        line, = ax.plot(xs, ys, marker="o", ms=3, label=label)
        if len(c) > 3 and c[3] is not None:
            ys, ystd = np.array(ys), np.array(c[3])
            ax.fill_between(xs, ys - ystd, ys + ystd, alpha=0.15, color=line.get_color())
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def montage(path, examples, base, ncol=4):
    """Grid of predictions (CONTAINS cadaver imagery -> name it *.private.png).

    examples: list of dict(image, q=(x,y), true, pred, correct).
    """
    n = len(examples)
    if n == 0:
        return
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3, nrow * 3.3))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, ex in zip(axes, examples):
        im = np.array(Image.open(base / ex["image"]).convert("RGB"))
        x, y = ex["q"]
        ax.imshow(im)
        ax.scatter([x], [y], s=140, c="cyan", edgecolors="red", linewidths=2)
        ok = ex["correct"]
        ax.set_title(f"{'O' if ok else 'X'} T:{ex['true']}\nP:{ex['pred']}",
                     color=("green" if ok else "red"), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write(d: Path, report_md: str, metrics: dict):
    (d / "report.md").write_text(report_md, encoding="utf-8")
    (d / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    update_index()


def update_index():
    lines = ["# 실험 인덱스 (experiments)", "",
             "각 실험은 서브폴더에 한글 `report.md` + figure + `metrics.json`.", "",
             "| ID | 제목 | 날짜 | 핵심 결과 |", "|---|---|---|---|"]
    for p in sorted(EXP.glob("[0-9]*-*")):
        mj = p / "metrics.json"
        if not mj.exists():
            continue
        m = json.loads(mj.read_text(encoding="utf-8"))
        lines.append(f"| [{p.name}]({p.name}/report.md) | {m.get('title','')} | "
                     f"{m.get('date','')} | {m.get('headline','')} |")
    (EXP / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
