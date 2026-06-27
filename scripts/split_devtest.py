"""Canonical FIXED dev/test split for data/merged_final — the sealed test (CLAUDE.md §1.7).

A photo-twin BLOCK split (exact∪corr≥0.90 — leak-safe) assigns 20% of blocks to a SEALED TEST
that is created ONCE (fixed seed) and reused by every experiment from exp 043 on, so the test
is never leaked through repeated method/HP selection. Selection & tuning happen on dev (10-seed
CV); final numbers are reported once on the sealed test.

    .venv/bin/python scripts/split_devtest.py        # create/print the split
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from eval_merged import load, photo_blocks, BASE  # noqa: E402

TEST_FRAC = 0.20
SEED = 20260628  # FIXED — do not change, or the sealed test leaks across experiments


def get_split():
    """Return {image: 'dev'|'test'} (cached in data/merged_final/_split.json)."""
    sp = BASE / "_split.json"
    if sp.exists():
        return json.loads(sp.read_text())
    rows = load()
    images = sorted(set(r["image"] for r in rows))
    bc = BASE / "_blocks.json"
    if bc.exists():
        img_block = json.loads(bc.read_text())
    else:
        print("computing photo-twin blocks...")
        img_block = photo_blocks(images)
        bc.write_text(json.dumps(img_block))
    blocks = sorted(set(img_block.values()))
    rng = np.random.default_rng(SEED); rng.shuffle(blocks)
    nt = max(1, int(round(len(blocks) * TEST_FRAC)))
    test_blocks = set(blocks[:nt])
    split = {im: ("test" if img_block[im] in test_blocks else "dev") for im in images}
    sp.write_text(json.dumps(split))
    return split


if __name__ == "__main__":
    s = get_split()
    rows = load()
    by = collections.Counter(s[r["image"]] for r in rows)
    print(f"blocks→ dev/test split (seed {SEED}, test {int(TEST_FRAC*100)}%)")
    print(f"  images: {collections.Counter(s.values())}")
    print(f"  triples: {dict(by)}")
