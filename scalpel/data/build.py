"""Build a structured QuizLink triple dataset (HANDOUT v2 §3, §5).

Parses every QuizLink PDF into ``(clean_I, q, y)`` triples and writes them to
disk in a systematic, reload-friendly layout:

    out/
      images/<pdf-slug>__p<page>.png   clean dissection photo (one per page, deduped)
      triples.jsonl                    one JSON object per (I, q, y) triple
      vocab.json                       closed vocabulary  name -> id

Each JSONL record::

    {"image": "images/quizlink-1__p012.png", "q": [855, 197],
     "label": "deltoid muscle", "label_id": 5, "page": 12,
     "src": "quizlink-1.pdf", "w": 1700, "h": 1275}

Run (inside the project venv, with the tesseract binary on PATH)::

    python -m scalpel.data.build --pdf-dir data/quizlink --out data/triples
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .parse import parse_quizlink
from .vocab import Vocab


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:50] or "pdf"


def build(pdf_dir: str, out_dir: str, vocab: Vocab | None = None) -> dict:
    from PIL import Image

    vocab = vocab or Vocab()
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    pdfs = sorted(Path(pdf_dir).glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"no PDFs under {pdf_dir}")

    n_tri, n_pdf_ok, saved = 0, 0, set()
    with (out / "triples.jsonl").open("w", encoding="utf-8") as f:
        for pdf in pdfs:
            try:
                triples = parse_quizlink(pdf, vocab)
            except Exception as e:  # noqa: BLE001
                print(f"  skip {pdf.name}: {type(e).__name__}: {e}")
                continue
            if triples:
                n_pdf_ok += 1
            for t in triples:
                key = f"{_slug(pdf.stem)}__p{t.page:03d}"
                rel = f"images/{key}.png"
                if key not in saved:
                    Image.fromarray(t.image).save(out / rel)
                    saved.add(key)
                rec = {
                    "image": rel, "q": [int(t.q[0]), int(t.q[1])], "label": t.label,
                    "label_id": vocab.index(t.label), "page": int(t.page),
                    "src": pdf.name, "w": int(t.image.shape[1]), "h": int(t.image.shape[0]),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_tri += 1
            print(f"  {pdf.name}: {len(triples)} triples")
    vocab.save(out / "vocab.json")
    stats = {"pdfs": len(pdfs), "pdfs_with_triples": n_pdf_ok, "triples": n_tri,
             "images": len(saved), "vocab": len(vocab)}
    print(f"\nDONE: {stats} -> {out}/triples.jsonl")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Build QuizLink (I,q,y) triple dataset")
    ap.add_argument("--pdf-dir", default="data/quizlink")
    ap.add_argument("--out", default="data/triples")
    a = ap.parse_args()
    build(a.pdf_dir, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
