"""SCALPEL real-data pipeline (HANDOUT v2 — replaces the synthetic renderer).

Turns real BlueLink QuizLink dissection PDFs into ``(I, q, y)`` triples:

* :mod:`scalpel.data.crawl`   - fetch QuizLink PDFs (Playwright headless + gdown)
* :mod:`scalpel.data.parse`   - QuizLink PDF -> (clean_I, q, y)   [core]
* :mod:`scalpel.data.augment` - photometric augmentation (q follows the transform)
* :mod:`scalpel.data.loader`  - specimen-level splits -> gallery / testset
* :mod:`scalpel.data.vocab`   - label normalization + closed vocabulary

Heavy deps (playwright, gdown, fitz, cv2, pytesseract) are imported lazily inside
the functions that need them, so importing this package stays cheap.
"""

CREDIT = (
    "BlueLink, © B. Kathleen Alsup & Glenn M. Fox, University of Michigan — "
    "used for non-commercial educational purposes."
)
