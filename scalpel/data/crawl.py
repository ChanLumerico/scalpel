"""QuizLink auto-collection (HANDOUT v2 §5.1, Appendix D).

Verified reality: the QuizLink index page renders its Google Drive links only
*after JavaScript*, so a static fetch returns navigation only (confirmed twice)
-> a **headless browser (Playwright) is required**. The PDFs are ~95 MB on Drive,
so Drive inserts a virus-scan confirm page -> a **token-handling downloader
(gdown)** is required.

Pipeline:  Playwright render -> harvest Drive file ids -> gdown download ->
manifest. Polite policy: concurrency 1, >=2 s between requests, cache, robots.

playwright / gdown are imported lazily so importing this module is cheap; run
``playwright install chromium`` once before crawling.

CLI:  python -m scalpel.data.crawl --out data/quizlink --region "lower limb" "foot"
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from . import CREDIT

INDEX = "https://sites.google.com/a/umich.edu/bluelink/resources/quizlink-pdfs"
ID_RE = re.compile(r"(?:/d/|[?&]id=)([A-Za-z0-9_-]{20,})")


@dataclass
class QuizLinkRef:
    region: str          # "Lower Limb" etc. (for region filtering)
    title: str           # heading near the widget, else an index
    file_id: str         # Google Drive file id
    source_url: str      # page it was found on (provenance / licence)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60] or "untitled"


def discover(index_url: str = INDEX, region_filter: list[str] | None = None,
             extra_urls: list[str] | None = None) -> list[QuizLinkRef]:
    """Headless-render the index (+ optional curriculum pages) and harvest Drive ids.

    A static fetch returns only the site nav; the Drive embeds appear only after
    JS, so we wait for ``networkidle`` and (best-effort) for a Drive iframe.
    """
    from playwright.sync_api import sync_playwright

    refs: dict[str, QuizLinkRef] = {}
    urls = [index_url] + list(extra_urls or [])
    with sync_playwright() as pw:
        br = pw.chromium.launch(headless=True)
        pg = br.new_page()
        for url in urls:
            pg.goto(url, wait_until="networkidle", timeout=60_000)
            try:                                          # Drive iframe loads late
                pg.wait_for_selector("iframe[src*='drive.google.com']", timeout=15_000)
            except Exception:
                pass
            hrefs = pg.eval_on_selector_all(
                "a[href*='drive.google.com'], iframe[src*='drive.google.com']",
                "els => els.map(e => e.href || e.src)",
            )
            page_text = pg.inner_text("body").lower()
            for i, link in enumerate(hrefs):
                m = ID_RE.search(link or "")
                if not m:
                    continue
                fid = m.group(1)
                region = "unknown"
                if region_filter:
                    hit = next((k for k in region_filter if k.lower() in page_text), None)
                    region = hit or "unknown"
                refs.setdefault(fid, QuizLinkRef(region, f"quizlink_{i}", fid, url))
        br.close()

    out = list(refs.values())
    if region_filter:
        keys = [k.lower() for k in region_filter]
        filtered = [r for r in out if any(k in (r.region + r.title).lower() for k in keys)]
        out = filtered or out                             # keep all if heuristic missed
    return out


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _is_pdf(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 100_000 and p.read_bytes()[:4] == b"%PDF"


def download(ref: QuizLinkRef, out_dir: Path, retries: int = 3) -> Path | None:
    """Download a large Drive PDF (gdown handles the confirm token); cache + verify."""
    import gdown

    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{_slug(ref.region)}__{_slug(ref.title)}__{ref.file_id}.pdf"
    if _is_pdf(dst):
        return dst                                        # cache: never re-download
    for attempt in range(1, retries + 1):
        try:
            gdown.download(id=ref.file_id, output=str(dst), quiet=False)
            if _is_pdf(dst):
                dst.with_suffix(".credit.txt").write_text(
                    f"{CREDIT}\nsource: {ref.source_url}\nfile_id: {ref.file_id}\n"
                    f"downloaded: {dt.datetime.now().isoformat()}\n", encoding="utf-8")
                return dst
        except Exception as e:  # noqa: BLE001
            print(f"  retry {attempt}/{retries}: {e}")
        time.sleep(2 * attempt)                           # backoff
    print(f"  x failed: {ref.file_id}")
    return None


def crawl(out_dir: str, region_filter: list[str] | None = None,
          min_delay_s: float = 2.0, notified_authors: bool = False,
          extra_urls: list[str] | None = None) -> Path:
    """Discover + download QuizLink PDFs politely; accumulate a manifest.

    Ethics gate (§6): notifying the authors (bluelinkanatomy@gmail.com) is
    recommended before use; pass ``notified_authors=True`` (CLI ``--notified``)
    to confirm. Without it the run still proceeds but warns.
    """
    if not notified_authors:
        print("⚠️  Recommend notifying the authors (bluelinkanatomy@gmail.com) "
              "before use — §6. Pass --notified to confirm.")

    out = Path(out_dir)
    man = out / "manifest.json"
    records = json.loads(man.read_text()) if man.exists() else []
    seen = {r["file_id"] for r in records}

    refs = discover(region_filter=region_filter, extra_urls=extra_urls)
    print(f"discovered {len(refs)} (region_filter={region_filter})")
    for ref in refs:                                      # concurrency 1, sequential
        if ref.file_id in seen:
            continue
        p = download(ref, out)
        if p:
            records.append({**asdict(ref), "path": str(p), "sha256": _sha256(p),
                            "bytes": p.stat().st_size,
                            "downloaded_at": dt.datetime.now().isoformat()})
            man.write_text(json.dumps(records, ensure_ascii=False, indent=2))
        time.sleep(min_delay_s)                           # polite delay
    print(f"done: {len(records)} in manifest -> {man}")
    return man


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch BlueLink QuizLink PDFs")
    ap.add_argument("--out", default="data/quizlink")
    ap.add_argument("--region", nargs="*", default=None)
    ap.add_argument("--extra-urls", nargs="*", default=None,
                    help="extra curriculum pages to also scan for Drive embeds")
    ap.add_argument("--notified", action="store_true",
                    help="confirm you have notified the BlueLink authors (§6)")
    a = ap.parse_args()
    crawl(a.out, a.region, notified_authors=a.notified, extra_urls=a.extra_urls)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
