#!/usr/bin/env python3
"""Extract and hash-lock the prespecified Efron passage from its source PDF."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re

from pypdf import PdfReader

from efron_contract import (
    ROOT,
    SOURCE_EXCERPT_SHA256,
    SOURCE_PARAGRAPHS,
    SOURCE_PDF_BASENAME,
    SOURCE_PDF_SHA256,
    SOURCE_PHYSICAL_PAGE,
    SOURCE_WORDS,
)


DEFAULT_OUTPUT = ROOT / "efron_excerpt.txt"
PARAGRAPH_STARTS = (
    "False discovery rates, Benjamini and Hochberg’s seminal contribution,",
    "The later chapters are at pains to show the limitations",
    "In moving beyond the confines of classical statistics,",
    "The classical era of statistics can itself be divided into two periods:",
)
FINAL_SENTENCE = "general, being applicable to both estimation and hypothesis testing."
TYPESETTING_REPAIRS = {
    "large-\nscale": "large-scale",
    "anal-\nysis": "analysis",
    "exam-\nples": "examples",
    "rep-\nresented": "represented",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def extract_excerpt(page_text: str) -> str:
    """Select the four locked paragraphs and repair PDF line wrapping."""

    text = page_text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    excerpt_start = text.find(PARAGRAPH_STARTS[0])
    excerpt_end_start = text.find(FINAL_SENTENCE, excerpt_start)
    if excerpt_start < 0 or excerpt_end_start < 0:
        raise RuntimeError("the frozen passage endpoints were not found")
    text = text[excerpt_start : excerpt_end_start + len(FINAL_SENTENCE)]
    for broken, repaired in TYPESETTING_REPAIRS.items():
        if text.count(broken) != 1:
            raise RuntimeError(f"expected PDF hyphenation not found exactly once: {broken!r}")
        text = text.replace(broken, repaired)
    if re.search(r"[A-Za-z]-\s*\n\s*[a-z]", text):
        raise RuntimeError("unreviewed line-end hyphenation remains")

    starts = [text.find(marker) for marker in PARAGRAPH_STARTS]
    if any(position < 0 for position in starts) or starts != sorted(starts):
        raise RuntimeError("the four paragraph starts were not found in order")
    end = text.find(FINAL_SENTENCE, starts[-1]) + len(FINAL_SENTENCE)
    boundaries = starts[1:] + [end]
    paragraphs = [
        re.sub(r"\s+", " ", text[start:stop]).strip()
        for start, stop in zip(starts, boundaries, strict=True)
    ]
    return "\n\n".join(paragraphs) + "\n"


def validate_excerpt(text: str) -> None:
    observed = (
        len(text.rstrip("\n").split("\n\n")),
        len(text.split()),
        sha256_bytes(text.encode("utf-8")),
    )
    expected = (SOURCE_PARAGRAPHS, SOURCE_WORDS, SOURCE_EXCERPT_SHA256)
    if observed != expected:
        raise RuntimeError(f"frozen excerpt mismatch: {observed!r} != {expected!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    raw_pdf = args.pdf.read_bytes()
    if args.pdf.name != SOURCE_PDF_BASENAME:
        raise RuntimeError(f"source PDF must be named {SOURCE_PDF_BASENAME}")
    if sha256_bytes(raw_pdf) != SOURCE_PDF_SHA256:
        raise RuntimeError("source PDF hash mismatch")

    reader = PdfReader(args.pdf)
    if len(reader.pages) != 3:
        raise RuntimeError(f"source PDF page count changed: {len(reader.pages)} != 3")
    excerpt = extract_excerpt(reader.pages[SOURCE_PHYSICAL_PAGE - 1].extract_text())
    validate_excerpt(excerpt)
    args.output.write_text(excerpt, encoding="utf-8")
    print(f"verified Efron excerpt written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
