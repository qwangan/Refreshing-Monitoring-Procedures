"""Fail-closed constants for the prespecified Efron mixed document."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL_NAME = "facebook/opt-1.3b"
MODEL_REVISION = "3f5c25d0bc631cb57ac65913f76e22c2dfb61d62"
MODEL_DTYPE = "torch.float32"
MODEL_PARAMETER_COUNT = 1_315_758_080
MODEL_VOCAB_SIZE = 50_272

SOURCE_URL = "Bradley Efron (2010), Large-Scale Inference, Prologue, pp. x-xi"
SOURCE_TITLE = "Large-Scale Inference: Empirical Bayes Methods for Estimation, Testing, and Prediction"
SOURCE_AUTHOR = "Bradley Efron"
SOURCE_DATE = "2010"
SOURCE_PARAGRAPHS = 4
SOURCE_WORDS = 365
SOURCE_NORMALIZATION = "PDF prose, dehyphenated line wraps, typographic apostrophe, four paragraphs, terminal LF, v1"
SOURCE_PDF_SHA256 = "0cf0c89e58c6ec008a08e46972e8d34e8e433dc3e2d86148f99b5b683309787d"
SOURCE_EXCERPT_SHA256 = "9ff471d30727dd95510a6374f43b6280deaa375657e67e26bf6233f07a7e1e63"
SOURCE_OPT_TOKEN_COUNT = 443
SOURCE_OPT_TOKEN_SHA256 = "52772f30d7f3a1a4518213a358b8d43e2e48779afed389973884381966ca86ec"

PROMPT = "Large-Scale Inference\nBradley Efron\n2010\n\n"
CASE_ID = "efron-prologue-balanced-sentence-tournament-opt13b-v4"
CASE_MASTER_SEED = 20260922000100
BIT_GENERATOR = "PCG64DXSM"
TABLE_STREAM = 0
ORDINARY_STREAM = 1
PIVOT_STREAM = 2
TOURNAMENT_SAMPLE_STREAM = 3
TEMPERATURE = 1.0
WATERMARK_BLOCKS = 2

THRESHOLD_EXACT = 49
THRESHOLD_DISPLAY = 49
SWZ_CAP = 0.5
DETECTOR_STRATEGY = "adaptive_cumulative"
LAYERS = 30

ALLOWED_METRICS = (
    "reports",
    "token_power",
    "token_fdp",
    "token_iou",
    "final_report_fdp",
    "report_ufdp",
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def validate_source(source_path: Path | None = None) -> str:
    source_path = source_path or ROOT / "efron_excerpt.txt"
    text = source_path.read_text(encoding="utf-8")
    if sha256_text(text) != SOURCE_EXCERPT_SHA256:
        raise RuntimeError("normalized Efron passage hash changed")
    if len(text.split()) != SOURCE_WORDS:
        raise RuntimeError("normalized Efron word count changed")
    return text
