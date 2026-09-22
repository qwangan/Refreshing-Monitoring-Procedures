#!/usr/bin/env python3
"""Fail-closed scientific contract for the Efron mixed-text case study."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent

MODEL_NAME = "facebook/opt-1.3b"
MODEL_REVISION = "3f5c25d0bc631cb57ac65913f76e22c2dfb61d62"
MODEL_DTYPE = "torch.float32"
MODEL_PARAMETER_COUNT = 1_315_758_080
MODEL_VOCAB_SIZE = 50_272

SOURCE_PDF_BASENAME = "BradleyEfron_2010_Prologue_Large-ScaleInferenceE.pdf"
SOURCE_PDF_SHA256 = (
    "0cf0c89e58c6ec008a08e46972e8d34e8e433dc3e2d86148f99b5b683309787d"
)
SOURCE_TITLE = "Large-Scale Inference"
SOURCE_AUTHOR = "Bradley Efron"
SOURCE_DATE = "2010"
SOURCE_PRINTED_PAGE = "x"
SOURCE_PHYSICAL_PAGE = 2
SOURCE_PARAGRAPHS = 4
SOURCE_WORDS = 365
SOURCE_NORMALIZATION = "printed-page-x-four-paragraphs-pdf-v1"
SOURCE_EXCERPT_SHA256 = (
    "9ff471d30727dd95510a6374f43b6280deaa375657e67e26bf6233f07a7e1e63"
)
SOURCE_OPT_TOKEN_COUNT = 443
SOURCE_OPT_TOKEN_SHA256 = (
    "52772f30d7f3a1a4518213a358b8d43e2e48779afed389973884381966ca86ec"
)

PROMPT = "Large-Scale Inference\nBradley Efron\n2010\n\n"
CASE_ID = "efron-large-scale-inference-balanced-sentence-gumbel-opt13b-v4"
CASE_MASTER_SEED = 20260922000100
BIT_GENERATOR = "PCG64DXSM"
KEY_STREAM = 0
ORDINARY_STREAM = 1
TEMPERATURE = 1.0
WATERMARK_BLOCKS = 2

THRESHOLD_EXACT = 49
THRESHOLD_DISPLAY = 49
SWZ_CAP = 0.5
DETECTOR_STRATEGY = "adaptive_cumulative"
DETECTOR_SOURCE_SHA256 = (
    "354621e8ebd54244418e9fd9a1003fc84a47ddccf146aea3dd352c79f72ee52c"
)

ALLOWED_METRICS = (
    "reports",
    "token_power",
    "token_fdp",
    "token_iou",
    "final_report_fdp",
    "report_ufdp",
)

FIGURE_LAYOUT = "single continuous passage; source paragraph breaks collapsed for display only"
FIGURE_AI_STYLE = "italic"
FIGURE_REJECTION_STYLE = "all rejected text is colored and underlined"
FIGURE_TRUE_REJECTION_STYLE = "blue italic: rejected watermarked token"
FIGURE_FALSE_REJECTION_STYLE = "vermillion roman: rejected human token"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def contract_payload() -> dict[str, object]:
    return {
        "case_id": CASE_ID,
        "source": {
            "pdf_basename": SOURCE_PDF_BASENAME,
            "pdf_sha256": SOURCE_PDF_SHA256,
            "title": SOURCE_TITLE,
            "author": SOURCE_AUTHOR,
            "date": SOURCE_DATE,
            "printed_page": SOURCE_PRINTED_PAGE,
            "physical_page": SOURCE_PHYSICAL_PAGE,
            "paragraphs": SOURCE_PARAGRAPHS,
            "words": SOURCE_WORDS,
            "normalization": SOURCE_NORMALIZATION,
            "excerpt_sha256": SOURCE_EXCERPT_SHA256,
            "opt_token_count": SOURCE_OPT_TOKEN_COUNT,
            "opt_token_sha256": SOURCE_OPT_TOKEN_SHA256,
        },
        "model": {
            "name": MODEL_NAME,
            "revision": MODEL_REVISION,
            "dtype": MODEL_DTYPE,
            "parameter_count": MODEL_PARAMETER_COUNT,
            "vocab_size": MODEL_VOCAB_SIZE,
            "temperature": TEMPERATURE,
            "full_vocabulary": True,
            "top_k": None,
            "top_p": None,
            "repetition_penalty": None,
            "stop_at_eos": False,
        },
        "construction": {
            "pattern": ["human", "watermarked", "human", "watermarked", "human"],
            "watermark_blocks": WATERMARK_BLOCKS,
            "source_block_targets": [80, 100, 80, 100, 80],
            "watermark_tokens_per_block": "target 100; first sentence ending at or after token 90; maximum 140",
            "block_lengths": "sentence-aligned source partition optimized before generation against 80/100/80/100/80",
            "alternative_intervals": "recorded after deterministic generation",
            "human_split": "three deterministic sentence-aligned source spans targeting 80 tokens each",
            "replacement_not_insertion": True,
            "replacement_unit": "complete source sentences",
            "prompt": PROMPT,
        },
        "rng": {
            "master_seed": CASE_MASTER_SEED,
            "bit_generator": BIT_GENERATOR,
            "key_stream": KEY_STREAM,
            "ordinary_stream": ORDINARY_STREAM,
            "ordinary_stream_use": "unused: null tokens are fixed human tokens",
            "fresh_full_key_at_every_position": True,
            "no_seed_search_regeneration_or_outcome_retry": True,
            "design_change_protocol": (
                "after inspecting seeds 20260921091501 and 20260921091502, the block geometry "
                "was changed to a locked 80/100/80/100/80 sentence-aligned target; this design "
                "uses one fresh seed and its first completed run is accepted without retry"
            ),
            "python_numpy_torch_seeds_explicit": True,
            "deterministic_torch_algorithms": True,
        },
        "watermark": "exact full-vocabulary Gumbel-max",
        "human_null": "source token fixed without inspecting the current key vector",
        "detector": {
            "strategy": DETECTOR_STRATEGY,
            "weighted_adaptive_cap": SWZ_CAP,
            "threshold_exact": THRESHOLD_EXACT,
            "threshold_display": THRESHOLD_DISPLAY,
            "threshold_rule": "exact gamma 49; conservative integer ceiling for the general 10% bound",
            "localizer": "refreshing plus last global minimum",
            "source_sha256": DETECTOR_SOURCE_SHA256,
        },
        "figure": {
            "layout": FIGURE_LAYOUT,
            "source_tokenization_unchanged": True,
            "ai_text": FIGURE_AI_STYLE,
            "rejections": FIGURE_REJECTION_STYLE,
            "true_rejection": FIGURE_TRUE_REJECTION_STYLE,
            "false_rejection": FIGURE_FALSE_REJECTION_STYLE,
        },
        "reported_metrics": list(ALLOWED_METRICS),
        "single_path_descriptive_only": True,
    }


def validate_static_contract() -> dict[str, object]:
    import refreshing_swz

    expected_lock = contract_payload()
    detector_path = ROOT / "refreshing_swz.py"
    observed_hash = sha256_file(detector_path)
    if observed_hash != DETECTOR_SOURCE_SHA256:
        raise RuntimeError(
            f"certified detector hash mismatch: {observed_hash} != {DETECTOR_SOURCE_SHA256}"
        )
    if refreshing_swz.DEFAULT_SWZ_CAP != SWZ_CAP:
        raise RuntimeError("weighted-adaptive cap changed")
    calibrated = 1.0 / refreshing_swz.alpha_for_general_fdr_target(0.10)
    if THRESHOLD_EXACT != THRESHOLD_DISPLAY or THRESHOLD_EXACT != 49:
        raise RuntimeError("the case must use exact gamma=49")
    if math.ceil(calibrated) != THRESHOLD_EXACT:
        raise RuntimeError("gamma=49 is no longer the conservative integer ceiling")
    if (1.0 + math.log(THRESHOLD_EXACT)) / THRESHOLD_EXACT > 0.10:
        raise RuntimeError("gamma=49 no longer controls the stated general bound")
    return expected_lock


if __name__ == "__main__":
    print(json.dumps(validate_static_contract(), indent=2, sort_keys=True))
