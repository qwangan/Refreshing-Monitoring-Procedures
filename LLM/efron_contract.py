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

SOURCE_URL = "https://efron.ckirby.su.domains/other/2010LSIexcerpt.pdf"
SOURCE_TITLE = "Large-Scale Inference: Empirical Bayes Methods for Estimation, Testing, and Prediction"
SOURCE_AUTHOR = "Bradley Efron"
SOURCE_DATE = "2010"
SOURCE_PARAGRAPHS = 8
SOURCE_WORDS = 532
SOURCE_NORMALIZATION = "user-supplied-prologue-opening-through-real-examples-v1"
SOURCE_EXCERPT_SHA256 = (
    "954dd8df3f28ac92aaaf36955b044d747fb9af117cf28a7f682b4364ad647098"
)
SOURCE_OPT_TOKEN_COUNT = 655
SOURCE_OPT_TOKEN_SHA256 = (
    "efe39ecae122665f3c819dee7617d1d0a7988caa1a5fb0324d9e70fee5749f9a"
)

PROMPT = "Large-Scale Inference\nBradley Efron\nPrologue\n\n"
CASE_ID = "efron-large-scale-inference-human-watermark-opt13b-v1"
CASE_MASTER_SEED = 20260816090442
BIT_GENERATOR = "PCG64DXSM"
KEY_STREAM = 0
ORDINARY_STREAM = 1
TEMPERATURE = 1.0
WATERMARK_BLOCKS = 2
WATERMARK_BLOCK_TOKENS = 100
MIN_HUMAN_BLOCK_TOKENS = 25
BLOCK_LENGTHS = (152, 100, 152, 100, 151)
ALTERNATIVE_INTERVALS = ((153, 252), (405, 504))

THRESHOLD_EXACT = 49
THRESHOLD_DISPLAY = 49
SWZ_CAP = 0.5
DETECTOR_STRATEGY = "adaptive_cumulative"
DETECTOR_SOURCE_SHA256 = (
    "e146a0a76ac8ff3e5e95361e6704b422be3152a99c9ca774f66ae76d46106ca5"
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
            "url": SOURCE_URL,
            "title": SOURCE_TITLE,
            "author": SOURCE_AUTHOR,
            "date": SOURCE_DATE,
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
            "watermark_tokens_per_block": WATERMARK_BLOCK_TOKENS,
            "block_lengths": list(BLOCK_LENGTHS),
            "alternative_intervals": [list(interval) for interval in ALTERNATIVE_INTERVALS],
            "human_split": "balanced deterministic split of the remaining source tokens",
            "replacement_not_insertion": True,
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
