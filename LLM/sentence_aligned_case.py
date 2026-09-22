"""Shared design helpers for the sentence-aligned Efron mixed-text cases."""

from __future__ import annotations

import hashlib
import itertools
import os
from pathlib import Path
import random
import re
from typing import Sequence

import numpy as np


SOURCE_BLOCK_TARGETS = (80, 100, 80, 100, 80)
HUMAN_TARGET_TOKENS = 80
WATERMARK_TARGET_TOKENS = 100
MIN_GENERATED_TOKENS = 90
MAX_GENERATED_TOKENS = 140
SENTENCE_END = re.compile(r"[.!?][\"'\u2019\u201d)\]]*$")
SENTENCE_BOUNDARY = re.compile(
    r"[.!?][\"'\u2019\u201d)\]]*(?=\s+[A-Z(])"
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def token_ids_sha256(ids: Sequence[int]) -> str:
    values = np.asarray(ids, dtype="<i4")
    return hashlib.sha256(values.tobytes()).hexdigest()


def is_complete_sentence_text(text: str) -> bool:
    """Return whether the visible text ends with sentence punctuation."""

    return bool(SENTENCE_END.search(text.rstrip()))


def source_sentence_boundaries(source_text: str) -> list[int]:
    """Return deterministic source offsets immediately after full sentences."""

    if not source_text or not is_complete_sentence_text(source_text):
        raise RuntimeError("source text must end with a complete sentence")
    boundaries = [0]
    boundaries.extend(match.end() for match in SENTENCE_BOUNDARY.finditer(source_text))
    boundaries.append(len(source_text))
    boundaries = sorted(set(boundaries))
    if len(boundaries) < 6:
        raise RuntimeError("source contains too few complete sentences")
    return boundaries


def _balanced_source_partition(tokenizer, source_text: str) -> tuple[list[int], list[int]]:
    """Choose the sentence-aligned five-block split closest to 80/100/80/100/80."""

    boundaries = source_sentence_boundaries(source_text)
    cache: dict[tuple[int, int], list[int]] = {}

    def ids_between(start: int, end: int) -> list[int]:
        key = (start, end)
        if key not in cache:
            cache[key] = list(
                tokenizer(
                    source_text[start:end], add_special_tokens=False
                ).input_ids
            )
        return cache[key]

    best: tuple[tuple[object, ...], list[int], list[int]] | None = None
    for interior in itertools.combinations(boundaries[1:-1], 4):
        cuts = [0, *interior, len(source_text)]
        counts = [
            len(ids_between(start, end))
            for start, end in zip(cuts[:-1], cuts[1:])
        ]
        deviations = [
            abs(count - target)
            for count, target in zip(counts, SOURCE_BLOCK_TARGETS)
        ]
        key: tuple[object, ...] = (
            sum(deviation * deviation for deviation in deviations),
            max(deviations),
            tuple(deviations),
            tuple(interior),
        )
        if best is None or key < best[0]:
            best = (key, cuts, counts)

    if best is None:
        raise RuntimeError("no sentence-aligned five-block partition is available")
    _, cuts, counts = best
    return cuts, counts


def set_reproducible_seeds(torch, seed: int) -> dict[str, object]:
    """Seed every general-purpose RNG and request deterministic torch kernels."""

    legacy_seed = int(seed % (2**32))
    torch_seed = int(seed % (2**63 - 1))
    os.environ.setdefault("PYTHONHASHSEED", str(legacy_seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(legacy_seed)
    np.random.seed(legacy_seed)
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return {
        "master_seed": int(seed),
        "python_random_seed": legacy_seed,
        "numpy_legacy_seed": legacy_seed,
        "torch_seed": torch_seed,
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "tf32": False,
    }


def build_sentence_aligned_plan(tokenizer, source_text: str) -> tuple[list[dict], list[int]]:
    """Build the locked balanced H-W-H-W-H sentence-aligned plan."""

    source_ids = list(tokenizer(source_text, add_special_tokens=False).input_ids)
    if not source_ids:
        raise RuntimeError("source tokenization is empty")

    cuts, source_counts = _balanced_source_partition(tokenizer, source_text)
    kinds = ("human", "watermarked", "human", "watermarked", "human")
    blocks: list[dict] = []
    human_number = 0
    watermark_number = 0
    for index, (kind, start, end, source_count, source_target) in enumerate(
        zip(kinds, cuts[:-1], cuts[1:], source_counts, SOURCE_BLOCK_TARGETS)
    ):
        text = source_text[start:end]
        token_ids = list(tokenizer(text, add_special_tokens=False).input_ids)
        if len(token_ids) != source_count:
            raise AssertionError("balanced partition token count changed")
        if not is_complete_sentence_text(text):
            raise RuntimeError("a balanced source block does not end at a sentence boundary")
        common = {
            "source_character_start": start,
            "source_character_end": end,
            "source_text_sha256": sha256_text(text),
            "source_token_sha256": token_ids_sha256(token_ids),
            "source_token_count": source_count,
            "source_target_tokens": source_target,
            "partition_index": index,
        }
        if kind == "human":
            human_number += 1
            blocks.append(
                {
                    **common,
                    "name": f"Human source {human_number}",
                    "kind": "human",
                    "fixed_token_ids": token_ids,
                    "source_replaced_text": None,
                    "target_tokens": HUMAN_TARGET_TOKENS,
                }
            )
        else:
            watermark_number += 1
            blocks.append(
                {
                    **common,
                    "name": f"Watermarked sentence replacement {watermark_number}",
                    "kind": "watermarked",
                    "fixed_token_ids": None,
                    "source_replaced_text": text,
                    "target_tokens": WATERMARK_TARGET_TOKENS,
                    "minimum_tokens": MIN_GENERATED_TOKENS,
                    "maximum_tokens": MAX_GENERATED_TOKENS,
                }
            )

    if [block["kind"] for block in blocks] != [
        "human",
        "watermarked",
        "human",
        "watermarked",
        "human",
    ]:
        raise AssertionError("sentence-aligned plan is not H-W-H-W-H")
    return blocks, source_ids


def should_stop_generated_block(
    tokenizer,
    generated_ids: Sequence[int],
    *,
    minimum_tokens: int,
    maximum_tokens: int,
) -> bool:
    """Apply the prespecified sentence-completion stopping rule."""

    count = len(generated_ids)
    if count > maximum_tokens:
        raise RuntimeError(
            "no complete generated sentence appeared before the locked maximum"
        )
    if count < minimum_tokens:
        return False
    text = tokenizer.decode(
        list(generated_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return is_complete_sentence_text(text)


def finalize_generated_block(tokenizer, block: dict, generated_ids: Sequence[int]) -> str:
    """Validate and record one generated complete-sentence block."""

    text = tokenizer.decode(
        list(generated_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if len(generated_ids) < int(block["minimum_tokens"]):
        raise RuntimeError("generated block is shorter than its locked minimum")
    if len(generated_ids) > int(block["maximum_tokens"]):
        raise RuntimeError("generated block is longer than its locked maximum")
    if not is_complete_sentence_text(text):
        raise RuntimeError("generated block does not end at a sentence boundary")
    block["length"] = len(generated_ids)
    block["output_token_ids"] = list(generated_ids)
    block["output_text_sha256"] = sha256_text(text)
    block["output_text"] = text
    return text


def assign_output_intervals(blocks: Sequence[dict]) -> int:
    """Assign one-indexed token intervals after variable-length generation."""

    cursor = 1
    for block in blocks:
        if block["kind"] == "human":
            block["length"] = len(block["fixed_token_ids"])
        length = int(block["length"])
        block["start"] = cursor
        block["end"] = cursor + length - 1
        cursor += length
    return cursor - 1


def write_blind_text(output_dir: Path, mixed_text: str) -> Path:
    """Save unannotated prose for a content-only human readability check."""

    path = output_dir / "mixed_text_blind.txt"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(mixed_text, encoding="utf-8")
    os.replace(temporary, path)
    return path
