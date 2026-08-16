#!/usr/bin/env python3
"""Run the prespecified Efron human-versus-watermark case study.

The selected opening of the Prologue to *Large-Scale Inference* is tokenized once.
Two disjoint 100-token source spans are replaced by exact full-vocabulary
Gumbel-max OPT-1.3B continuations, leaving a balanced H-W-H-W-H document of
the same token length as the source. Human tokens are fixed without inspecting
the current fresh key vector; their selected key coordinates are exact null
pivots. This is one descriptive path, never a Monte Carlo estimate.
"""

from __future__ import annotations

import argparse
import codecs
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Iterable, Sequence

import numpy as np

from efron_contract import (
    ALLOWED_METRICS,
    ALTERNATIVE_INTERVALS,
    BIT_GENERATOR,
    BLOCK_LENGTHS,
    CASE_ID,
    CASE_MASTER_SEED,
    DETECTOR_STRATEGY,
    KEY_STREAM,
    MIN_HUMAN_BLOCK_TOKENS,
    MODEL_DTYPE,
    MODEL_NAME,
    MODEL_PARAMETER_COUNT,
    MODEL_REVISION,
    MODEL_VOCAB_SIZE,
    ORDINARY_STREAM,
    PROMPT,
    ROOT,
    SOURCE_EXCERPT_SHA256,
    SOURCE_OPT_TOKEN_COUNT,
    SOURCE_OPT_TOKEN_SHA256,
    SOURCE_URL,
    SWZ_CAP,
    TEMPERATURE,
    THRESHOLD_DISPLAY,
    THRESHOLD_EXACT,
    WATERMARK_BLOCK_TOKENS,
    canonical_json,
    sha256_file,
    validate_static_contract,
)
from refreshing_swz import Region, path_metrics, run_refreshing_from_pivots
DEFAULT_SOURCE = ROOT / "efron_prologue.txt"
DEFAULT_OUTPUT = ROOT.parent / "results" / "llm" / "efron_case"
TORCH_THREADS = 8
MPL_CACHE = Path(os.environ.get("EFRON_MPLCONFIGDIR", "/tmp/efron_opt13b_matplotlib"))
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

FIGURE_INK = "#25343D"
TRUE_REJECTION_COLOR = "#0072B2"
FALSE_REJECTION_COLOR = "#D55E00"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def token_ids_sha256(ids: Sequence[int]) -> str:
    values = np.asarray(ids, dtype="<i4")
    return hashlib.sha256(values.tobytes()).hexdigest()


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def write_json_atomic(path: Path, value: object) -> None:
    write_text_atomic(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_frozen_source(path: Path = DEFAULT_SOURCE) -> str:
    text = path.read_text(encoding="utf-8")
    if sha256_text(text) != SOURCE_EXCERPT_SHA256:
        raise RuntimeError("source excerpt is not the frozen Efron text")
    if len(text.rstrip("\n").split("\n\n")) != 8 or len(text.split()) != 532:
        raise RuntimeError("frozen Efron source structure changed")
    return text


def balanced_human_lengths(
    horizon: int,
    watermark_tokens: int = WATERMARK_BLOCK_TOKENS,
    minimum_human_tokens: int = MIN_HUMAN_BLOCK_TOKENS,
) -> tuple[int, int, int]:
    """Prespecified, outcome-independent human split for H-W-H-W-H."""

    if watermark_tokens < 1:
        raise ValueError("watermark_tokens must be positive")
    remaining = horizon - 2 * watermark_tokens
    if remaining < 3 * minimum_human_tokens:
        raise ValueError("source is too short for two watermark replacements")
    base, extra = divmod(remaining, 3)
    lengths = tuple(base + int(index < extra) for index in range(3))
    if max(lengths) - min(lengths) > 1 or sum(lengths) != remaining:
        raise AssertionError("balanced human split failed")
    return lengths  # type: ignore[return-value]


def _build_plan_from_source_ids(
    tokenizer,
    source_ids: Sequence[int],
    *,
    watermark_tokens: int,
    minimum_human_tokens: int,
) -> tuple[list[dict], int]:
    horizon = len(source_ids)
    human_lengths = balanced_human_lengths(
        horizon,
        watermark_tokens=watermark_tokens,
        minimum_human_tokens=minimum_human_tokens,
    )
    pattern = ("human", "watermarked", "human", "watermarked", "human")
    lengths = (
        human_lengths[0],
        watermark_tokens,
        human_lengths[1],
        watermark_tokens,
        human_lengths[2],
    )
    blocks: list[dict] = []
    source_cursor = 0
    output_cursor = 1
    human_index = 0
    watermark_index = 0
    for kind, length in zip(pattern, lengths, strict=True):
        source_slice = list(source_ids[source_cursor : source_cursor + length])
        if len(source_slice) != length:
            raise AssertionError("source plan overran the excerpt")
        if kind == "human":
            human_index += 1
            name = f"Human source {human_index}"
            fixed_ids: list[int] | None = source_slice
        else:
            watermark_index += 1
            name = f"Watermarked replacement {watermark_index}"
            fixed_ids = None
        blocks.append(
            {
                "name": name,
                "kind": kind,
                "start": output_cursor,
                "end": output_cursor + length - 1,
                "length": length,
                "source_start": source_cursor + 1,
                "source_end": source_cursor + length,
                "source_token_sha256": token_ids_sha256(source_slice),
                "fixed_token_ids": fixed_ids,
                "output_token_ids": None,
                "output_text_sha256": None,
            }
        )
        source_cursor += length
        output_cursor += length
    if source_cursor != horizon or output_cursor - 1 != horizon:
        raise AssertionError("block plan did not consume the source exactly")
    return blocks, horizon


def build_block_plan(
    tokenizer,
    source_text: str,
    *,
    watermark_tokens: int = WATERMARK_BLOCK_TOKENS,
    minimum_human_tokens: int = MIN_HUMAN_BLOCK_TOKENS,
) -> tuple[list[dict], int, list[int]]:
    source_ids = list(tokenizer(source_text, add_special_tokens=False).input_ids)
    if not source_ids:
        raise RuntimeError("source tokenization is empty")
    decoded = tokenizer.decode(
        source_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if decoded != source_text:
        raise RuntimeError("Efron source failed exact OPT tokenizer round trip")
    if sha256_text(source_text) == SOURCE_EXCERPT_SHA256:
        if len(source_ids) != SOURCE_OPT_TOKEN_COUNT:
            raise RuntimeError(
                f"pinned tokenizer produced {len(source_ids)} source tokens; "
                f"expected {SOURCE_OPT_TOKEN_COUNT}"
            )
        if token_ids_sha256(source_ids) != SOURCE_OPT_TOKEN_SHA256:
            raise RuntimeError("pinned Efron source token IDs changed")
    blocks, horizon = _build_plan_from_source_ids(
        tokenizer,
        source_ids,
        watermark_tokens=watermark_tokens,
        minimum_human_tokens=minimum_human_tokens,
    )
    if sha256_text(source_text) == SOURCE_EXCERPT_SHA256:
        if tuple(int(block["length"]) for block in blocks) != BLOCK_LENGTHS:
            raise RuntimeError("frozen block lengths changed")
        intervals = tuple(
            (int(block["start"]), int(block["end"]))
            for block in blocks
            if block["kind"] == "watermarked"
        )
        if intervals != ALTERNATIVE_INTERVALS:
            raise RuntimeError("frozen alternative intervals changed")
    return blocks, horizon, source_ids


def select_exact_gumbel_max(scaled_logits: np.ndarray, key_u: np.ndarray) -> int:
    """Exact full-vocabulary keyed choice used in the certified experiment."""

    logits = np.asarray(scaled_logits, dtype=np.float64)
    key = np.asarray(key_u, dtype=np.float64)
    if logits.ndim != 1 or key.shape != logits.shape:
        raise ValueError("logits and key must be equal-length vectors")
    if np.any(key <= 0.0) or np.any(key >= 1.0):
        raise ValueError("key values must lie strictly inside (0,1)")
    if not np.any(np.isfinite(logits)):
        raise ValueError("all logits are non-finite")
    return int(np.argmax(logits - np.log(-np.log(key))))


def report_union_mask(reports: Iterable, horizon: int) -> np.ndarray:
    mask = np.zeros(horizon, dtype=bool)
    for report in reports:
        start = int(report.interval_start)
        end = int(report.interval_end)
        if not 1 <= start <= end <= horizon:
            raise ValueError("report lies outside the monitored document")
        mask[start - 1 : end] = True
    return mask


def case_metrics(detector, regions: Sequence[Region], horizon: int) -> dict[str, float | int]:
    raw = path_metrics(detector.reports, regions, horizon=horizon)
    token_fdp = int(raw["false_positive_tokens"]) / max(int(raw["predicted_tokens"]), 1)
    metrics: dict[str, float | int] = {
        "reports": int(raw["reports"]),
        "token_power": float(raw["token_recall"]),
        "token_fdp": float(token_fdp),
        "token_iou": float(raw["token_iou"]),
        "final_report_fdp": float(raw["final_fdp"]),
        "report_ufdp": float(raw["uniform_fdp"]),
    }
    if tuple(metrics) != ALLOWED_METRICS:
        raise AssertionError("reported metric surface changed")
    return metrics


def validate_loaded_model(model, tokenizer) -> str:
    checkpoint = getattr(model.config, "_commit_hash", None)
    if checkpoint != MODEL_REVISION:
        raise RuntimeError(f"checkpoint mismatch: {checkpoint!r} != {MODEL_REVISION!r}")
    if getattr(model.config, "model_type", None) != "opt":
        raise RuntimeError("loaded checkpoint is not an OPT model")
    if int(model.config.vocab_size) != MODEL_VOCAB_SIZE:
        raise RuntimeError("OPT vocabulary size changed")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != MODEL_PARAMETER_COUNT:
        raise RuntimeError(
            f"parameter count mismatch: {parameter_count} != {MODEL_PARAMETER_COUNT}"
        )
    if str(next(model.parameters()).dtype) != MODEL_DTYPE:
        raise RuntimeError("model is not float32")
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("a fast tokenizer is required for exact figure offsets")
    return str(checkpoint)


def generate_mixed_document(tokenizer, model, torch, blocks: Sequence[dict]) -> dict[str, object]:
    horizon = sum(int(block["length"]) for block in blocks)
    prompt_ids = list(tokenizer(PROMPT, add_special_tokens=False).input_ids)
    if not prompt_ids:
        raise RuntimeError("prompt tokenization is empty")
    if len(prompt_ids) + horizon > int(model.config.max_position_embeddings):
        raise RuntimeError("mixed document exceeds the pinned model context window")

    vocab_size = int(model.config.vocab_size)
    key_rng = np.random.Generator(np.random.PCG64DXSM(CASE_MASTER_SEED))
    ordinary_seed = np.random.SeedSequence(
        CASE_MASTER_SEED,
        spawn_key=(ORDINARY_STREAM,),
    ).generate_state(4)
    token_ids = np.empty(horizon, dtype=np.int32)
    pivots = np.empty(horizon, dtype=np.float64)
    is_watermarked = np.empty(horizon, dtype=bool)
    block_index = np.empty(horizon, dtype=np.int8)
    is_eos = np.empty(horizon, dtype=bool)
    eos_id = tokenizer.eos_token_id

    device = next(model.parameters()).device
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    started = time.perf_counter()
    zero_t = 0
    with torch.inference_mode():
        initial = model(prompt_tensor, use_cache=True)
        next_logits = initial.logits[0, -1, :].detach().cpu().double().numpy()
        past = initial.past_key_values
        for current_block_index, block in enumerate(blocks):
            human_ids = block["fixed_token_ids"] if block["kind"] == "human" else None
            generated_ids: list[int] = []
            for offset in range(int(block["length"])):
                scaled_logits = np.asarray(next_logits / TEMPERATURE, dtype=np.float64)
                if not np.all(np.isfinite(scaled_logits)):
                    scaled_logits = scaled_logits.copy()
                    scaled_logits[~np.isfinite(scaled_logits)] = -np.inf

                # A human token is fixed before the fresh key draw and is never
                # selected by ordinary model sampling. The separate ordinary
                # RNG stream is therefore intentionally unconsumed.
                chosen_id = int(human_ids[offset]) if human_ids is not None else None
                key_u = key_rng.random(vocab_size, dtype=np.float64)
                np.maximum(key_u, np.nextafter(0.0, 1.0), out=key_u)
                np.minimum(key_u, np.nextafter(1.0, 0.0), out=key_u)
                if chosen_id is None:
                    chosen_id = select_exact_gumbel_max(scaled_logits, key_u)

                token_ids[zero_t] = chosen_id
                pivots[zero_t] = key_u[chosen_id]
                is_watermarked[zero_t] = block["kind"] == "watermarked"
                block_index[zero_t] = current_block_index
                is_eos[zero_t] = eos_id is not None and chosen_id == eos_id
                generated_ids.append(chosen_id)

                chosen = torch.tensor([[chosen_id]], dtype=torch.long, device=device)
                updated = model(chosen, past_key_values=past, use_cache=True)
                next_logits = updated.logits[0, -1, :].detach().cpu().double().numpy()
                past = updated.past_key_values
                zero_t += 1
                if zero_t % 100 == 0 or zero_t == horizon:
                    print(f"generated/scored {zero_t}/{horizon} monitored tokens", flush=True)

            block["output_token_ids"] = generated_ids
            block_text = tokenizer.decode(
                generated_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            block["output_text_sha256"] = sha256_text(block_text)

    if zero_t != horizon:
        raise AssertionError("generation stopped before the frozen horizon")
    expected_truth = np.asarray(
        [
            block["kind"] == "watermarked"
            for block in blocks
            for _ in range(int(block["length"]))
        ],
        dtype=bool,
    )
    if not np.array_equal(is_watermarked, expected_truth):
        raise AssertionError("truth mask does not match the frozen block plan")
    return {
        "token_id": token_ids,
        "pivot_y": pivots,
        "is_watermarked": is_watermarked,
        "block_index": block_index,
        "is_eos": is_eos,
        "generation_seconds": time.perf_counter() - started,
        "prompt_token_count": len(prompt_ids),
        "ordinary_stream_seed_words": [int(value) for value in ordinary_seed],
    }


def token_masks_to_character_masks(
    tokenizer,
    text: str,
    token_ids: Sequence[int],
    truth: Sequence[bool],
    selected: Sequence[bool],
) -> tuple[np.ndarray, np.ndarray]:
    """Map saved OPT token masks to decoded characters without assuming a unique BPE encoding."""

    if len(truth) != len(token_ids) or len(selected) != len(token_ids):
        raise ValueError("token masks have the wrong length")
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded.input_ids) == list(token_ids):
        truth_chars = np.zeros(len(text), dtype=bool)
        selected_chars = np.zeros(len(text), dtype=bool)
        touched = np.zeros(len(text), dtype=bool)
        for (start, end), is_truth, is_selected in zip(
            encoded.offset_mapping,
            truth,
            selected,
            strict=True,
        ):
            if end > start:
                truth_chars[start:end] = bool(is_truth)
                selected_chars[start:end] = bool(is_selected)
                touched[start:end] = True
        visible = ~np.asarray([character.isspace() for character in text])
        if text and not np.all(touched | ~visible):
            raise RuntimeError("fast-tokenizer offsets left visible characters unmapped")
        return truth_chars, selected_chars

    # BPE tokenization is not uniquely recoverable from decoded text. Adjacent
    # saved tokens may re-encode as one longer merge even though the sequence is
    # a valid model output. Handle complete per-token fragments first.
    pieces = [
        tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for token_id in token_ids
    ]
    if "".join(pieces) == text:
        truth_chars = np.zeros(len(text), dtype=bool)
        selected_chars = np.zeros(len(text), dtype=bool)
        cursor = 0
        for piece, is_truth, is_selected in zip(pieces, truth, selected, strict=True):
            end = cursor + len(piece)
            truth_chars[cursor:end] = bool(is_truth)
            selected_chars[cursor:end] = bool(is_selected)
            cursor = end
        return truth_chars, selected_chars

    # OPT uses GPT-2 byte-level BPE. A Unicode character can span token
    # boundaries, so reconstruct exact bytes and attribute each displayed
    # character to every saved token that supplied one of its bytes.
    return byte_level_token_masks_to_character_masks(
        tokenizer,
        text,
        token_ids,
        truth,
        selected,
    )


def gpt2_byte_decoder() -> dict[str, int]:
    """Return the reversible byte alphabet used by GPT-2 and OPT tokenizers."""

    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    unicode_values = list(byte_values)
    offset = 0
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            unicode_values.append(256 + offset)
            offset += 1
    return {
        chr(character): byte
        for byte, character in zip(byte_values, unicode_values, strict=True)
    }


def byte_level_token_masks_to_character_masks(
    tokenizer,
    text: str,
    token_ids: Sequence[int],
    truth: Sequence[bool],
    selected: Sequence[bool],
) -> tuple[np.ndarray, np.ndarray]:
    """Map GPT-2/OPT byte-level tokens to characters without re-tokenizing."""

    byte_decoder = gpt2_byte_decoder()
    tokens = tokenizer.convert_ids_to_tokens(list(token_ids))
    if isinstance(tokens, str):
        tokens = [tokens]
    if len(tokens) != len(token_ids):
        raise RuntimeError("tokenizer returned the wrong number of token strings")

    special_ids = set(getattr(tokenizer, "all_special_ids", ()))
    raw = bytearray()
    byte_owners: list[int] = []
    for index, (token_id, token) in enumerate(zip(token_ids, tokens, strict=True)):
        if int(token_id) in special_ids:
            token_bytes = tokenizer.decode(
                [int(token_id)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ).encode("utf-8")
        else:
            try:
                token_bytes = bytes(byte_decoder[character] for character in token)
            except KeyError as error:
                raise RuntimeError(
                    "saved tokens cannot be mapped with the OPT byte decoder"
                ) from error
        raw.extend(token_bytes)
        byte_owners.extend([index] * len(token_bytes))

    error_spans: list[tuple[int, int]] = []

    def record_decode_error(error: UnicodeDecodeError) -> tuple[str, int]:
        error_spans.append((error.start, error.end))
        return "\ufffd", error.end

    error_handler = f"efron_opt_character_map_{id(error_spans)}"
    codecs.register_error(error_handler, record_decode_error)
    decoded = bytes(raw).decode("utf-8", errors=error_handler)
    if decoded != text:
        raise RuntimeError("saved OPT token bytes do not reproduce the decoded text")

    truth_chars = np.zeros(len(text), dtype=bool)
    selected_chars = np.zeros(len(text), dtype=bool)
    byte_cursor = 0
    error_cursor = 0
    for character_index, character in enumerate(text):
        if (
            error_cursor < len(error_spans)
            and error_spans[error_cursor][0] == byte_cursor
        ):
            end = error_spans[error_cursor][1]
            error_cursor += 1
        else:
            encoded_character = character.encode("utf-8")
            end = byte_cursor + len(encoded_character)
            if bytes(raw[byte_cursor:end]) != encoded_character:
                raise RuntimeError("could not align decoded OPT text with token bytes")
        owners = byte_owners[byte_cursor:end]
        truth_chars[character_index] = any(bool(truth[owner]) for owner in owners)
        selected_chars[character_index] = any(bool(selected[owner]) for owner in owners)
        byte_cursor = end
    if byte_cursor != len(raw) or error_cursor != len(error_spans):
        raise RuntimeError("OPT token-byte mapping did not consume the decoded text")
    return truth_chars, selected_chars


def wrap_masked_text(
    text: str,
    truth: np.ndarray,
    selected: np.ndarray,
    *,
    width: int = 92,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Collapse source paragraph breaks and wrap one continuous passage."""

    if len(text) != truth.size or len(text) != selected.size:
        raise ValueError("text and character masks differ in length")

    flat_characters: list[str] = []
    flat_truth: list[bool] = []
    flat_selected: list[bool] = []
    index = 0
    while index < len(text):
        if not text[index].isspace():
            flat_characters.append(text[index])
            flat_truth.append(bool(truth[index]))
            flat_selected.append(bool(selected[index]))
            index += 1
            continue
        run_end = index + 1
        while run_end < len(text) and text[run_end].isspace():
            run_end += 1
        if flat_characters and run_end < len(text):
            flat_characters.append(" ")
            flat_truth.append(bool(np.any(truth[index:run_end])))
            flat_selected.append(bool(np.any(selected[index:run_end])))
        index = run_end

    flat_text = "".join(flat_characters)
    truth_flat = np.asarray(flat_truth, dtype=bool)
    selected_flat = np.asarray(flat_selected, dtype=bool)
    lines: list[tuple[str, np.ndarray, np.ndarray]] = []
    cursor = 0
    while cursor < len(flat_text):
        target = min(cursor + width, len(flat_text))
        if target < len(flat_text):
            cut = flat_text.rfind(" ", cursor, target + 1)
            if cut <= cursor:
                cut = target
        else:
            cut = target
        visible_end = cut
        while visible_end > cursor and flat_text[visible_end - 1] == " ":
            visible_end -= 1
        if visible_end > cursor:
            lines.append(
                (
                    flat_text[cursor:visible_end],
                    truth_flat[cursor:visible_end].copy(),
                    selected_flat[cursor:visible_end].copy(),
                )
            )
        cursor = cut
        while cursor < len(flat_text) and flat_text[cursor] == " ":
            cursor += 1
    return lines


def styled_line_runs(
    line: str,
    truth: np.ndarray,
    selected: np.ndarray,
) -> list[dict[str, object]]:
    """Return the fail-closed four-state visual encoding for one text line."""

    if len(line) != truth.size or len(line) != selected.size:
        raise ValueError("line and character masks differ in length")
    if not line:
        return []
    states = truth.astype(np.uint8) + 2 * selected.astype(np.uint8)
    boundaries = np.flatnonzero(np.r_[True, states[1:] != states[:-1], True])
    runs: list[dict[str, object]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        is_ai = bool(truth[start])
        is_rejected = bool(selected[start])
        if is_rejected and is_ai:
            color = TRUE_REJECTION_COLOR
            rejection_class = "true"
        elif is_rejected:
            color = FALSE_REJECTION_COLOR
            rejection_class = "false"
        else:
            color = FIGURE_INK
            rejection_class = None
        runs.append(
            {
                "start": int(start),
                "end": int(end),
                "text": line[start:end],
                "is_ai": is_ai,
                "is_rejected": is_rejected,
                "fontstyle": "italic" if is_ai else "normal",
                "color": color,
                "underline": is_rejected,
                "rejection_class": rejection_class,
            }
        )
    return runs


def render_case_box(
    tokenizer,
    output_pdf: Path,
    output_png: Path,
    token_ids: np.ndarray,
    truth: np.ndarray,
    selected: np.ndarray,
    metrics: dict[str, float | int],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D
    from matplotlib.transforms import ScaledTranslation, blended_transform_factory

    text = tokenizer.decode(
        token_ids.astype(int).tolist(),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    truth_chars, selected_chars = token_masks_to_character_masks(
        tokenizer,
        text,
        token_ids.astype(int).tolist(),
        truth.tolist(),
        selected.tolist(),
    )
    lines = wrap_masked_text(text, truth_chars, selected_chars)
    height = max(4.2, 2.0 + 0.175 * len(lines))
    fig = plt.figure(figsize=(8.0, height), facecolor="white")
    text_ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0.035, right=0.985, top=0.91, bottom=0.105)

    text_ax.set_xlim(0.0, 1.0)
    text_ax.set_ylim(len(lines) + 0.3, -0.8)
    text_ax.set_xticks([])
    text_ax.set_yticks([])
    for spine in text_ax.spines.values():
        spine.set_color("#AAB7BE")
        spine.set_linewidth(0.9)
    fontsize = 8.2
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axis_width_pixels = float(text_ax.get_window_extent(renderer).width)
    if axis_width_pixels <= 0.0:
        raise RuntimeError("figure text box has invalid width")
    text_transform = blended_transform_factory(text_ax.transAxes, text_ax.transData)
    underline_transform = text_transform + ScaledTranslation(
        0.0,
        -0.92 * fontsize / 72.0,
        fig.dpi_scale_trans,
    )
    for line_number, (line, line_truth, line_selected) in enumerate(lines):
        y = line_number + 0.15
        x_pixels = 0.012 * axis_width_pixels
        for run in styled_line_runs(line, line_truth, line_selected):
            font = FontProperties(
                family="DejaVu Serif",
                style=str(run["fontstyle"]),
                size=fontsize,
            )
            run_text = str(run["text"])
            run_width_pixels = float(
                renderer.get_text_width_height_descent(run_text, font, ismath=False)[0]
            )
            x_start = x_pixels / axis_width_pixels
            x_end = (x_pixels + run_width_pixels) / axis_width_pixels
            text_ax.text(
                x_start,
                y,
                run_text,
                ha="left",
                va="top",
                fontproperties=font,
                color=str(run["color"]),
                transform=text_transform,
                zorder=1,
            )
            if bool(run["underline"]):
                text_ax.plot(
                    [x_start, x_end],
                    [y, y],
                    color=str(run["color"]),
                    linewidth=0.9,
                    solid_capstyle="butt",
                    transform=underline_transform,
                    zorder=2,
                )
            x_pixels += run_width_pixels
        if x_pixels > 0.988 * axis_width_pixels:
            raise RuntimeError("wrapped figure line exceeds the text box")

    metric_line = (
        f"Reports {int(metrics['reports'])}   |   "
        f"token Power {float(metrics['token_power']):.3f}   |   "
        f"token FDP {float(metrics['token_fdp']):.3f}   |   "
        f"token IoU {float(metrics['token_iou']):.3f}"
    )
    fig.suptitle(
        "Human/watermarked localization in Bradley Efron's Large-Scale Inference",
        x=0.5,
        y=0.995,
        fontsize=12.5,
        fontweight="bold",
        color=FIGURE_INK,
    )
    fig.text(
        0.5,
        0.958,
        metric_line,
        ha="center",
        va="top",
        fontsize=8.7,
        color=TRUE_REJECTION_COLOR,
    )
    legend = fig.legend(
        handles=[
            Line2D([], [], color="none", label="human text (roman)"),
            Line2D([], [], color="none", label="AI text (italic)"),
            Line2D([], [], color=TRUE_REJECTION_COLOR, linewidth=1.6, label="true rejection (AI)"),
            Line2D([], [], color=FALSE_REJECTION_COLOR, linewidth=1.6, label="false rejection (human)"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.026),
        ncol=4,
        frameon=False,
        fontsize=7.8,
    )
    legend.get_texts()[1].set_fontstyle("italic")
    legend.get_texts()[2].set_color(TRUE_REJECTION_COLOR)
    legend.get_texts()[3].set_color(FALSE_REJECTION_COLOR)
    fig.text(
        0.01,
        0.006,
        "Source: Bradley Efron (2010); one prespecified descriptive path, not an FDR estimate.",
        ha="left",
        va="bottom",
        fontsize=6.8,
        color=FIGURE_INK,
    )
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(output_png, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def contiguous_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1) + 1
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def render_refreshing_process(
    output_pdf: Path,
    output_png: Path,
    arrays: dict[str, np.ndarray],
    blocks: Sequence[dict],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    pivots = np.asarray(arrays["pivot_y"], dtype=float)
    truth = np.asarray(arrays["is_watermarked"], dtype=bool)
    detector = run_refreshing_from_pivots(
        pivots,
        strategy=DETECTOR_STRATEGY,
        threshold=THRESHOLD_EXACT,
        cap=SWZ_CAP,
    )
    selected = report_union_mask(detector.reports, pivots.size)
    if not np.array_equal(selected, arrays["selected_by_any_report"]):
        raise RuntimeError("refreshing-process replay differs from saved selection")

    human_color = "#D7DEE2"
    watermark_color = "#F4DDAF"
    curve_color = "#0B6E8E"
    alarm_color = "#B94A48"
    report_color = "#188977"
    ink = "#263640"
    grid = "#D5DEE3"
    positions = np.arange(1, pivots.size + 1)

    fig = plt.figure(figsize=(12.5, 7.4), facecolor="white")
    layout = fig.add_gridspec(3, 1, height_ratios=(0.75, 1.55, 3.0), hspace=0.16)
    source_axis = fig.add_subplot(layout[0])
    pivot_axis = fig.add_subplot(layout[1], sharex=source_axis)
    process_axis = fig.add_subplot(layout[2], sharex=source_axis)

    for block in blocks:
        start, end = int(block["start"]), int(block["end"])
        is_watermarked = block["kind"] == "watermarked"
        source_axis.broken_barh(
            [(start - 0.5, end - start + 1)],
            (1.02, 0.56),
            facecolors=watermark_color if is_watermarked else human_color,
            edgecolors="white",
            linewidth=1.1,
        )
        source_axis.text(
            (start + end) / 2,
            1.30,
            "WM" if is_watermarked else "H",
            ha="center",
            va="center",
            fontsize=11,
            color=ink,
        )
        if is_watermarked:
            for axis in (pivot_axis, process_axis):
                axis.axvspan(
                    start - 0.5,
                    end + 0.5,
                    color=watermark_color,
                    alpha=0.58,
                    linewidth=0,
                    zorder=0,
                )
    for start, end in contiguous_intervals(selected):
        source_axis.broken_barh(
            [(start - 0.5, end - start + 1)],
            (0.18, 0.42),
            facecolors=report_color,
            edgecolors="white",
            linewidth=0.7,
        )
    source_axis.text(
        -0.015,
        1.30,
        "source",
        transform=source_axis.get_yaxis_transform(),
        ha="right",
        va="center",
        fontsize=11,
        color=ink,
    )
    source_axis.text(
        -0.015,
        0.39,
        "selected",
        transform=source_axis.get_yaxis_transform(),
        ha="right",
        va="center",
        fontsize=11,
        color=ink,
    )
    source_axis.set_ylim(0.0, 1.75)
    source_axis.axis("off")

    pivot_axis.plot(positions, pivots, color="#5F6D75", linewidth=0.8, zorder=2)
    pivot_axis.set_ylim(0.0, 1.0)
    pivot_axis.set_ylabel(r"Pivotal statistic $Y_t$", fontsize=11)
    pivot_axis.tick_params(axis="x", labelbottom=False)

    candidate = np.asarray(detector.candidate_logwealth, dtype=float)
    process_axis.plot(positions, candidate, color=curve_color, linewidth=1.35, zorder=3)
    process_axis.axhline(
        math.log(THRESHOLD_EXACT),
        color=alarm_color,
        linestyle="--",
        linewidth=1.4,
        zorder=2,
    )
    alarm = np.asarray(detector.alarm, dtype=bool)
    process_axis.scatter(
        positions[alarm],
        candidate[alarm],
        color=alarm_color,
        s=22,
        zorder=4,
    )
    lower = min(float(candidate.min()), -1.0)
    report_y = lower + 0.35
    for report in detector.reports:
        process_axis.plot(
            [report.interval_start, report.interval_end],
            [report_y, report_y],
            color=report_color,
            linewidth=3.2,
            solid_capstyle="butt",
            zorder=4,
        )
    process_axis.set_ylim(
        lower - 0.15,
        max(float(candidate.max()), math.log(THRESHOLD_EXACT)) + 0.55,
    )
    process_axis.set_ylabel("log refreshing process", fontsize=11)
    process_axis.set_xlabel("Token position", fontsize=11)
    process_axis.set_xlim(0.5, pivots.size + 0.5)

    for axis in (pivot_axis, process_axis):
        axis.grid(color=grid, linewidth=0.65)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#9FADB5")
        axis.tick_params(labelsize=10, colors=ink)

    fig.suptitle(
        "Refreshing process for mixed Efron and watermarked text",
        fontsize=15,
        color=ink,
        y=0.99,
    )
    fig.legend(
        handles=[
            Line2D([], [], color=curve_color, linewidth=1.8, label="log refreshing process"),
            Line2D([], [], color=alarm_color, linestyle="--", linewidth=1.6,
                   label=r"threshold $\log(49)$"),
            Line2D([], [], marker="o", linestyle="none", color=alarm_color,
                   markersize=6, label="alarm and refresh"),
            Line2D([], [], color=report_color, linewidth=3.2, label="localized report"),
            Patch(facecolor=watermark_color, label="watermarked OPT-1.3B block"),
            Patch(facecolor=human_color, label="fixed Efron block"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=3,
        frameon=False,
        fontsize=9.5,
    )
    fig.subplots_adjust(left=0.105, right=0.985, top=0.93, bottom=0.15)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, facecolor="white")
    fig.savefig(output_png, dpi=300, facecolor="white")
    plt.close(fig)


def latex_figure_fragment(metrics: dict[str, float | int], pdf_name: str) -> str:
    return rf"""\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{{pdf_name}}}
\caption{{A prespecified mixed human/watermarked excerpt from Bradley Efron's
\emph{{Large-Scale Inference}}. Two 100-token source spans are replaced by
watermarked OPT-1.3B continuations, and paragraph breaks are collapsed for
display only. All AI text is italic. Every rejected span is colored and
underlined: blue for a true rejection of AI text and vermillion for a false
rejection of human text. The one-path values are
Power={float(metrics['token_power']):.3f}, FDP={float(metrics['token_fdp']):.3f},
and IoU={float(metrics['token_iou']):.3f}. They are descriptive and are not
Monte Carlo estimates of power or FDR.}}
\label{{fig:efron-human-watermark}}
\end{{figure}}
"""


def serialize_blocks(blocks: Sequence[dict]) -> list[dict]:
    return [
        {
            key: value
            for key, value in block.items()
            if key not in {"fixed_token_ids", "output_token_ids"}
        }
        for block in blocks
    ]


def alternative_regions(blocks: Sequence[dict]) -> tuple[Region, ...]:
    return tuple(
        Region(int(block["start"]), int(block["end"]))
        for block in blocks
        if block["kind"] == "watermarked"
    )


def block_coverage(blocks: Sequence[dict], selected: np.ndarray) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for block in blocks:
        start, end = int(block["start"]), int(block["end"])
        count = int(selected[start - 1 : end].sum())
        result.append(
            {
                "name": block["name"],
                "kind": block["kind"],
                "start": start,
                "end": end,
                "tokens": int(block["length"]),
                "selected_tokens": count,
            }
        )
    return result


def build_derived_artifacts(
    tokenizer,
    output_dir: Path,
    arrays: dict[str, np.ndarray],
    blocks: Sequence[dict],
    metrics: dict[str, float | int],
) -> dict[str, object]:
    pdf_path = output_dir / "efron_case_box.pdf"
    png_path = output_dir / "efron_case_box.png"
    tex_path = output_dir / "efron_case_box.tex"
    process_pdf_path = output_dir / "efron_refreshing_process.pdf"
    process_png_path = output_dir / "efron_refreshing_process.png"
    render_case_box(
        tokenizer,
        pdf_path,
        png_path,
        arrays["token_id"],
        arrays["is_watermarked"],
        arrays["selected_by_any_report"],
        metrics,
    )
    write_text_atomic(tex_path, latex_figure_fragment(metrics, pdf_path.name))
    render_refreshing_process(
        process_pdf_path,
        process_png_path,
        arrays,
        blocks,
    )
    return {
        "figure_pdf": pdf_path.name,
        "figure_pdf_sha256": sha256_file(pdf_path),
        "figure_png": png_path.name,
        "figure_png_sha256": sha256_file(png_path),
        "latex_fragment": tex_path.name,
        "latex_fragment_sha256": sha256_file(tex_path),
        "refreshing_process_pdf": process_pdf_path.name,
        "refreshing_process_pdf_sha256": sha256_file(process_pdf_path),
        "refreshing_process_png": process_png_path.name,
        "refreshing_process_png_sha256": sha256_file(process_png_path),
    }


def replay_saved(tokenizer, output_dir: Path) -> int:
    record_path = output_dir / "case.json"
    arrays_path = output_dir / "case_arrays.npz"
    if not record_path.is_file() or not arrays_path.is_file():
        raise RuntimeError("saved case.json/case_arrays.npz are missing")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record["arrays_sha256"] != sha256_file(arrays_path):
        raise RuntimeError("saved array hash mismatch")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    detector = run_refreshing_from_pivots(
        arrays["pivot_y"],
        strategy=DETECTOR_STRATEGY,
        threshold=THRESHOLD_EXACT,
        cap=SWZ_CAP,
    )
    blocks = [dict(block) for block in record["blocks"]]
    regions = alternative_regions(blocks)
    metrics = case_metrics(detector, regions, int(arrays["pivot_y"].size))
    if metrics != record["metrics"]:
        raise RuntimeError("detector replay metrics differ from the saved record")
    selected = report_union_mask(detector.reports, int(arrays["pivot_y"].size))
    if not np.array_equal(selected, arrays["selected_by_any_report"]):
        raise RuntimeError("detector replay selection differs from saved arrays")
    derived = build_derived_artifacts(tokenizer, output_dir, arrays, blocks, metrics)
    manifest = {
        "case": record_path.name,
        "case_sha256": sha256_file(record_path),
        "arrays": arrays_path.name,
        "arrays_sha256": sha256_file(arrays_path),
        **derived,
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    print("replayed saved pivots and rebuilt the box; no text was regenerated")
    return 0


def load_model_and_tokenizer(*, device_name: str, local_files_only: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(TORCH_THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device(
        "cuda"
        if device_name == "cuda" or (device_name == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        local_files_only=local_files_only,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        local_files_only=local_files_only,
        dtype=torch.float32,
    )
    model.to(device)
    model.eval()
    checkpoint = validate_loaded_model(model, tokenizer)
    return tokenizer, model, torch, device, checkpoint


def run_smoke(tokenizer, model, torch, source_text: str) -> int:
    source_ids = list(tokenizer(source_text, add_special_tokens=False).input_ids)
    smoke_source_ids = source_ids[:13]
    blocks, horizon = _build_plan_from_source_ids(
        tokenizer,
        smoke_source_ids,
        watermark_tokens=2,
        minimum_human_tokens=3,
    )
    generated = generate_mixed_document(tokenizer, model, torch, blocks)
    pivots = np.asarray(generated["pivot_y"])
    if horizon != 13 or pivots.shape != (13,) or not np.all((pivots > 0) & (pivots < 1)):
        raise RuntimeError("model smoke failed")
    detector = run_refreshing_from_pivots(
        pivots,
        strategy=DETECTOR_STRATEGY,
        threshold=THRESHOLD_EXACT,
        cap=SWZ_CAP,
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "profile": "non-scientific-smoke",
                "tokens": horizon,
                "reports": len(detector.reports),
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="run a 13-token non-scientific model smoke and exit")
    parser.add_argument("--rebuild-derived", action="store_true", help="replay saved pivots and rebuild only the box")
    args = parser.parse_args(argv)

    validate_static_contract()
    source_text = load_frozen_source(args.source)

    if args.rebuild_derived:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
            local_files_only=args.local_files_only,
            use_fast=True,
        )
        return replay_saved(tokenizer, args.output_dir)

    if not args.smoke and any(
        (args.output_dir / name).exists() for name in ("case.json", "case_arrays.npz")
    ):
        raise SystemExit(
            "scientific case output already exists; the runner refuses regeneration or outcome-based retry"
        )

    print(f"loading {MODEL_NAME}@{MODEL_REVISION} in float32", flush=True)
    load_started = time.perf_counter()
    tokenizer, model, torch, device, checkpoint = load_model_and_tokenizer(
        device_name=args.device,
        local_files_only=args.local_files_only,
    )
    model_load_seconds = time.perf_counter() - load_started
    print(f"model loaded on {device} in {model_load_seconds:.1f} seconds", flush=True)
    if args.smoke:
        return run_smoke(tokenizer, model, torch, source_text)

    blocks, horizon, source_ids = build_block_plan(tokenizer, source_text)
    generated = generate_mixed_document(tokenizer, model, torch, blocks)
    pivots = np.asarray(generated["pivot_y"], dtype=np.float64)
    detector = run_refreshing_from_pivots(
        pivots,
        strategy=DETECTOR_STRATEGY,
        threshold=THRESHOLD_EXACT,
        cap=SWZ_CAP,
    )
    regions = alternative_regions(blocks)
    metrics = case_metrics(detector, regions, horizon)
    selected = report_union_mask(detector.reports, horizon)

    arrays = {
        "token_id": np.asarray(generated["token_id"], dtype=np.int32),
        "pivot_y": pivots,
        "is_watermarked": np.asarray(generated["is_watermarked"], dtype=bool),
        "block_index": np.asarray(generated["block_index"], dtype=np.int8),
        "is_eos": np.asarray(generated["is_eos"], dtype=bool),
        "selected_by_any_report": selected,
        "e_factor": detector.e_factor,
        "bet_fraction": detector.bet_fraction,
        "candidate_logwealth": detector.candidate_logwealth,
        "postreset_logwealth": detector.postreset_logwealth,
        "running_min_time": detector.running_min_time,
        "alarm": detector.alarm,
        "report_after_time": detector.report_after_time,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = args.output_dir / "case_arrays.npz"
    record_path = args.output_dir / "case.json"
    write_npz_atomic(arrays_path, arrays)

    mixed_text = tokenizer.decode(
        arrays["token_id"].astype(int).tolist(),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    record = {
        "schema_version": 2,
        "case_id": CASE_ID,
        "single_path_descriptive_only": True,
        "no_seed_selection_regeneration_or_outcome_retry": True,
        "source_url": SOURCE_URL,
        "source_excerpt_sha256": SOURCE_EXCERPT_SHA256,
        "source_token_count": len(source_ids),
        "source_token_sha256": token_ids_sha256(source_ids),
        "mixed_text_sha256": sha256_text(mixed_text),
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "model_checkpoint_commit": checkpoint,
        "model_dtype": str(next(model.parameters()).dtype),
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "temperature": TEMPERATURE,
        "full_vocabulary": True,
        "watermark_rule": "exact full-vocabulary Gumbel-max",
        "human_null_rule": "source token fixed without inspecting the current key vector",
        "fresh_full_key_at_every_position": True,
        "stop_at_eos": False,
        "case_master_seed": CASE_MASTER_SEED,
        "bit_generator": BIT_GENERATOR,
        "key_stream": KEY_STREAM,
        "ordinary_stream": ORDINARY_STREAM,
        "ordinary_stream_use": "unused because null tokens are fixed human tokens",
        "ordinary_stream_seed_words": generated["ordinary_stream_seed_words"],
        "prompt": PROMPT,
        "prompt_token_count": generated["prompt_token_count"],
        "threshold_exact": THRESHOLD_EXACT,
        "threshold_display": THRESHOLD_DISPLAY,
        "eprocess": f"weighted adaptive e-process, cap {SWZ_CAP:g}",
        "localizer": "refreshing plus last global minimum",
        "blocks": serialize_blocks(blocks),
        "alternative_intervals": [[region.start, region.end] for region in regions],
        "reports": [asdict(report) for report in detector.reports],
        "metrics": metrics,
        "audit_counts": {
            "horizon": horizon,
            "human_tokens": int((~arrays["is_watermarked"]).sum()),
            "watermarked_tokens": int(arrays["is_watermarked"].sum()),
            "selected_tokens": int(selected.sum()),
            "watermarked_selected": int((selected & arrays["is_watermarked"]).sum()),
            "human_selected": int((selected & ~arrays["is_watermarked"]).sum()),
            "block_coverage": block_coverage(blocks, selected),
        },
        "runtime": {
            "model_load_seconds": model_load_seconds,
            "generation_seconds": generated["generation_seconds"],
            "torch_threads": TORCH_THREADS,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "arrays_file": arrays_path.name,
        "arrays_sha256": sha256_file(arrays_path),
    }
    record["record_sha256"] = hashlib.sha256(
        canonical_json(record).encode("utf-8")
    ).hexdigest()
    write_json_atomic(record_path, record)
    derived = build_derived_artifacts(tokenizer, args.output_dir, arrays, blocks, metrics)
    manifest = {
        "case": record_path.name,
        "case_sha256": sha256_file(record_path),
        "arrays": arrays_path.name,
        "arrays_sha256": sha256_file(arrays_path),
        **derived,
    }
    write_json_atomic(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"status": "completed", "metrics": metrics}, indent=2))
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
