"""Exact scalar-pivot mathematics for the ideal Tournament watermark."""

from __future__ import annotations

import itertools
import math
from typing import Iterable

import numpy as np


LAYERS = 30
COMPETITORS_PER_MATCH = 2
BINOMIAL_PMF = np.asarray(
    [math.comb(LAYERS, s) / (2**LAYERS) for s in range(LAYERS + 1)],
    dtype=np.float64,
)
BINOMIAL_CDF_LOWER = np.concatenate(
    (np.asarray([0.0], dtype=np.float64), np.cumsum(BINOMIAL_PMF[:-1]))
)
FLOAT32_MASS_TOLERANCE = 8.0 * np.finfo(np.float32).eps


def normalize_float32_tournament_mass(updated_probabilities):
    """Remove float32 roundoff below zero and normalize each probability row.

    The Tournament update is nonnegative and has unit mass mathematically.
    When its Bernoulli mass ``q`` rounds just above one, float32 can nevertheless
    create tiny negative entries. Values within a documented machine-precision
    tolerance are clipped; larger negatives still fail loudly.
    """

    if str(updated_probabilities.dtype) != "torch.float32":
        raise TypeError("Tournament mass repair requires torch.float32 input")
    if updated_probabilities.ndim not in (1, 2):
        raise ValueError("Tournament probabilities must be one- or two-dimensional")

    was_vector = updated_probabilities.ndim == 1
    rows = updated_probabilities.unsqueeze(0) if was_vector else updated_probabilities
    row_minimum = rows.amin(dim=1)
    if not bool(row_minimum.isfinite().all()):
        raise RuntimeError("Tournament recursion produced nonfinite mass")
    worst = float(row_minimum.min().item())
    if worst < -FLOAT32_MASS_TOLERANCE:
        raise RuntimeError(
            "Tournament recursion produced material negative mass: "
            f"{worst:.9g} < {-FLOAT32_MASS_TOLERANCE:.9g}"
        )

    clipped = rows.clamp_min(0.0)
    totals = clipped.sum(dim=1, keepdim=True)
    if not bool(totals.isfinite().all()) or bool((totals <= 0.0).any()):
        raise RuntimeError("Tournament recursion lost all probability mass")
    roundoff = (-row_minimum).clamp_min(0.0).maximum(
        (totals[:, 0] - 1.0).abs()
    )
    normalized = clipped / totals
    if was_vector:
        return normalized[0], roundoff[0]
    return normalized, roundoff


def tournament_distribution(
    probabilities: Iterable[float], g_table: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the exact recursion and return final probabilities and mass errors.

    ``g_table`` has shape ``(m, vocabulary)`` and contains only zeroes/ones.
    Computation is float64 here for mathematical tests; model generation uses
    the identical recursion in float32 on the locked model device.
    """

    p = np.asarray(list(probabilities), dtype=np.float64)
    g = np.asarray(g_table)
    if p.ndim != 1 or p.size < 2:
        raise ValueError("probabilities must be a one-dimensional vocabulary law")
    if g.ndim != 2 or g.shape[1] != p.size:
        raise ValueError("g_table must have shape (layers, vocabulary)")
    if np.any((g != 0) & (g != 1)):
        raise ValueError("g-values must be Bernoulli bits")
    if np.any(p < 0.0) or not np.all(np.isfinite(p)):
        raise ValueError("probabilities must be finite and nonnegative")
    if not np.isclose(p.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("probabilities must sum to one")

    errors = np.empty(g.shape[0], dtype=np.float64)
    for layer, bits in enumerate(g):
        q = float(np.dot(p, bits))
        updated = p * (1.0 + bits - q)
        if np.any(updated < -1e-15):
            raise AssertionError("Tournament recursion produced negative mass")
        updated = np.maximum(updated, 0.0)
        total = float(updated.sum())
        errors[layer] = total - 1.0
        if not np.isfinite(total) or total <= 0.0:
            raise AssertionError("Tournament recursion lost all probability mass")
        p = updated / total  # exact identity; removes floating-point error only
    return p, errors


def pairwise_layer_bruteforce(probabilities: Iterable[float], g: Iterable[int]) -> np.ndarray:
    """Enumerate the two contestants and a fair tie break for one layer."""

    p = np.asarray(list(probabilities), dtype=np.float64)
    bits = np.asarray(list(g), dtype=np.uint8)
    if p.ndim != 1 or bits.shape != p.shape:
        raise ValueError("probabilities and g must have the same vector shape")
    out = np.zeros_like(p)
    for left, right in itertools.product(range(p.size), repeat=2):
        pair_probability = p[left] * p[right]
        if bits[left] > bits[right]:
            out[left] += pair_probability
        elif bits[right] > bits[left]:
            out[right] += pair_probability
        else:
            out[left] += 0.5 * pair_probability
            out[right] += 0.5 * pair_probability
    return out


def brute_force_tournament_distribution(
    probabilities: Iterable[float], g_table: np.ndarray
) -> np.ndarray:
    """Small-vocabulary audit by repeated explicit pair enumeration.

    This expands each layer's two competitors, not all ``2**m`` leaves.
    It is deliberately an independent implementation of the recursion.
    """

    p = np.asarray(list(probabilities), dtype=np.float64)
    for bits in np.asarray(g_table):
        p = pairwise_layer_bruteforce(p, bits)
    return p


def binomial_randomized_pit(
    scores: Iterable[int] | np.ndarray,
    uniforms: Iterable[float] | np.ndarray,
) -> np.ndarray:
    """Return ``F0(S-1) + V P0(S)`` for ``S~Binomial(30,1/2)``."""

    s = np.asarray(scores)
    v = np.asarray(uniforms, dtype=np.float64)
    if s.shape != v.shape:
        raise ValueError("scores and uniforms must have the same shape")
    if np.any(s != np.floor(s)) or np.any((s < 0) | (s > LAYERS)):
        raise ValueError(f"scores must be integers in [0,{LAYERS}]")
    if np.any(~np.isfinite(v)) or np.any((v < 0.0) | (v >= 1.0)):
        raise ValueError("randomizers must lie in [0,1)")
    indices = s.astype(np.int64)
    result = BINOMIAL_CDF_LOWER[indices] + v * BINOMIAL_PMF[indices]
    # The mathematical value is strictly below one for V<1. Roundoff in the
    # top Binomial cell can otherwise produce exactly 1.0 and an infinite L.
    return np.minimum(result, np.nextafter(1.0, 0.0))


def stable_calibrator(pivots: Iterable[float] | np.ndarray) -> np.ndarray:
    """Return ``L=-log(1-Y)`` by a stable log-survival calculation."""

    y = np.asarray(pivots, dtype=np.float64)
    if np.any(~np.isfinite(y)) or np.any((y < 0.0) | (y >= 1.0)):
        raise ValueError("pivots must lie in [0,1)")
    return -np.log1p(-y)
