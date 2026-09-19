"""SWZ e-processes, refreshing, localization, and evaluation.

The code in this module is deliberately model agnostic.  It consumes the
Gumbel-max pivots ``Y_t`` saved by a language-model generation run and replays
several betting rules on exactly the same path.

Conventions
-----------
* Tokens and true regions are one-based.
* A refreshing report is ``(sigma, tau]`` and therefore contains the integer
  tokens ``sigma + 1, ..., tau``.
* A report is *localized false* iff its reported interval contains no token
  from any true watermark region.  This is the truth definition used for the
  report-level FDP/FDR calculations below.  Merge and bridge metrics are
  localization-quality diagnostics, not false-discovery labels.
* ``uniform_fdr`` below estimates ``E[sup_t FDP_t]``.  This is intentionally
  distinct from ``sup_t E[FDP_t]`` and from pooled false/total report counts.

The adaptive rule is the weighted adaptive log calibrator in Su, Wang, and Zhao:

    X_t = -log(1 - Y_t),
    eta_1 = 0,
    eta_t = argmax_{eta in [0, cap]}
              sum_{s < t} log(1 - eta + eta X_s),
    E_t = 1 - eta_t + eta_t X_t.

The paper uses ``cap=1/2`` in its implementations.  We call the betting
fraction ``eta`` to avoid confusing it with unrelated lambda notation in the
refreshing paper.

This module also implements the endpoint-adjusted Online Grenander (OG)
calibrator in SWZ equation (13),

    argmax_f sum_{s < t} log f(1-Y_s) + .5 log f(0) + .5 log f(1),

over decreasing densities on ``[0,1]``.  The optimizer is the weighted
Grenander maximum-likelihood estimator, computed by pooled adjacent violators.
The SWZ average process is implemented as the process-level mixture

    M_t^avg = .5 M_t^WA + .5 M_t^OG,

not as an average of the two current e-factors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


FLOAT_TINY = np.finfo(np.float64).tiny
DEFAULT_SWZ_CAP = 0.5
DEFAULT_OG_ENDPOINT_WEIGHT = 0.5


@dataclass(frozen=True)
class GrenanderFit:
    """A decreasing step density on ``[0,1]``.

    ``density[j]`` applies to ``(left[j], right[j]]``; the first block also
    supplies the value at zero.  This left-continuous convention matches SWZ.
    """

    left: np.ndarray
    right: np.ndarray
    density: np.ndarray
    total_weight: float
    log_likelihood: float

    def evaluate(self, values: float | Iterable[float]) -> float | np.ndarray:
        """Evaluate the fitted left-continuous density."""

        scalar = np.isscalar(values)
        value = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(value)):
            raise ValueError("evaluation points must be finite")
        if np.any((value < 0.0) | (value > 1.0)):
            raise ValueError("evaluation points must lie in [0, 1]")
        indices = np.searchsorted(self.right, value, side="left")
        indices = np.minimum(indices, self.density.size - 1)
        result = self.density[indices]
        return float(result) if scalar else result

    @property
    def integral(self) -> float:
        """Exact integral of the step density over ``[0,1]``."""

        return float(np.sum((self.right - self.left) * self.density))


@dataclass(frozen=True, order=True)
class Region:
    """A closed, one-based token region ``[start, end]``."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 1 or self.end < self.start:
            raise ValueError(f"invalid one-based closed region [{self.start}, {self.end}]")

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class Report:
    """One atomic refreshing report with interval ``(sigma, tau]``."""

    report: int
    previous_tau: int
    block_start: int
    sigma: int
    interval_start: int
    tau: int
    interval_end: int
    interval_length: int
    candidate_logwealth: float
    candidate_wealth: float
    localizer_log_min: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DetectorResult:
    """Complete token trace and reports from one detector replay."""

    strategy: str
    threshold: float
    cap: float | None
    fixed_eta: float | None
    pivot_y: np.ndarray
    calibrator_x: np.ndarray
    bet_fraction: np.ndarray
    e_factor: np.ndarray
    log_e_factor: np.ndarray
    candidate_logwealth: np.ndarray
    postreset_logwealth: np.ndarray
    running_min_time: np.ndarray
    running_min_logwealth: np.ndarray
    alarm: np.ndarray
    report_after_time: np.ndarray
    reports: tuple[Report, ...]

    def trace_dict(self) -> dict[str, np.ndarray]:
        """Return trace arrays with stable, CSV-friendly names."""

        return {
            "pivot_y": self.pivot_y,
            "calibrator_x": self.calibrator_x,
            "bet_fraction": self.bet_fraction,
            "e_factor": self.e_factor,
            "log_e_factor": self.log_e_factor,
            "candidate_logwealth": self.candidate_logwealth,
            "postreset_logwealth": self.postreset_logwealth,
            "running_min_time": self.running_min_time,
            "running_min_logwealth": self.running_min_logwealth,
            "alarm": self.alarm,
            "report_after_time": self.report_after_time,
        }

    def report_dicts(self) -> list[dict]:
        return [report.as_dict() for report in self.reports]


def _one_dimensional_float_array(values: Iterable[float], name: str) -> np.ndarray:
    if isinstance(values, np.ndarray):
        result = np.asarray(values, dtype=np.float64)
    else:
        result = np.asarray(list(values), dtype=np.float64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def calibrator_values(pivot_y: Iterable[float]) -> np.ndarray:
    """Return ``X_t=-log(1-Y_t)`` using the SWZ log calibrator.

    Exact endpoints are accepted.  ``Y=1`` is clipped only to the smallest
    positive float so that a finite log value can be stored.  Gumbel keys are
    continuous, so this guard has probability zero in the mathematical model.
    """

    y = _one_dimensional_float_array(pivot_y, "pivot_y")
    if np.any((y < 0.0) | (y > 1.0)):
        raise ValueError("every pivot must lie in [0, 1]")
    p = np.clip(1.0 - y, FLOAT_TINY, 1.0)
    return -np.log(p)


def swz_log_objective(eta: float, history_x: Iterable[float]) -> float:
    """Evaluate the concave past log-wealth objective optimized by SWZ."""

    if not 0.0 <= eta < 1.0:
        raise ValueError("eta must lie in [0, 1)")
    x = _one_dimensional_float_array(history_x, "history_x")
    if np.any(x < 0.0):
        raise ValueError("calibrator values must be nonnegative")
    return float(np.log1p(eta * (x - 1.0)).sum())


def swz_optimal_bet(
    history_x: Iterable[float],
    cap: float = DEFAULT_SWZ_CAP,
    *,
    tolerance: float = 1e-13,
    max_iterations: int = 80,
) -> float:
    """Compute the exact one-dimensional SWZ empirical-log optimum.

    The derivative is

    ``sum_s (X_s-1) / (1 + eta (X_s-1))``.

    It is nonincreasing, so boundary checks followed by bisection find the
    global maximizer without a generic optimizer.  Empty history returns zero,
    giving the paper's ``eta_1=0``.
    """

    if not 0.0 < cap < 1.0:
        raise ValueError("cap must lie strictly between zero and one")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")

    x = _one_dimensional_float_array(history_x, "history_x")
    if np.any(x < 0.0):
        raise ValueError("calibrator values must be nonnegative")
    if x.size == 0:
        return 0.0

    z = x - 1.0
    derivative_at_zero = float(z.sum())
    if derivative_at_zero <= 0.0:
        return 0.0

    derivative_at_cap = float(np.sum(z / (1.0 + cap * z)))
    if derivative_at_cap >= 0.0:
        return float(cap)

    lower, upper = 0.0, float(cap)
    for _ in range(max_iterations):
        midpoint = 0.5 * (lower + upper)
        derivative = float(np.sum(z / (1.0 + midpoint * z)))
        if derivative > 0.0:
            lower = midpoint
        else:
            upper = midpoint
        if upper - lower <= tolerance:
            break
    return 0.5 * (lower + upper)


def fixed_e_factors(pivot_y: Iterable[float], eta: float) -> np.ndarray:
    """Return ``1-eta+eta[-log(1-Y_t)]`` for a fixed betting fraction."""

    if not 0.0 <= eta <= 1.0:
        raise ValueError("fixed eta must lie in [0, 1]")
    x = calibrator_values(pivot_y)
    return (1.0 - eta) + eta * x


def endpoint_adjusted_grenander_fit(
    history_p: Iterable[float],
    *,
    endpoint_weight: float = DEFAULT_OG_ENDPOINT_WEIGHT,
) -> GrenanderFit:
    """Fit SWZ's endpoint-adjusted Online Grenander calibrator.

    For past p-values ``p_s=1-Y_s``, SWZ equation (13) maximizes

    ``sum_s log f(p_s) + .5 log f(0) + .5 log f(1)``

    over decreasing densities on ``[0,1]``.  More generally this function
    allows the common endpoint weight to be supplied explicitly.  The
    likelihood is the weighted Grenander likelihood with a fractional
    observation at each endpoint.

    After sorting the distinct support points, an observation at a positive
    support point belongs to the interval ending at that point.  Continuity at
    zero assigns the weight at zero to the first positive-width interval.
    The decreasing-density MLE is then obtained by pooling adjacent intervals
    whenever their unconstrained heights increase from left to right.
    """

    if not np.isfinite(endpoint_weight) or endpoint_weight <= 0.0:
        raise ValueError("endpoint_weight must be finite and positive")
    p = _one_dimensional_float_array(history_p, "history_p")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("every p-value must lie in [0, 1]")

    observations = np.concatenate(([0.0], p, [1.0]))
    weights = np.concatenate(
        ([float(endpoint_weight)], np.ones(p.size), [float(endpoint_weight)])
    )
    order = np.argsort(observations, kind="stable")
    sorted_observations = observations[order]
    sorted_weights = weights[order]
    support, first_indices = np.unique(sorted_observations, return_index=True)
    support_weights = np.add.reduceat(sorted_weights, first_indices)
    if support.size < 2 or support[0] != 0.0 or support[-1] != 1.0:
        raise AssertionError("endpoint augmentation must span [0,1]")

    widths = np.diff(support)
    if np.any(widths <= 0.0):
        raise AssertionError("distinct support points must have positive spacings")
    # A left-continuous step density assigns observations at support[j+1] to
    # (support[j], support[j+1]].  The value at zero equals the first interval
    # height, so its fractional endpoint count is added to the first bin.
    counts = support_weights[1:].astype(np.float64, copy=True)
    counts[0] += float(support_weights[0])
    total_weight = float(p.size + 2.0 * endpoint_weight)
    if not np.isclose(float(counts.sum()), total_weight, rtol=0.0, atol=1e-12):
        raise AssertionError("Grenander bin weights do not sum to total weight")

    block_left: list[float] = []
    block_right: list[float] = []
    block_width: list[float] = []
    block_count: list[float] = []
    for index, (width, count) in enumerate(zip(widths, counts, strict=True)):
        block_left.append(float(support[index]))
        block_right.append(float(support[index + 1]))
        block_width.append(float(width))
        block_count.append(float(count))
        # Pool an adjacent violation of the required nonincreasing heights.
        while (
            len(block_count) >= 2
            and block_count[-2] / block_width[-2]
            < block_count[-1] / block_width[-1]
        ):
            block_right[-2] = block_right[-1]
            block_width[-2] += block_width[-1]
            block_count[-2] += block_count[-1]
            block_left.pop()
            block_right.pop()
            block_width.pop()
            block_count.pop()

    left = np.asarray(block_left, dtype=np.float64)
    right = np.asarray(block_right, dtype=np.float64)
    block_width_array = np.asarray(block_width, dtype=np.float64)
    block_count_array = np.asarray(block_count, dtype=np.float64)
    density = block_count_array / (total_weight * block_width_array)
    log_likelihood = float(np.sum(block_count_array * np.log(density)))
    fit = GrenanderFit(
        left=left,
        right=right,
        density=density,
        total_weight=total_weight,
        log_likelihood=log_likelihood,
    )
    if not np.isclose(fit.integral, 1.0, rtol=2e-15, atol=2e-15):
        raise AssertionError("fitted Grenander calibrator does not integrate to one")
    if np.any(np.diff(fit.density) > 1e-14):
        raise AssertionError("fitted Grenander calibrator is not decreasing")
    return fit


def online_grenander_e_factors(pivot_y: Iterable[float]) -> np.ndarray:
    """Return cumulative-history SWZ Online Grenander e-factors.

    The factor at time ``t`` is fitted strictly from pivots before ``t``.
    This reference implementation favors transparency and mathematical audit:
    it refits the weighted Grenander estimator at each token.  Horizon 600 is
    small enough for smoke tests; a full 1,400-path replay should benchmark or
    optimize this routine before launch.
    """

    y = _one_dimensional_float_array(pivot_y, "pivot_y")
    if np.any((y < 0.0) | (y > 1.0)):
        raise ValueError("every pivot must lie in [0, 1]")
    p = 1.0 - y
    e_values = np.empty(y.size, dtype=np.float64)
    for zero_t, p_t in enumerate(p):
        fit = endpoint_adjusted_grenander_fit(p[:zero_t])
        e_values[zero_t] = fit.evaluate(float(p_t))
    return e_values


class _RefreshingState:
    """Internal state machine shared by precomputed and adaptive runs."""

    def __init__(self, horizon: int, threshold: float) -> None:
        if horizon < 0:
            raise ValueError("horizon cannot be negative")
        if not np.isfinite(threshold) or threshold <= 1.0:
            raise ValueError("threshold must be finite and greater than one")
        self.threshold = float(threshold)
        self.log_threshold = math.log(self.threshold)
        self.post_logwealth = 0.0
        self.min_logwealth = 0.0
        self.sigma = 0
        self.previous_tau = 0
        self.report_number = 0
        self.reports: list[Report] = []

        self.candidate = np.empty(horizon, dtype=np.float64)
        self.postreset = np.empty(horizon, dtype=np.float64)
        self.min_time = np.empty(horizon, dtype=np.int64)
        self.min_value = np.empty(horizon, dtype=np.float64)
        self.alarm = np.zeros(horizon, dtype=bool)
        self.report_count = np.empty(horizon, dtype=np.int64)

    def step(self, zero_t: int, e_value: float) -> bool:
        if np.isnan(e_value) or e_value < 0.0 or np.isinf(e_value):
            raise ValueError(f"invalid e-factor at t={zero_t + 1}: {e_value}")
        t = zero_t + 1
        log_e = -math.inf if e_value == 0.0 else math.log(e_value)
        candidate = self.post_logwealth + log_e

        # ``<=`` deliberately selects the last global minimum, including ties.
        if candidate <= self.min_logwealth:
            self.min_logwealth = candidate
            self.sigma = t

        crossed = candidate >= self.log_threshold
        if crossed:
            self.report_number += 1
            try:
                candidate_wealth = math.exp(candidate)
            except OverflowError:
                candidate_wealth = math.inf
            self.reports.append(
                Report(
                    report=self.report_number,
                    previous_tau=self.previous_tau,
                    block_start=self.previous_tau + 1,
                    sigma=self.sigma,
                    interval_start=self.sigma + 1,
                    tau=t,
                    interval_end=t,
                    interval_length=t - self.sigma,
                    candidate_logwealth=candidate,
                    candidate_wealth=candidate_wealth,
                    localizer_log_min=self.min_logwealth,
                )
            )
            post = 0.0
        else:
            post = candidate

        self.candidate[zero_t] = candidate
        self.postreset[zero_t] = post
        self.min_time[zero_t] = self.sigma
        self.min_value[zero_t] = self.min_logwealth
        self.alarm[zero_t] = crossed
        self.report_count[zero_t] = self.report_number

        if crossed:
            self.previous_tau = t
            self.post_logwealth = 0.0
            self.min_logwealth = 0.0
            self.sigma = t
        else:
            self.post_logwealth = candidate
        return crossed


def _make_result(
    *,
    strategy: str,
    threshold: float,
    cap: float | None,
    fixed_eta: float | None,
    pivot_y: np.ndarray,
    calibrator_x: np.ndarray,
    bet_fraction: np.ndarray,
    e_factor: np.ndarray,
    state: _RefreshingState,
) -> DetectorResult:
    with np.errstate(divide="ignore"):
        log_e = np.log(e_factor)
    return DetectorResult(
        strategy=strategy,
        threshold=float(threshold),
        cap=cap,
        fixed_eta=fixed_eta,
        pivot_y=pivot_y.copy(),
        calibrator_x=calibrator_x.copy(),
        bet_fraction=bet_fraction.copy(),
        e_factor=e_factor.copy(),
        log_e_factor=log_e,
        candidate_logwealth=state.candidate,
        postreset_logwealth=state.postreset,
        running_min_time=state.min_time,
        running_min_logwealth=state.min_value,
        alarm=state.alarm,
        report_after_time=state.report_count,
        reports=tuple(state.reports),
    )


def run_refreshing_from_evalues(
    e_values: Iterable[float],
    *,
    threshold: float = 21.0,
) -> DetectorResult:
    """Run refreshing and last-global-minimum localization on given e-factors."""

    e = _one_dimensional_float_array(e_values, "e_values")
    if np.any(e < 0.0):
        raise ValueError("e-factors must be nonnegative")
    state = _RefreshingState(e.size, threshold)
    for zero_t, e_t in enumerate(e):
        state.step(zero_t, float(e_t))
    nan = np.full(e.size, np.nan, dtype=np.float64)
    return _make_result(
        strategy="precomputed",
        threshold=threshold,
        cap=None,
        fixed_eta=None,
        pivot_y=nan,
        calibrator_x=nan,
        bet_fraction=nan,
        e_factor=e,
        state=state,
    )


def run_average_from_component_evalues(
    weight_adaptive_e_values: Iterable[float],
    online_grenander_e_values: Iterable[float],
    *,
    threshold: float = 21.0,
) -> DetectorResult:
    """Refresh the exact 50/50 average from precomputed component factors.

    This is valid for cumulative calibrator adaptation, because each component
    factor is chosen from the strict past and does not depend on capital-reset
    thresholds.  Component capitals, mixture capital, and the localizer still
    reset at every crossing.  The returned effective factors are successive
    ratios of the within-block arithmetic-average capital.
    """

    wa = _one_dimensional_float_array(
        weight_adaptive_e_values, "weight_adaptive_e_values"
    )
    og = _one_dimensional_float_array(
        online_grenander_e_values, "online_grenander_e_values"
    )
    if wa.shape != og.shape:
        raise ValueError("component e-factor sequences must have the same shape")
    if np.any(wa <= 0.0) or np.any(og <= 0.0):
        raise ValueError("average-process component e-factors must be positive")

    horizon = wa.size
    effective = np.empty(horizon, dtype=np.float64)
    state = _RefreshingState(horizon, threshold)
    wa_logcapital = 0.0
    og_logcapital = 0.0
    mixture_logcapital = 0.0
    log_two = math.log(2.0)
    for zero_t, (wa_factor, og_factor) in enumerate(zip(wa, og, strict=True)):
        new_wa = wa_logcapital + math.log(float(wa_factor))
        new_og = og_logcapital + math.log(float(og_factor))
        new_mixture = float(np.logaddexp(new_wa, new_og) - log_two)
        effective_factor = math.exp(new_mixture - mixture_logcapital)
        effective[zero_t] = effective_factor
        crossed = state.step(zero_t, effective_factor)
        if crossed:
            wa_logcapital = 0.0
            og_logcapital = 0.0
            mixture_logcapital = 0.0
        else:
            wa_logcapital = new_wa
            og_logcapital = new_og
            mixture_logcapital = new_mixture

    nan = np.full(horizon, np.nan, dtype=np.float64)
    return _make_result(
        strategy="average_precomputed_components",
        threshold=threshold,
        cap=None,
        fixed_eta=None,
        pivot_y=nan,
        calibrator_x=nan,
        bet_fraction=nan,
        e_factor=effective,
        state=state,
    )


def run_refreshing_from_pivots(
    pivot_y: Iterable[float],
    *,
    strategy: str = "adaptive_cumulative",
    threshold: float = 21.0,
    cap: float = DEFAULT_SWZ_CAP,
    fixed_eta: float | None = None,
    optimizer_tolerance: float = 1e-13,
) -> DetectorResult:
    """Replay a fixed or adaptive SWZ detector on one pivot path.

    Parameters
    ----------
    strategy:
        ``"adaptive_cumulative"`` keeps all past calibrator observations when
        refreshing capital.  ``"adaptive_block_reset"`` discards the bettor's
        history after each alarm, so the first token of every new block uses
        ``eta=0``.  ``"fixed"`` uses ``fixed_eta`` at every token.

    The detector capital and last-minimum localizer reset after every crossing
    for all three strategies.  Only the adaptive learner's reset behavior
    differs between the two adaptive strategies.
    """

    allowed = {"adaptive_cumulative", "adaptive_block_reset", "fixed"}
    if strategy not in allowed:
        raise ValueError(f"strategy must be one of {sorted(allowed)}")
    if strategy == "fixed":
        if fixed_eta is None or not 0.0 <= fixed_eta <= 1.0:
            raise ValueError("fixed strategy requires fixed_eta in [0, 1]")
    elif fixed_eta is not None:
        raise ValueError("fixed_eta is only meaningful for strategy='fixed'")
    if strategy != "fixed" and not 0.0 < cap < 1.0:
        raise ValueError("cap must lie strictly between zero and one")

    y = _one_dimensional_float_array(pivot_y, "pivot_y")
    x = calibrator_values(y)
    horizon = y.size
    eta_values = np.empty(horizon, dtype=np.float64)
    e_values = np.empty(horizon, dtype=np.float64)
    state = _RefreshingState(horizon, threshold)
    block_history_start = 0

    for zero_t in range(horizon):
        if strategy == "fixed":
            eta_t = float(fixed_eta)
        else:
            history_start = 0 if strategy == "adaptive_cumulative" else block_history_start
            eta_t = swz_optimal_bet(
                x[history_start:zero_t],
                cap=cap,
                tolerance=optimizer_tolerance,
            )
        e_t = (1.0 - eta_t) + eta_t * float(x[zero_t])
        eta_values[zero_t] = eta_t
        e_values[zero_t] = e_t
        crossed = state.step(zero_t, e_t)
        if crossed and strategy == "adaptive_block_reset":
            # The crossing observation belongs to the completed block.  It is
            # not used to choose the first bet after refresh.
            block_history_start = zero_t + 1

    return _make_result(
        strategy=strategy,
        threshold=threshold,
        cap=None if strategy == "fixed" else float(cap),
        fixed_eta=float(fixed_eta) if strategy == "fixed" else None,
        pivot_y=y,
        calibrator_x=x,
        bet_fraction=eta_values,
        e_factor=e_values,
        state=state,
    )


def run_online_grenander_from_pivots(
    pivot_y: Iterable[float],
    *,
    strategy: str = "og_cumulative",
    threshold: float = 21.0,
) -> DetectorResult:
    """Replay SWZ's endpoint-adjusted Online Grenander e-process.

    ``og_cumulative`` retains every past pivot when detector capital refreshes.
    ``og_block_reset`` is our predictable extension: after a crossing, both
    detector capital and the Grenander fitting history restart.  SWZ study the
    cumulative process; they do not propose the refreshing block-reset variant.
    """

    allowed = {"og_cumulative", "og_block_reset"}
    if strategy not in allowed:
        raise ValueError(f"strategy must be one of {sorted(allowed)}")
    y = _one_dimensional_float_array(pivot_y, "pivot_y")
    x = calibrator_values(y)  # retained for the common DetectorResult schema
    p = 1.0 - y
    horizon = y.size
    e_values = np.empty(horizon, dtype=np.float64)
    no_bet_fraction = np.full(horizon, np.nan, dtype=np.float64)
    state = _RefreshingState(horizon, threshold)
    history_start = 0

    for zero_t, p_t in enumerate(p):
        fit = endpoint_adjusted_grenander_fit(p[history_start:zero_t])
        e_t = float(fit.evaluate(float(p_t)))
        e_values[zero_t] = e_t
        crossed = state.step(zero_t, e_t)
        if crossed and strategy == "og_block_reset":
            history_start = zero_t + 1

    return _make_result(
        strategy=strategy,
        threshold=threshold,
        cap=None,
        fixed_eta=None,
        pivot_y=y,
        calibrator_x=x,
        bet_fraction=no_bet_fraction,
        e_factor=e_values,
        state=state,
    )


def run_average_from_pivots(
    pivot_y: Iterable[float],
    *,
    strategy: str = "average_cumulative",
    threshold: float = 21.0,
    cap: float = DEFAULT_SWZ_CAP,
    optimizer_tolerance: float = 1e-13,
) -> DetectorResult:
    """Replay SWZ's 50/50 weighted-adaptive/OG average e-process.

    SWZ average the two *process capitals*:

    ``M_t = .5 M_t^WA + .5 M_t^OG``.

    Consequently the effective factor supplied to the refreshing state is the
    ratio of successive mixture capitals, not ``.5(E_t^WA+E_t^OG)``.  At a
    refreshing alarm both component capitals restart at one.  In cumulative
    mode their calibrators still learn from the entire strict past; in the
    block-reset extension their fitting histories restart as well.
    """

    allowed = {"average_cumulative", "average_block_reset"}
    if strategy not in allowed:
        raise ValueError(f"strategy must be one of {sorted(allowed)}")
    if not 0.0 < cap < 1.0:
        raise ValueError("cap must lie strictly between zero and one")
    y = _one_dimensional_float_array(pivot_y, "pivot_y")
    x = calibrator_values(y)
    p = 1.0 - y
    horizon = y.size
    eta_values = np.empty(horizon, dtype=np.float64)
    effective_e_values = np.empty(horizon, dtype=np.float64)
    state = _RefreshingState(horizon, threshold)
    history_start = 0
    wa_logcapital = 0.0
    og_logcapital = 0.0
    mixture_logcapital = 0.0
    log_two = math.log(2.0)

    for zero_t, p_t in enumerate(p):
        eta_t = swz_optimal_bet(
            x[history_start:zero_t],
            cap=cap,
            tolerance=optimizer_tolerance,
        )
        wa_factor = (1.0 - eta_t) + eta_t * float(x[zero_t])
        og_fit = endpoint_adjusted_grenander_fit(p[history_start:zero_t])
        og_factor = float(og_fit.evaluate(float(p_t)))

        new_wa_logcapital = wa_logcapital + math.log(wa_factor)
        new_og_logcapital = og_logcapital + math.log(og_factor)
        new_mixture_logcapital = float(
            np.logaddexp(new_wa_logcapital, new_og_logcapital) - log_two
        )
        effective_log_factor = new_mixture_logcapital - mixture_logcapital
        effective_factor = math.exp(effective_log_factor)
        eta_values[zero_t] = eta_t
        effective_e_values[zero_t] = effective_factor
        crossed = state.step(zero_t, effective_factor)

        if crossed:
            wa_logcapital = 0.0
            og_logcapital = 0.0
            mixture_logcapital = 0.0
            if strategy == "average_block_reset":
                history_start = zero_t + 1
        else:
            wa_logcapital = new_wa_logcapital
            og_logcapital = new_og_logcapital
            mixture_logcapital = new_mixture_logcapital

    return _make_result(
        strategy=strategy,
        threshold=threshold,
        cap=float(cap),
        fixed_eta=None,
        pivot_y=y,
        calibrator_x=x,
        bet_fraction=eta_values,
        e_factor=effective_e_values,
        state=state,
    )


def _normalize_regions(regions: Sequence[Region | Sequence[int]]) -> tuple[Region, ...]:
    normalized = tuple(
        region if isinstance(region, Region) else Region(int(region[0]), int(region[1]))
        for region in regions
    )
    normalized = tuple(sorted(normalized))
    for left, right in zip(normalized, normalized[1:]):
        if right.start <= left.end:
            raise ValueError("true regions must be disjoint; merge overlapping regions first")
    return normalized


def overlap_tokens(
    start: int,
    end: int,
    target_start: int,
    target_end: int,
) -> int:
    """Number of integer tokens shared by two closed intervals."""

    return max(0, min(end, target_end) - max(start, target_start) + 1)


def _report_record(report: Report | Mapping) -> dict:
    record = report.as_dict() if isinstance(report, Report) else dict(report)
    if "interval_start" not in record:
        record["interval_start"] = int(record["sigma"]) + 1
    if "interval_end" not in record:
        record["interval_end"] = int(record["tau"])
    if "sigma" not in record:
        record["sigma"] = int(record["interval_start"]) - 1
    if "tau" not in record:
        record["tau"] = int(record["interval_end"])
    start, end = int(record["interval_start"]), int(record["interval_end"])
    if start < 1 or end < start:
        raise ValueError(f"invalid report interval [{start}, {end}]")
    record["interval_start"] = start
    record["interval_end"] = end
    record["interval_length"] = end - start + 1
    return record


def annotate_reports(
    reports: Sequence[Report | Mapping],
    true_regions: Sequence[Region | Sequence[int]],
) -> list[dict]:
    """Add truth, merge, bridge, boundary, purity, and IoU fields to reports."""

    regions = _normalize_regions(true_regions)
    annotated: list[dict] = []
    for report in reports:
        record = _report_record(report)
        start, end = record["interval_start"], record["interval_end"]
        length = record["interval_length"]
        overlaps = tuple(overlap_tokens(start, end, r.start, r.end) for r in regions)
        overlap_indices = tuple(i + 1 for i, value in enumerate(overlaps) if value > 0)
        signal_tokens = int(sum(overlaps))
        ious = tuple(
            overlap / (length + region.length - overlap)
            for overlap, region in zip(overlaps, regions)
        )
        if ious and max(ious) > 0.0:
            best_zero_index = int(np.argmax(ious))
            best_region_index: int | None = best_zero_index + 1
            best_iou = float(ious[best_zero_index])
            best_region = regions[best_zero_index]
            start_error: float = float(start - best_region.start)
            end_error: float = float(end - best_region.end)
            contains_best = start <= best_region.start and end >= best_region.end
        else:
            best_region_index = None
            best_iou = 0.0
            start_error = math.nan
            end_error = math.nan
            contains_best = False

        merged_gaps = tuple(
            i + 1
            for i in range(max(0, len(regions) - 1))
            if overlaps[i] > 0 and overlaps[i + 1] > 0
        )
        fully_covered_gaps = tuple(
            i + 1
            for i, (left, right) in enumerate(zip(regions, regions[1:]))
            if right.start > left.end + 1
            and start <= left.end + 1
            and end >= right.start - 1
        )
        record.update(
            {
                "overlap_tokens_by_region": overlaps,
                "overlapping_region_indices": overlap_indices,
                "overlapping_region_count": len(overlap_indices),
                "signal_overlap_tokens": signal_tokens,
                "null_tokens_in_report": length - signal_tokens,
                "localized_true": signal_tokens > 0,
                "localized_false": signal_tokens == 0,
                "purity": signal_tokens / length,
                "iou_by_region": ious,
                "best_region_index": best_region_index,
                "best_region_iou": best_iou,
                "start_error_to_best_region": start_error,
                "end_error_to_best_region": end_error,
                "absolute_start_error_to_best_region": abs(start_error),
                "absolute_end_error_to_best_region": abs(end_error),
                "contains_entire_best_region": contains_best,
                "merge": len(overlap_indices) >= 2,
                "merged_gap_indices": merged_gaps,
                "merged_gap_count": len(merged_gaps),
                "full_gap_bridge": bool(merged_gaps),
                "fully_covered_null_gap_indices": fully_covered_gaps,
                "fully_covered_null_gap_count": len(fully_covered_gaps),
            }
        )
        annotated.append(record)
    return annotated


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else math.nan


def path_metrics(
    reports: Sequence[Report | Mapping],
    true_regions: Sequence[Region | Sequence[int]],
    *,
    horizon: int,
) -> dict:
    """Compute report-, region-, and token-level metrics for one path.

    The report sequence is ordered by ``(tau, report)`` before FDP calculations.
    Token metrics use the union of all reported intervals, avoiding double
    counting when atomic reports overlap.
    """

    regions = _normalize_regions(true_regions)
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if regions and regions[-1].end > horizon:
        raise ValueError("a true region extends beyond the requested horizon")

    annotated = annotate_reports(reports, regions)
    annotated.sort(key=lambda item: (int(item["tau"]), int(item.get("report", 0))))
    if annotated and max(int(item["interval_end"]) for item in annotated) > horizon:
        raise ValueError("a report extends beyond the requested horizon")

    false_indicators = np.asarray(
        [bool(item["localized_false"]) for item in annotated], dtype=np.int64
    )
    if false_indicators.size:
        sequential_fdp = np.cumsum(false_indicators) / np.arange(1, false_indicators.size + 1)
        final_fdp = float(sequential_fdp[-1])
        uniform_fdp = float(sequential_fdp.max())
    else:
        sequential_fdp = np.empty(0, dtype=np.float64)
        final_fdp = uniform_fdp = 0.0

    truth = np.zeros(horizon + 1, dtype=bool)
    predicted = np.zeros(horizon + 1, dtype=bool)
    for region in regions:
        truth[region.start : region.end + 1] = True
    for item in annotated:
        predicted[item["interval_start"] : item["interval_end"] + 1] = True
    true_positive = int(np.sum(truth & predicted))
    false_positive = int(np.sum(~truth & predicted))
    false_negative = int(np.sum(truth & ~predicted))
    predicted_tokens = int(predicted.sum())
    true_tokens = int(truth.sum())
    union_tokens = true_positive + false_positive + false_negative

    report_counts_by_region = tuple(
        sum(item["overlap_tokens_by_region"][index] > 0 for item in annotated)
        for index in range(len(regions))
    )
    region_detected_any = tuple(count > 0 for count in report_counts_by_region)
    region_detected_timely = []
    region_delays = []
    region_best_ious = []
    region_fully_covered = []
    region_start_errors = []
    region_end_errors = []
    for index, region in enumerate(regions):
        timely = [
            item
            for item in annotated
            if item["overlap_tokens_by_region"][index] > 0
            and region.start <= int(item["tau"]) <= region.end
        ]
        region_detected_timely.append(bool(timely))
        region_delays.append(
            int(min(int(item["tau"]) for item in timely) - region.start) if timely else math.nan
        )
        if annotated:
            region_ious = [float(item["iou_by_region"][index]) for item in annotated]
            best_report_index = int(np.argmax(region_ious))
            best_report = annotated[best_report_index]
            region_best_ious.append(region_ious[best_report_index])
            if region_ious[best_report_index] > 0:
                region_start_errors.append(int(best_report["interval_start"] - region.start))
                region_end_errors.append(int(best_report["interval_end"] - region.end))
            else:
                region_start_errors.append(math.nan)
                region_end_errors.append(math.nan)
        else:
            region_best_ious.append(0.0)
            region_start_errors.append(math.nan)
            region_end_errors.append(math.nan)
        region_fully_covered.append(bool(np.all(predicted[region.start : region.end + 1])))

    merged_gap_indices = sorted(
        {index for item in annotated for index in item["merged_gap_indices"]}
    )
    covered_gap_indices = sorted(
        {index for item in annotated for index in item["fully_covered_null_gap_indices"]}
    )
    merge_reports = int(sum(bool(item["merge"]) for item in annotated))
    false_reports = int(false_indicators.sum())
    fragmented_regions = int(sum(count > 1 for count in report_counts_by_region))
    fragmentation_excess = int(sum(max(0, count - 1) for count in report_counts_by_region))

    return {
        "reports": len(annotated),
        "true_reports": len(annotated) - false_reports,
        "false_reports": false_reports,
        "any_false_report": false_reports > 0,
        "final_fdp": final_fdp,
        "uniform_fdp": uniform_fdp,
        "sequential_fdp": tuple(float(value) for value in sequential_fdp),
        "report_times": tuple(int(item["tau"]) for item in annotated),
        "merge_reports": merge_reports,
        "merge_report_rate": _safe_divide(merge_reports, len(annotated)),
        "any_merge": merge_reports > 0,
        "merged_gap_indices": tuple(merged_gap_indices),
        "merged_gap_count": len(merged_gap_indices),
        "any_full_gap_bridge": bool(merged_gap_indices),
        "fully_covered_null_gap_indices": tuple(covered_gap_indices),
        "fully_covered_null_gap_count": len(covered_gap_indices),
        "any_full_null_gap_covered": bool(covered_gap_indices),
        "fragmented_regions": fragmented_regions,
        "fragmentation_excess_reports": fragmentation_excess,
        "report_counts_by_region": report_counts_by_region,
        "regions": len(regions),
        "regions_detected_any": int(sum(region_detected_any)),
        "regionwise_recall": _safe_divide(sum(region_detected_any), len(regions)),
        "all_regions_detected": bool(regions) and all(region_detected_any),
        "regions_detected_timely": int(sum(region_detected_timely)),
        "regionwise_timely_recall": _safe_divide(sum(region_detected_timely), len(regions)),
        "all_regions_detected_timely": bool(regions) and all(region_detected_timely),
        "region_detected_any": tuple(region_detected_any),
        "region_detected_timely": tuple(region_detected_timely),
        "region_detection_delays": tuple(region_delays),
        "region_best_ious": tuple(region_best_ious),
        "mean_region_best_iou": (
            float(np.mean(region_best_ious)) if region_best_ious else math.nan
        ),
        "region_start_errors": tuple(region_start_errors),
        "region_end_errors": tuple(region_end_errors),
        "region_fully_covered": tuple(region_fully_covered),
        "regions_fully_covered": int(sum(region_fully_covered)),
        "all_regions_fully_covered": bool(regions) and all(region_fully_covered),
        "true_positive_tokens": true_positive,
        "false_positive_tokens": false_positive,
        "false_negative_tokens": false_negative,
        "predicted_tokens": predicted_tokens,
        "true_watermark_tokens": true_tokens,
        "token_precision": _safe_divide(true_positive, predicted_tokens),
        "token_recall": _safe_divide(true_positive, true_tokens),
        "token_iou": 1.0 if union_tokens == 0 else true_positive / union_tokens,
        "token_f1": (
            _safe_divide(2 * true_positive, 2 * true_positive + false_positive + false_negative)
        ),
        "mean_report_purity": (
            float(np.mean([item["purity"] for item in annotated]))
            if annotated
            else math.nan
        ),
        "mean_report_best_iou": (
            float(np.mean([item["best_region_iou"] for item in annotated]))
            if annotated
            else math.nan
        ),
    }


def _fdp_timeline(annotated_reports: Sequence[Mapping], horizon: int) -> np.ndarray:
    timeline = np.zeros(horizon, dtype=np.float64)
    reports = sorted(
        annotated_reports,
        key=lambda item: (int(item["tau"]), int(item.get("report", 0))),
    )
    false_count = 0
    previous_t = 1
    current_fdp = 0.0
    for index, item in enumerate(reports, start=1):
        tau = int(item["tau"])
        if not 1 <= tau <= horizon:
            raise ValueError("report time lies outside the batch horizon")
        if tau > previous_t:
            timeline[previous_t - 1 : tau - 1] = current_fdp
        false_count += int(bool(item["localized_false"]))
        current_fdp = false_count / index
        timeline[tau - 1 :] = current_fdp
        previous_t = tau
    return timeline


def batch_metrics(
    reports_by_path: Sequence[Sequence[Report | Mapping]],
    regions_by_path: Sequence[Sequence[Region | Sequence[int]]] | Sequence[Region | Sequence[int]],
    *,
    horizon: int,
) -> dict:
    """Aggregate pathwise localization metrics and the pointwise FDR curve.

    ``regions_by_path`` may either be one common region sequence or a sequence
    with one region sequence per path.  Paths, never atomic reports, are the
    Monte Carlo sampling units.
    """

    n_paths = len(reports_by_path)
    if n_paths < 1:
        raise ValueError("at least one path is required")

    # Disambiguate a common list [(a,b), ...] from a path-specific nested list.
    region_input = list(regions_by_path)
    common_regions = False
    if not region_input:
        common_regions = True
    else:
        first = region_input[0]
        common_regions = isinstance(first, Region) or (
            isinstance(first, Sequence)
            and len(first) == 2
            and all(np.isscalar(value) for value in first)
        )
    if common_regions:
        region_sets = [region_input] * n_paths
    else:
        if len(region_input) != n_paths:
            raise ValueError("path-specific regions must have one entry per path")
        region_sets = region_input

    path_rows: list[dict] = []
    curve_sum = np.zeros(horizon, dtype=np.float64)
    curve_sum_squares = np.zeros(horizon, dtype=np.float64)
    for path_index, (reports, regions) in enumerate(zip(reports_by_path, region_sets), start=1):
        annotated = annotate_reports(reports, regions)
        metrics = path_metrics(annotated, regions, horizon=horizon)
        metrics["path"] = path_index
        path_rows.append(metrics)
        timeline = _fdp_timeline(annotated, horizon)
        curve_sum += timeline
        curve_sum_squares += timeline * timeline

    pointwise_fdr = curve_sum / n_paths
    if n_paths > 1:
        variance = np.maximum(
            0.0,
            (curve_sum_squares - n_paths * pointwise_fdr**2) / (n_paths - 1),
        )
        pointwise_mcse = np.sqrt(variance / n_paths)
    else:
        pointwise_mcse = np.full(horizon, np.nan, dtype=np.float64)

    uniform_values = np.asarray([row["uniform_fdp"] for row in path_rows], dtype=float)
    final_values = np.asarray([row["final_fdp"] for row in path_rows], dtype=float)
    any_false = np.asarray([row["any_false_report"] for row in path_rows], dtype=float)

    def mean_and_mcse(values: np.ndarray) -> tuple[float, float]:
        mean = float(values.mean())
        mcse = float(values.std(ddof=1) / math.sqrt(values.size)) if values.size > 1 else math.nan
        return mean, mcse

    uniform_mean, uniform_mcse = mean_and_mcse(uniform_values)
    final_mean, final_mcse = mean_and_mcse(final_values)
    false_probability, false_probability_mcse = mean_and_mcse(any_false)

    numeric_mean_fields = [
        "reports",
        "false_reports",
        "merge_reports",
        "fragmented_regions",
        "regionwise_recall",
        "regionwise_timely_recall",
        "token_precision",
        "token_recall",
        "token_iou",
        "mean_region_best_iou",
        "mean_report_purity",
    ]
    localization_means = {}
    for field in numeric_mean_fields:
        values = np.asarray([row[field] for row in path_rows], dtype=float)
        localization_means[field] = float(np.nanmean(values)) if np.any(~np.isnan(values)) else math.nan

    return {
        "n_paths": n_paths,
        "horizon": horizon,
        "uniform_fdr": uniform_mean,
        "uniform_fdr_mcse": uniform_mcse,
        "final_fdr": final_mean,
        "final_fdr_mcse": final_mcse,
        "probability_any_false_report": false_probability,
        "probability_any_false_report_mcse": false_probability_mcse,
        "supremum_pointwise_fdr": float(pointwise_fdr.max()),
        "time_of_maximum_pointwise_fdr": int(np.argmax(pointwise_fdr) + 1),
        "time": np.arange(1, horizon + 1, dtype=np.int64),
        "pointwise_fdr": pointwise_fdr,
        "pointwise_fdr_mcse": pointwise_mcse,
        "localization_path_means": localization_means,
        "path_metrics": path_rows,
    }


def general_dependence_bound(alpha: float) -> float:
    """The explicit general-dependence upper bound ``alpha(1+log(1/alpha))``."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    return float(alpha * (1.0 + math.log(1.0 / alpha)))


def alpha_for_general_fdr_target(target: float, *, tolerance: float = 1e-14) -> float:
    """Invert the explicit general-dependence bound for a desired target."""

    if not 0.0 < target < 1.0:
        raise ValueError("target must lie in (0, 1)")
    lower, upper = FLOAT_TINY, 1.0 - np.finfo(np.float64).eps
    for _ in range(200):
        midpoint = 0.5 * (lower + upper)
        if general_dependence_bound(midpoint) < target:
            lower = midpoint
        else:
            upper = midpoint
        if upper - lower <= tolerance:
            break
    return 0.5 * (lower + upper)


def threshold_for_general_fdr_target(target: float) -> float:
    """Return ``1/alpha`` for the explicit general-dependence FDR target."""

    return 1.0 / alpha_for_general_fdr_target(target)
