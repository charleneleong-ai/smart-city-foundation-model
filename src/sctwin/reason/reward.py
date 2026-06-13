"""Verifiable rewards for the urban-reasoning environment (RLVR).

Each returns a value in [0, 1] (1 = perfect). They encode the spec's strongest novelty
mechanisms: accuracy/coverage against held-out measurements (mechanism 1), the signed
interventional effect vs a real before/after delta (mechanism 3, causal), and a
conservation-law check (mechanism 4, process reward). An RLVR policy is rewarded only when
its claims pass these *checkable* tests — that is what makes the city a verifiable-reward
environment.
"""

import math


def _decay(a: float, b: float, scale: float) -> float:
    """exp(-|a - b| / scale): 1 at an exact match, decaying with the gap; exact-equality
    when scale is non-positive."""
    return math.exp(-abs(a - b) / scale) if scale > 0 else float(a == b)


def accuracy_reward(predicted: float, actual: float, *, scale: float) -> float:
    """1 at an exact match, decaying with error (scale = the natural spread of the quantity,
    e.g. its std)."""
    return _decay(predicted, actual, scale)


def coverage_reward(predicted: float, lo: float, hi: float) -> float:
    """1 iff the prediction falls inside the calibrated interval [lo, hi]."""
    return 1.0 if lo <= predicted <= hi else 0.0


def interventional_reward(pred_delta: float, true_delta: float, *, scale: float) -> float:
    """Score a predicted intervention effect against the real before/after delta: half for
    the right *direction* (sign), half for the right *magnitude*. Causal validity, not just
    forecast accuracy."""
    direction = 1.0 if (pred_delta >= 0) == (true_delta >= 0) else 0.0
    return 0.5 * direction + 0.5 * _decay(pred_delta, true_delta, scale)


def conservation_reward(parts: list[float], whole: float, *, tol: float = 0.05) -> float:
    """1 iff the parts sum to the whole within tol (energy/mass balance), else decays with
    the relative residual — a physics process-reward on a reasoning step."""
    denom = abs(whole) or 1.0
    residual = abs(sum(parts) - whole) / denom
    return 1.0 if residual <= tol else max(0.0, 1.0 - residual)
