"""Interventional verifiable-reward environment (paper mechanism 3 — causal, not forecast).

Forecasting asks "what *will* load be?"; planning asks "what *changes* if I retrofit district D
or shift the tariff?". This poses the second question as a verifiable-reward task: a policy
predicts an intervention's effect (Δ on a decision metric), and the reward scores that predicted
Δ against a counterfactual *oracle* Δ (sign + magnitude, via `interventional_reward`). The oracle
is a transparent physics proxy applied to the cell's real demand — a stand-in for a calibrated
simulator (EnergyPlus) or a real before/after natural experiment, which slot in behind the same
interface.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
import polars as pl

from sctwin.reason.reward import interventional_reward

_PEAK_HOURS = tuple(range(17, 22))  # 17:00–21:00 evening peak (tariff target window)
_HEATING_BASE_C = 18.0  # heating-degree threshold (matches the EV-charging demand model)
_METRICS: dict[str, Callable[[np.ndarray], float]] = {
    "peak": lambda a: float(a.max()),
    "mean": lambda a: float(a.mean()),
    "total": lambda a: float(a.sum()),
}


@dataclass(frozen=True)
class Intervention:
    """A counterfactual policy lever on one cell's demand:
    - `retrofit`: cut the heating-sensitive part of load by `factor` (insulation/envelope).
    - `tariff`:   time-of-use shift — move `factor` of peak-hour load to off-peak (energy-conserving).
    `metric` is the decision-relevant summary the effect is measured on (peak / mean / total)."""

    kind: str
    cell: str
    factor: float
    metric: str = "peak"

    def __post_init__(self) -> None:
        if self.kind not in ("retrofit", "tariff"):
            raise ValueError(f"kind must be 'retrofit' or 'tariff', got {self.kind!r}")
        if self.metric not in _METRICS:
            raise ValueError(f"metric must be one of {tuple(_METRICS)}, got {self.metric!r}")
        if not 0.0 <= self.factor <= 1.0:
            raise ValueError(f"factor must be in [0, 1], got {self.factor}")


def _heating_degrees(temp: np.ndarray) -> np.ndarray:
    return np.maximum(_HEATING_BASE_C - temp, 0.0)


def _heating_sensitivity(load: np.ndarray, hdd: np.ndarray) -> float:
    """OLS slope of load on heating-degrees (load per degree). cov and var are both population
    moments (bias=True / ddof=0), so their ratio is the exact least-squares slope."""
    if hdd.var() == 0:
        return 0.0
    return float(np.cov(load, hdd, bias=True)[0, 1] / hdd.var())


def _cell_series(frame: pl.DataFrame, cell: str) -> pl.DataFrame:
    return frame.filter(pl.col("cell") == cell).sort("time")


def counterfactual(demand: pl.DataFrame, weather: pl.DataFrame, iv: Intervention) -> pl.DataFrame:
    """Apply the intervention's physics proxy to the cell's hourly demand, returning the
    counterfactual (same schema, `value` replaced). `retrofit` needs the cell's weather (heating
    degrees); `tariff` is weather-free. Counterfactual load is clipped non-negative."""
    d = _cell_series(demand, iv.cell)
    load = d["value"].to_numpy().astype(float)
    if iv.kind == "retrofit":
        temp = (
            d.select("time")
            .join(_cell_series(weather, iv.cell).select("time", "value"), on="time", how="left")
            .sort("time")["value"]
            .fill_null(strategy="forward")
            .fill_null(
                _HEATING_BASE_C
            )  # leading gaps (no prior temp) -> base temp -> zero heating degrees
            .to_numpy()
            .astype(float)
        )
        hdd = _heating_degrees(temp)
        cf = load - iv.factor * _heating_sensitivity(load, hdd) * hdd  # remove heating-driven load
    else:  # "tariff" (kind validated in __post_init__)
        peak = d["time"].dt.hour().is_in(_PEAK_HOURS).to_numpy()
        cf = load.copy()
        if (off := ~peak).any() and peak.any():  # a shift needs both peak and off-peak hours
            cf[peak] *= 1.0 - iv.factor
            cf[off] += (
                iv.factor * load[peak].sum() / off.sum()
            )  # redistribute peak energy (conserves total)
    return d.with_columns(pl.Series("value", np.clip(cf, 0.0, None)))


def effect(baseline: pl.DataFrame, cf: pl.DataFrame, metric: str) -> float:
    """Signed change in the decision metric (counterfactual − baseline); negative = a reduction."""
    agg = _METRICS[metric]
    return agg(cf["value"].to_numpy()) - agg(baseline["value"].to_numpy())


@dataclass(frozen=True)
class InterventionQuestion:
    """Ask the reasoner: what is `intervention`'s effect on the cell's `metric`? The oracle's
    counterfactual Δ is the verifiable ground truth, hidden from the policy."""

    intervention: Intervention
    true_delta: float
    scale: float


IPolicy = Callable[[InterventionQuestion], float]


class InterventionEnvironment:
    """The city as a verifiable-reward environment for *interventional* claims. Each question
    poses an intervention; a policy predicts its effect (Δ metric); the reward scores the
    predicted Δ against the counterfactual-oracle Δ via `interventional_reward` (half sign, half
    magnitude). `rollout` returns the policy's mean reward — its RLVR return. This is what makes
    interventional validity, not forecast accuracy, the trained objective."""

    def __init__(
        self,
        demand: pl.DataFrame,
        weather: pl.DataFrame,
        interventions: list[Intervention],
        *,
        scale: float | None = None,
    ) -> None:
        self._questions: list[InterventionQuestion] = []
        for iv in interventions:
            base = _cell_series(demand, iv.cell)
            true_delta = effect(base, counterfactual(demand, weather, iv), iv.metric)
            std = base[
                "value"
            ].std()  # float at runtime; std stubs widen to float | timedelta | None
            sc = scale if scale is not None else (cast(float, std) if std else 1.0)
            self._questions.append(InterventionQuestion(iv, true_delta, sc))

    def questions(self) -> list[InterventionQuestion]:
        return self._questions

    def reward(self, q: InterventionQuestion, pred_delta: float) -> float:
        return interventional_reward(pred_delta, q.true_delta, scale=q.scale)

    def rollout(self, policy: IPolicy) -> dict[str, float]:
        rewards = [self.reward(q, policy(q)) for q in self._questions]
        return {"mean_reward": sum(rewards) / len(rewards), "n": float(len(rewards))}
