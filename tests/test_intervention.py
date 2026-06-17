from datetime import datetime

import polars as pl
import pytest

from sctwin.reason.intervention import (
    Intervention,
    InterventionEnvironment,
    counterfactual,
    effect,
)

_C = "abc"


def _frame(times: list[datetime], values: list[float], *, layer: str) -> pl.DataFrame:
    return pl.DataFrame({"cell": _C, "time": times, "layer": layer, "value": values})


def _hours(*hh: int) -> list[datetime]:
    return [datetime(2023, 1, 1, h) for h in hh]


# load = 100 + 2*HDD with HDD = [8, 4, 16, 0] (temps below) -> known heating sensitivity beta = 2
_RETROFIT_TIMES = _hours(0, 1, 2, 3)
_TEMPS = [10.0, 14.0, 2.0, 18.0]  # HDD = max(18 - T, 0) = [8, 4, 16, 0]
_LOAD = [116.0, 108.0, 132.0, 100.0]
_DEMAND = _frame(_RETROFIT_TIMES, _LOAD, layer="load")
_WEATHER = _frame(_RETROFIT_TIMES, _TEMPS, layer="weather.t2m")


class TestRetrofit:
    """Retrofit removes factor × (heating-driven load) = factor × beta × HDD."""

    def test_cuts_heating_driven_load_by_expected_amount(self):
        cf = counterfactual(_DEMAND, _WEATHER, Intervention("retrofit", _C, factor=0.5))
        # cf = load - 0.5*2*HDD = load - HDD
        assert cf["value"].to_list() == [108.0, 104.0, 116.0, 100.0]

    @pytest.mark.parametrize("metric,expected", [("mean", -7.0), ("peak", -16.0), ("total", -28.0)])
    def test_effect_is_a_reduction_matching_the_physics(self, metric, expected):
        iv = Intervention("retrofit", _C, factor=0.5, metric=metric)
        assert effect(_DEMAND, counterfactual(_DEMAND, _WEATHER, iv), metric) == pytest.approx(
            expected
        )

    def test_no_heating_degrees_means_no_effect(self):
        warm = _frame(
            _RETROFIT_TIMES, [20.0, 22.0, 25.0, 19.0], layer="weather.t2m"
        )  # all >= 18 -> HDD 0
        cf = counterfactual(_DEMAND, warm, Intervention("retrofit", _C, factor=0.9))
        assert cf["value"].to_list() == _LOAD

    def test_missing_weather_rows_are_backfilled_not_crashed(self):
        gappy = _frame(_RETROFIT_TIMES[1:], _TEMPS[1:], layer="weather.t2m")  # leading time absent
        cf = counterfactual(_DEMAND, gappy, Intervention("retrofit", _C, factor=0.5))
        assert cf["value"][0] == _LOAD[0]  # leading gap -> base temp -> no retrofit at that hour
        assert all(v >= 0 for v in cf["value"].to_list())


class TestTariff:
    """Time-of-use shift moves peak-hour load to off-peak — conserves total energy, cuts the peak."""

    _TARIFF = _frame(
        _hours(3, 4, 18, 19), [2.0, 2.0, 10.0, 10.0], layer="load"
    )  # 18/19 are peak hrs

    def test_conserves_total_but_reduces_peak(self):
        iv_total = Intervention("tariff", _C, factor=0.5, metric="total")
        iv_peak = Intervention("tariff", _C, factor=0.5, metric="peak")
        assert effect(
            self._TARIFF, counterfactual(self._TARIFF, _WEATHER, iv_total), "total"
        ) == pytest.approx(0.0)
        assert effect(
            self._TARIFF, counterfactual(self._TARIFF, _WEATHER, iv_peak), "peak"
        ) == pytest.approx(-3.0)

    def test_ignores_weather(self):
        iv = Intervention("tariff", _C, factor=0.5, metric="peak")
        garbage = _frame(
            [t for t in self._TARIFF["time"]], [99.0, 99.0, 99.0, 99.0], layer="weather.t2m"
        )
        a = counterfactual(self._TARIFF, _WEATHER, iv)["value"].to_list()
        assert a == counterfactual(self._TARIFF, garbage, iv)["value"].to_list()

    def test_all_peak_cell_is_a_noop(self):
        all_peak = _frame(
            _hours(18, 19, 20), [10.0, 10.0, 10.0], layer="load"
        )  # no off-peak to shift to
        cf = counterfactual(all_peak, _WEATHER, Intervention("tariff", _C, factor=0.5))
        assert cf["value"].to_list() == [10.0, 10.0, 10.0]  # can't shift -> total + peak unchanged


@pytest.mark.parametrize(
    "bad", [{"kind": "upzone"}, {"metric": "median"}, {"factor": 1.5}, {"factor": -0.1}]
)
def test_invalid_intervention_rejected(bad):
    with pytest.raises(ValueError):
        Intervention(**{"kind": "retrofit", "cell": _C, "factor": 0.5, **bad})


class TestEnvironment:
    """The verifiable-reward env scores predicted Δ against the oracle Δ (sign + magnitude)."""

    def _env(self) -> InterventionEnvironment:
        return InterventionEnvironment(
            _DEMAND, _WEATHER, [Intervention("retrofit", _C, 0.5, "mean")]
        )

    def test_question_carries_the_oracle_delta(self):
        q = self._env().questions()[0]
        assert q.true_delta == pytest.approx(
            -7.0
        )  # env wires the counterfactual effect into the question

    def test_perfect_policy_scores_one(self):
        env = self._env()
        out = env.rollout(lambda q: q.true_delta)  # an oracle policy
        assert out == {"mean_reward": pytest.approx(1.0), "n": 1.0}

    def test_wrong_sign_loses_the_direction_half(self):
        env = self._env()
        q = env.questions()[0]
        assert (
            env.reward(q, -q.true_delta) < 0.5
        )  # right magnitude, wrong sign -> below the 0.5 sign floor
