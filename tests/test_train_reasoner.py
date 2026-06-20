import sys
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps"))

from sctwin.adapters.demand import LCLTariffAdapter, NEEDRetrofitAdapter  # noqa: E402

from train_reasoner import build_real_intervention_samples, intervention_env  # noqa: E402

_TIMES = [datetime(2023, 1, 1, h) for h in range(4)]
_TEMPS = [10.0, 14.0, 2.0, 18.0]  # HDD = [8, 4, 16, 0], mean 7
_LOAD_A = [116.0, 108.0, 132.0, 100.0]  # 100 + 2*HDD -> beta = 2
_LOAD_B = [132.0, 116.0, 164.0, 100.0]  # 100 + 4*HDD -> beta = 4 (a colder-sensitive cell)


def _layer(cell: str, layer: str, values: list[float]) -> pl.DataFrame:
    return pl.DataFrame({"cell": cell, "time": _TIMES, "layer": layer, "value": values})


def test_intervention_env_builds_one_question_per_cell_with_per_cell_oracle_delta():
    demand = pl.concat([_layer("a", "load", _LOAD_A), _layer("b", "load", _LOAD_B)])
    weather = pl.concat([_layer("a", "weather.t2m", _TEMPS), _layer("b", "weather.t2m", _TEMPS)])
    env = intervention_env(demand, weather, kind="retrofit", factor=0.5)
    by_cell = {q.intervention.cell: q.true_delta for q in env.questions()}
    # mean Δ = -0.5*beta*mean(HDD): per-cell, not a shared constant (a: -0.5*2*7, b: -0.5*4*7)
    assert by_cell == {"a": pytest.approx(-7.0), "b": pytest.approx(-14.0)}


# LCL ToU 5->3 / Std 4->4 across 2012/2013 (DiD peak = -2); NEED treated -1250 vs control -150 (DiD = -1100)
_LCL_2Y = pl.DataFrame(
    {
        "stdorToU": ["ToU", "ToU", "Std", "Std"],
        "DateTime": [f"{y}-06-01 18:00:00.0000000" for y in (2012, 2013)] * 2,
        "value": [5.0, 3.0, 4.0, 4.0],
    }
)
_NEED_DID = pl.DataFrame(
    {
        "LOFT_FLAG": [1, 1, 0, 0],
        "Econ2010": [5000.0, 6000.0, 5500.0, 6500.0],
        "Econ2013": [4000.0, 4500.0, 5400.0, 6300.0],
    }
)


class _StubLCL(LCLTariffAdapter):
    def _read(self) -> pl.DataFrame:
        return _LCL_2Y


class _StubNEED(NEEDRetrofitAdapter):
    def _read(self) -> pl.DataFrame:
        return _NEED_DID


def test_build_real_intervention_samples_uses_measured_did_targets():
    cols, env = build_real_intervention_samples("c", tariff=_StubLCL("x"), retrofit=_StubNEED("x"))
    assert set(cols) == {"prompt", "true_delta", "scale"}
    by_kind = {q.intervention.kind: q.true_delta for q in env.questions()}
    assert by_kind == {
        "tariff": pytest.approx(-2.0),
        "retrofit": pytest.approx(-1100.0),
    }  # the DiD Δ


@pytest.mark.parametrize(
    "tariff,retrofit,kind", [(_StubLCL("x"), None, "tariff"), (None, _StubNEED("x"), "retrofit")]
)
def test_build_real_intervention_samples_single_adapter(tariff, retrofit, kind):
    cols, env = build_real_intervention_samples("c", tariff=tariff, retrofit=retrofit)
    assert [q.intervention.kind for q in env.questions()] == [
        kind
    ]  # omitting one adapter -> one question
    assert len(cols["true_delta"]) == 1


def test_build_real_intervention_samples_needs_an_adapter():
    with pytest.raises(ValueError, match="real oracle"):
        build_real_intervention_samples("c")
