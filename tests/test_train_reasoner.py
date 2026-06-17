import sys
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps"))

from train_reasoner import intervention_env  # noqa: E402

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
