from datetime import datetime, timezone

import h3
import polars as pl

from sctwin.fire import (
    _bearing,
    dryness_field,
    fuel_dryness,
    simulate,
    spread_step,
    wind_factor,
)

RES = 8
CENTER = h3.latlng_to_cell(34.05, -118.24, RES)  # Los Angeles
NEIGHBOURS = [n for n in h3.grid_disk(CENTER, 1) if n != CENTER]


def test_fuel_dryness_is_monotone_and_bounded():
    assert fuel_dryness(35, 10, 0) > fuel_dryness(5, 100, 0)
    assert fuel_dryness(5, 100, 0) == 0.0  # no heat -> nothing ignites
    assert fuel_dryness(35, 10, 5) == 0.0  # >= 5 mm recent rain fully suppresses
    assert 0.0 <= fuel_dryness(45, 0, 0) <= 1.0  # clamped


def test_wind_factor_downwind_beats_upwind():
    # wind FROM the north (0 deg) -> fire runs south; a southerly bearing (180) is downwind
    assert wind_factor(180, 0) > wind_factor(90, 0) > wind_factor(0, 0)
    assert wind_factor(0, 0) == 0.0  # due upwind
    assert abs(wind_factor(180, 0) - 1.0) < 1e-9  # due downwind


def test_spread_step_prefers_the_downwind_neighbour():
    dryness = {n: 1.0 for n in NEIGHBOURS}
    wind_from = 0.0  # northerly -> fire pushes south
    ignited = spread_step({CENTER}, dryness, wind_from, wind_speed=40, threshold=0.5)
    factor = {n: wind_factor(_bearing(CENTER, n), wind_from) for n in NEIGHBOURS}
    assert max(NEIGHBOURS, key=factor.get) in ignited  # most downwind ignites
    assert min(NEIGHBOURS, key=factor.get) not in ignited  # most upwind does not


def test_spread_step_ignores_wet_and_already_burning_cells():
    assert spread_step({CENTER}, {n: 0.0 for n in NEIGHBOURS}, 0.0, wind_speed=40) == set()
    out = spread_step({CENTER, *NEIGHBOURS}, {n: 1.0 for n in NEIGHBOURS}, 0.0, wind_speed=40)
    assert out.isdisjoint({CENTER, *NEIGHBOURS})  # nothing already burning is re-reported


def test_simulate_records_increasing_arrival_steps():
    field = {c: 1.0 for c in h3.grid_disk(CENTER, 3)}  # uniform-dry 3-ring disk
    arrival = simulate({CENTER}, field, wind_from_deg=0.0, steps=3, wind_speed=40, threshold=0.4)
    assert arrival[CENTER] == 0
    assert len(arrival) > 1 and max(arrival.values()) >= 1  # fire grew over steps


def test_dryness_field_reads_weather_layers():
    at = datetime(2025, 1, 7, tzinfo=timezone.utc)
    frame = pl.DataFrame(
        {
            "cell": [CENTER, CENTER, CENTER],
            "time": [at, at, at],
            "layer": ["t2m", "rh", "precip"],
            "value": [35.0, 10.0, 0.0],
        }
    )
    field = dryness_field(frame, at)
    assert field[CENTER] == fuel_dryness(35.0, 10.0, 0.0) > 0.0
