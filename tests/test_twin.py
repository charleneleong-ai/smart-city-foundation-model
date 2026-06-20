import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps"))

import twin  # noqa: E402
from presets import PRESETS  # noqa: E402


def _synth_weather(cells, start, end):
    """Open-Meteo-shaped hourly frame: 24 h for every date in [start, end] (what the archive returns)."""
    d, days = start.date(), []
    while d <= end.date():
        days += [datetime(d.year, d.month, d.day, h, tzinfo=timezone.utc) for h in range(24)]
        d += timedelta(days=1)
    rows = [
        {  # `c.h3` (str) matches the registry's cell dtype; per-cell offset keeps the GBM well-posed
            "cell": c.h3,
            "time": t,
            "layer": "t2m",
            "value": 8.0 + 3.0 * math.sin(t.hour / 24 * 2 * math.pi) + (i % 7) * 0.4,
        }
        for i, c in enumerate(cells)
        for t in days
    ]
    return pl.DataFrame(rows)


def test_input_layers_animate_the_full_multiday_window(monkeypatch):
    monkeypatch.setattr(twin, "_weather", _synth_weather)
    m = twin.twin_map("t", PRESETS["london"], "2025-03-01", days=3, res=8)

    temp = m["layers"][0]
    assert temp["name"] == "temperature" and temp["group"] == "Inputs"
    assert len(temp["frames"]) == 72  # 3 days × 24 h continuous (was a single 24 h day)
    assert "2025" in temp["frames"][-1]["label"]  # multi-day window -> labels carry year · month day · time

    iv = next(L for L in m["layers"] if L["group"].startswith("Intervention"))
    assert len(iv["frames"]) == 72  # intervention surface spans the same window, not just day 1

    # every Inputs frame readout names the year/month/day so a multi-day Play reads continuously
    assert all(("2025" in temp["frames"][k]["label"] and "Mar" in temp["frames"][k]["label"]) for k in (0, 36, 71))
