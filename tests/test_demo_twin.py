import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps"))

from demo_twin import _grid_stride, _sample_dates  # noqa: E402


def test_sample_dates_drops_months_after_today():
    dates = _sample_dates("2026-01-01", "all", 1, [2025, 2026], today="2026-06-20")
    assert "2025-12-01" in dates  # full prior year kept
    assert "2026-06-01" in dates and "2026-07-01" not in dates  # current year clamped to today
    assert all(d <= "2026-06-20" for d in dates)


def test_sample_dates_single_date_when_months_none():
    assert _sample_dates("2020-03-15", "none", 1, [2020]) == ["2020-03-15"]


def test_grid_stride_rectangular_vs_ragged():
    full = _sample_dates("2025-01-01", "all", 1, [2023, 2024, 2025], today="2026-06-20")
    assert _grid_stride(full, [2023, 2024, 2025]) == 12  # 3 full years -> rectangular, year sub-picker
    ragged = _sample_dates("2026-01-01", "all", 1, [2023, 2024, 2025, 2026], today="2026-06-20")
    assert _grid_stride(ragged, [2023, 2024, 2025, 2026]) is None  # 2026 partial -> flat Month axis
    assert _grid_stride(["2020-01-01"], [2020]) is None  # single year -> flat
