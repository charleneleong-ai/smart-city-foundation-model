import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps"))

from presets import PRESETS  # noqa: E402

from sctwin.app.cells import cells_in_bbox  # noqa: E402


def test_every_preset_bbox_yields_bounded_cells():
    # default source is batched Open-Meteo (cap 1000); ERA5 lifts it further (one grid request)
    for name, p in PRESETS.items():
        n = len(cells_in_bbox(p["south"], p["west"], p["north"], p["east"], p["res"]))
        assert 0 < n <= 1000, f"{name}: {n} cells"


def test_presets_have_required_view_fields():
    required = {"south", "west", "north", "east", "res", "lat", "lon"}
    for name, p in PRESETS.items():
        assert required <= set(p), f"{name} missing {required - set(p)}"
