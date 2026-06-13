import pytest

from sctwin.app.cells import cells_in_bbox
from sctwin.geo import center_of

BOX = dict(south=51.50, west=-0.13, north=51.52, east=-0.10)


def test_cells_cover_bbox_and_lie_inside():
    cells = cells_in_bbox(**BOX, res=8)
    assert len(cells) > 0
    for c in cells:  # every returned cell's center is within the bbox (small margin)
        lat, lon = center_of(c)
        assert 51.49 <= lat <= 51.53
        assert -0.14 <= lon <= -0.09


def test_finer_resolution_returns_more_cells():
    assert len(cells_in_bbox(**BOX, res=9)) > len(cells_in_bbox(**BOX, res=7))


def test_invalid_resolution_rejected():
    with pytest.raises(ValueError):
        cells_in_bbox(**BOX, res=99)
