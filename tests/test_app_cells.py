import pytest

from sctwin.app.cells import cells_in_bbox, global_cells
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


def test_global_cells_tile_the_whole_planet():
    # the 122 base cells have 7^res children each (minus the 12 pentagons' missing child)
    assert len(global_cells(0)) == 122
    assert len(global_cells(2)) == 5882  # what a bbox polygon can't produce for the whole globe
    assert len(global_cells(3)) > len(global_cells(2))
