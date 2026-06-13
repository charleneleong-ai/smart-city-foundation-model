import pytest

from sctwin.geo import cell_of, center_of


def test_cell_of_is_deterministic_for_same_point():
    c1 = cell_of(51.5074, -0.1278, res=9)
    c2 = cell_of(51.5074, -0.1278, res=9)
    assert c1 == c2
    # nearby points in the same ~170m hex collapse to one cell
    assert cell_of(51.5076, -0.1276, res=9) == c1


def test_center_round_trips_within_cell():
    c = cell_of(51.5074, -0.1278, res=9)
    lat, lon = center_of(c)
    assert cell_of(lat, lon, res=9) == c


def test_resolution_changes_cell():
    assert cell_of(51.5074, -0.1278, res=7) != cell_of(51.5074, -0.1278, res=9)


@pytest.mark.parametrize("res", [-1, 16])
def test_invalid_resolution_rejected(res):
    with pytest.raises(ValueError):
        cell_of(51.5074, -0.1278, res=res)
