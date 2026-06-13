import h3

from sctwin.geo import Cell, validate_res


def cells_in_bbox(south: float, west: float, north: float, east: float, res: int) -> list[Cell]:
    validate_res(res)
    poly = h3.LatLngPoly([(south, west), (south, east), (north, east), (north, west)])
    return [Cell(h3=cell, res=res) for cell in h3.polygon_to_cells(poly, res)]
