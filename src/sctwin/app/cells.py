import h3

from sctwin.geo import Cell, validate_res


def cells_in_bbox(south: float, west: float, north: float, east: float, res: int) -> list[Cell]:
    validate_res(res)
    poly = h3.LatLngPoly([(south, west), (south, east), (north, east), (north, west)])
    return [Cell(h3=cell, res=res) for cell in h3.polygon_to_cells(poly, res)]


def global_cells(res: int) -> list[Cell]:
    """Every H3 cell on Earth at `res` — the children of the 122 base cells. (A global bbox is a
    degenerate polygon, so polygon_to_cells can't tile the whole planet.)"""
    validate_res(res)
    return [Cell(h3=c, res=res) for r0 in h3.get_res0_cells() for c in h3.cell_to_children(r0, res)]
