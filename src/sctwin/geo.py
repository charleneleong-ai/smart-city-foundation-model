from dataclasses import dataclass

import h3


@dataclass(frozen=True)
class Cell:
    h3: str
    res: int


def validate_res(res: int) -> None:
    if not 0 <= res <= 15:
        raise ValueError(f"H3 resolution must be 0..15, got {res}")


def cell_of(lat: float, lon: float, res: int) -> Cell:
    validate_res(res)
    return Cell(h3=h3.latlng_to_cell(lat, lon, res), res=res)


def center_of(cell: Cell) -> tuple[float, float]:
    return h3.cell_to_latlng(cell.h3)
