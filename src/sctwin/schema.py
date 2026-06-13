from datetime import datetime, timezone

import polars as pl
from pydantic import BaseModel, field_validator

CANONICAL_SCHEMA: dict[str, pl.DataType] = {
    "cell": pl.String(),
    "time": pl.Datetime("us", "UTC"),
    "layer": pl.String(),
    "value": pl.Float64(),
}
CANONICAL_COLUMNS = list(CANONICAL_SCHEMA)


def empty_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=CANONICAL_SCHEMA)


class LayerRecord(BaseModel):
    cell: str
    time: datetime
    layer: str
    value: float

    @field_validator("time")
    @classmethod
    def _to_utc(cls, v: datetime) -> datetime:
        return v.replace(tzinfo=timezone.utc) if v.tzinfo is None else v.astimezone(timezone.utc)


def validate_frame(df: pl.DataFrame) -> pl.DataFrame:
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    return df.select(CANONICAL_COLUMNS)
