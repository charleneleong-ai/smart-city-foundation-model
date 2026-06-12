from datetime import datetime, timezone

import polars as pl
import pytest
from pydantic import ValidationError

from sctwin.schema import CANONICAL_COLUMNS, LayerRecord, validate_frame


def test_record_coerces_time_to_utc():
    rec = LayerRecord(
        cell="891f1d4894bffff", time=datetime(2020, 1, 1, 12), layer="t2m", value=4.5
    )
    assert rec.time.tzinfo == timezone.utc


def test_record_rejects_missing_value():
    with pytest.raises(ValidationError):
        LayerRecord(cell="891f1d4894bffff", time=datetime(2020, 1, 1), layer="t2m")


def test_validate_frame_accepts_canonical_columns():
    df = pl.DataFrame(
        {
            "cell": ["891f1d4894bffff"],
            "time": [datetime(2020, 1, 1, tzinfo=timezone.utc)],
            "layer": ["t2m"],
            "value": [4.5],
        }
    )
    out = validate_frame(df)
    assert out.columns == CANONICAL_COLUMNS


def test_validate_frame_rejects_missing_column():
    df = pl.DataFrame({"cell": ["x"], "time": [datetime(2020, 1, 1)], "value": [1.0]})
    with pytest.raises(ValueError, match="missing columns"):
        validate_frame(df)
