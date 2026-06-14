"""Earth-observation features per H3 cell — static covariates that ground the demand proxy.

First: population (GHSL, via the Kontur Population dataset). A cell's demand should scale with
the people actually there, not a synthetic gradient. Kontur is H3-native (GHSL fused with
building footprints / OSM), so it maps onto the twin's grid by H3 ancestry — no raster
sampling. Download the Kontur hex GeoPackage (HDX), convert to an (h3, population) table, and
`population_by_cell` aggregates it to the twin's resolution. The geo-FM half of the cascade
(AlphaEarth embeddings, night-time lights, land cover) plugs in here as more feature loaders.
"""

import h3
import polars as pl

# Kontur Population — global density for 400 m / 3 km / 22 km H3 hexagons (HDX, H3-native)
KONTUR_URL = "https://data.humdata.org/dataset/kontur-population-dataset"


def population_by_cell(kontur: pl.DataFrame, res: int) -> dict[str, float]:
    """Aggregate a Kontur (h3, population) frame up to resolution `res` — sum the finer Kontur
    hexes into each ancestor cell of the twin's grid. Returns {cell_h3: population}."""
    parent = pl.col("h3").map_elements(lambda c: h3.cell_to_parent(c, res), return_dtype=pl.String)
    agg = kontur.with_columns(parent.alias("cell")).group_by("cell").agg(pl.col("population").sum())
    return dict(agg.iter_rows())
