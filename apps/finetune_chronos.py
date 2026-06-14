"""Is zero-shot the ceiling? Fine-tune Chronos-2 (LoRA) on the *history* of the series and
compare zero-shot vs fine-tuned on the held-out test window. Leakage-safe: the fine-tune sees
only the context the forecast is conditioned on (the first 75%, exactly what verify() uses as
history), never the test window.

Run (GPU box recommended — CPU fine-tuning is slow):
    uv run --extra forecast --extra tsfm python apps/finetune_chronos.py --steps 200
"""

from datetime import datetime, timezone
from typing import Annotated

import polars as pl
import typer

from sctwin.adapters.cache import CachingAdapter
from sctwin.adapters.demand import AEMODemandAdapter
from sctwin.adapters.open_meteo import OpenMeteoWeatherAdapter
from sctwin.forecast.chronos import ChronosForecaster, _time_split
from sctwin.forecast.features import BASE_FEATURES, LAGS_BY_FREQ, build_supervised, regularize, resample
from sctwin.geo import cell_of


def main(
    region: Annotated[str, typer.Option(help="AEMO region")] = "NSW1",
    start: Annotated[str, typer.Option(help="YYYY-MM-DD")] = "2021-01-01",
    end: Annotated[str, typer.Option(help="YYYY-MM-DD")] = "2023-12-31",
    freq: Annotated[str, typer.Option(help="hour|day|week|month")] = "day",
    steps: Annotated[int, typer.Option(help="fine-tune steps")] = 200,
    mode: Annotated[str, typer.Option(help="lora|full")] = "lora",
    lr: Annotated[float, typer.Option(help="learning rate")] = 1e-4,
    device: Annotated[str, typer.Option(help="cpu|cuda")] = "cpu",
) -> None:
    """Compare zero-shot vs fine-tuned Chronos-2 on real AEMO demand + weather."""
    s = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    e = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    cells = [cell_of(-33.87, 151.21, 7)]
    demand = resample(AEMODemandAdapter(region=region).fetch(cells, s, e), freq, agg="sum")
    weather = resample(CachingAdapter(OpenMeteoWeatherAdapter(), ".cache/open-meteo").fetch(cells, s, e), freq)
    sup = build_supervised(regularize(demand, freq), weather, lags=LAGS_BY_FREQ[freq])

    import torch
    from chronos import Chronos2Pipeline

    base = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device)
    zero = ChronosForecaster(pipeline=base).verify(sup, covariates=BASE_FEATURES)

    # leakage-safe: fine-tune only on the history verify() conditions the forecast on
    history, _, horizon = _time_split(sup, 0.25)
    train = [
        torch.tensor(history.filter(pl.col("cell") == c)["y"].to_numpy(), dtype=torch.float32)
        for c in history["cell"].unique().to_list()
    ]
    print(f"fine-tuning ({mode}, {steps} steps) on {len(train)} series, horizon {horizon} ...")
    tuned = base.fit(train, prediction_length=horizon, finetune_mode=mode, num_steps=steps, learning_rate=lr)
    finetuned = ChronosForecaster(pipeline=tuned).verify(sup, covariates=BASE_FEATURES)

    print(f"\n{region} {freq} — is zero-shot the ceiling?")
    print(f"  Chronos-2 zero-shot    MAE {float(zero['abs_error'].mean()):.1f}")
    print(f"  Chronos-2 fine-tuned   MAE {float(finetuned['abs_error'].mean()):.1f}  ({mode}, {steps} steps)")


if __name__ == "__main__":
    typer.run(main)
