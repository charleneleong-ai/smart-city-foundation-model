"""SP7 — train the urban reasoning model with RLVR (GSPO/GRPO) against a verifiable
environment, using Unsloth (fast LoRA) + TRL's GRPOTrainer.

Two tasks (--task):
- forecast (default): predict a held-out cell-hour load; reward = accuracy + interval coverage
  vs the SP5 verification frame (sctwin.reason.grpo).
- intervention: predict an intervention's effect (Δ load from a retrofit / tariff); reward =
  sign + magnitude vs the counterfactual oracle (sctwin.reason.intervention_policy). This is the
  causal, planning-facing objective — interventional validity, not forecast accuracy.

The policy LLM reasons and emits \\boxed{number}; the reward parses + scores it (the verifiable
reward). GSPO (sequence-level importance) is the default; --algo grpo for token-level.

Needs a CUDA GPU with `unsloth` + `trl` installed (CUDA-specific, install on the box). The reward
glue + environments (sctwin.reason) are pure + tested without a GPU.

Run (on the GPU box):
    uv run --extra forecast python apps/train_reasoner.py --city london --task intervention \
        --model unsloth/Qwen2.5-1.5B-Instruct --steps 200
"""

from datetime import datetime, timedelta, timezone
from typing import Annotated

import polars as pl
import typer

from sctwin.adapters.cache import CachingAdapter
from sctwin.adapters.open_meteo import OpenMeteoWeatherAdapter
from sctwin.app.cells import cells_in_bbox
from sctwin.forecast.baselines import GBMForecaster
from sctwin.forecast.features import FEATURE_COLS, build_supervised
from sctwin.reason.environment import ReasoningEnvironment
from sctwin.reason.grpo import make_reward_fn, prompt_for
from sctwin.reason.intervention import Intervention, InterventionEnvironment
from sctwin.reason.intervention_policy import make_reward_fn as iv_reward_fn
from sctwin.reason.intervention_policy import oracle_effect, training_records, zero_effect
from sctwin.registry import Registry
from sctwin.verify.results import verification_frame

from presets import PRESETS
from twin import _synth_load


def _weather(city: str, date: str, days: int, res: int) -> pl.DataFrame:
    """Cached Open-Meteo 2 m temperature over the preset's cells for [date, date+days)."""
    p = PRESETS[city]
    cells = cells_in_bbox(p["south"], p["west"], p["north"], p["east"], res)
    start = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
    reg = Registry()
    reg.register(CachingAdapter(OpenMeteoWeatherAdapter(), ".cache/open-meteo"))
    return reg.get("weather.t2m", cells, start, start + timedelta(days=days - 1))


def build_samples(city: str, date: str, days: int, res: int) -> tuple[dict, ReasoningEnvironment]:
    """{prompt, actual, lo, hi} columns per held-out test cell-hour, built from the same SP4
    features + SP5 verification the twin uses, plus the environment. Pure CPU + no HF deps —
    used by both training and baseline eval. Returns (columns, environment)."""
    wx = _weather(city, date, days, res)
    supervised = build_supervised(_synth_load(wx, res), wx)
    results = verification_frame(GBMForecaster(), supervised, FEATURE_COLS)
    rows = supervised.join(
        results.select("cell", "time", "y_true", "lo", "hi"), on=["cell", "time"]
    ).to_dicts()

    columns = {
        "prompt": [
            prompt_for(
                f"cell {r['cell']}, hour {int(r['hour'])}, HDD {r['hdd']:.1f}, "
                f"prev-hour load {r['y_lag_1']:.0f}, same-hour-yesterday {r['y_lag_24']:.0f}"
            )
            for r in rows
        ],
        "actual": [float(r["y_true"]) for r in rows],
        "lo": [float(r["lo"]) for r in rows],
        "hi": [float(r["hi"]) for r in rows],
    }
    return columns, ReasoningEnvironment(results)


def build_dataset(city: str, date: str, days: int, res: int):
    """build_samples wrapped as a HuggingFace Dataset for TRL. Returns (hf_dataset, environment)."""
    from datasets import Dataset

    columns, env = build_samples(city, date, days, res)
    return Dataset.from_dict(columns), env


def intervention_env(
    demand: pl.DataFrame, weather: pl.DataFrame, *, kind: str, factor: float
) -> InterventionEnvironment:
    """One intervention per cell (effect on mean load) → the interventional verifiable-reward env."""
    cells = demand["cell"].unique().to_list()
    return InterventionEnvironment(
        demand, weather, [Intervention(kind, c, factor, "mean") for c in cells]
    )


def build_intervention_dataset(
    city: str, date: str, days: int, res: int, *, kind: str, factor: float
):
    """Per-cell interventions → (hf_dataset, env). Rows are {prompt, true_delta, scale}; the hidden
    counterfactual Δ is the verifiable target the reward scores against."""
    from datasets import Dataset

    wx = _weather(city, date, days, res)
    env = intervention_env(_synth_load(wx, res), wx, kind=kind, factor=factor)
    return Dataset.from_list(training_records(env)), env


def main(
    city: Annotated[str, typer.Option(help="preset region")] = "london",
    date: Annotated[str, typer.Option(help="YYYY-MM-DD start")] = "2020-01-15",
    days: Annotated[int, typer.Option(help="days of history")] = 7,
    res: Annotated[int, typer.Option(help="H3 resolution")] = 8,
    task: Annotated[str, typer.Option(help="forecast or intervention")] = "forecast",
    kind: Annotated[str, typer.Option(help="intervention lever: retrofit or tariff")] = "retrofit",
    factor: Annotated[float, typer.Option(help="intervention intensity 0..1")] = 0.3,
    model: Annotated[
        str, typer.Option(help="base model (Unsloth)")
    ] = "unsloth/Qwen2.5-1.5B-Instruct",
    algo: Annotated[str, typer.Option(help="gspo (sequence-level) or grpo (token-level)")] = "gspo",
    steps: Annotated[int, typer.Option(help="training steps")] = 200,
    lora_rank: Annotated[int, typer.Option(help="LoRA rank")] = 16,
    num_generations: Annotated[int, typer.Option(help="rollouts per prompt")] = 8,
) -> None:
    """RLVR-train the urban reasoner against a verifiable environment (GSPO/GRPO)."""
    if city not in PRESETS:
        raise typer.BadParameter(f"--city must be one of {', '.join(sorted(PRESETS))}")
    if algo not in ("gspo", "grpo"):
        raise typer.BadParameter("--algo must be gspo or grpo")
    if task not in ("forecast", "intervention"):
        raise typer.BadParameter("--task must be forecast or intervention")

    from trl import GRPOConfig, GRPOTrainer  # CUDA box: pip install unsloth trl
    from unsloth import FastLanguageModel

    if task == "intervention":
        dataset, env = build_intervention_dataset(city, date, days, res, kind=kind, factor=factor)
        reward_funcs = [iv_reward_fn()]
        baselines = {  # the no-effect floor the LLM must beat; the oracle is the ceiling
            "no-effect floor": env.rollout(zero_effect)["mean_reward"],
            "oracle ceiling": env.rollout(oracle_effect)["mean_reward"],
        }
    else:
        dataset, env = build_dataset(city, date, days, res)
        reward_funcs = [make_reward_fn(env)]
        baselines = {
            "forecast (interval-centre)": env.rollout(lambda q: (q.lo + q.hi) / 2)["mean_reward"]
        }

    print(f"task: {task} | dataset: {len(dataset)} rows | algo: {algo}")
    for nm, val in baselines.items():
        print(f"  baseline {nm}: {val:.3f}")

    fast_model, tokenizer = FastLanguageModel.from_pretrained(
        model, max_seq_length=2048, load_in_4bit=True, fast_inference=True
    )
    fast_model = FastLanguageModel.get_peft_model(fast_model, r=lora_rank, lora_alpha=lora_rank * 2)

    config = GRPOConfig(
        output_dir=f"outputs/reasoner-{task}-{city}-{algo}",
        importance_sampling_level="sequence" if algo == "gspo" else "token",  # GSPO vs GRPO
        num_generations=num_generations,
        max_steps=steps,
        learning_rate=1e-5,
        max_completion_length=512,
        logging_steps=5,
        save_steps=max(steps // 4, 1),
    )
    trainer = GRPOTrainer(
        model=fast_model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,  # the verifiable reward — same for GSPO/GRPO/DGPO
        args=config,
        train_dataset=dataset,
    )
    trainer.train()


if __name__ == "__main__":
    typer.run(main)
