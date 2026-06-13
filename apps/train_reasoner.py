"""SP7 — train the urban reasoning model with RLVR (GSPO/GRPO) against the verifiable
environment, using Unsloth (fast LoRA) + TRL's GRPOTrainer.

The policy LLM is prompted to reason about a cell's load and emit \\boxed{number}; the
reward (sctwin.reason.grpo.make_reward_fn) parses it and scores it against the held-out
actual + calibrated interval — the verifiable reward. GSPO (sequence-level importance) is
the default; --algo grpo for token-level.

Needs a CUDA GPU with `unsloth` + `trl` installed (not in the repo extras — CUDA-specific,
install on the box). The reward glue (sctwin.reason) is pure + tested without a GPU.

Run (on the GPU box):
    uv run --extra forecast python apps/train_reasoner.py --city london --algo gspo \
        --model unsloth/Qwen2.5-1.5B-Instruct --steps 200
"""

from datetime import datetime, timedelta, timezone
from typing import Annotated

import typer

from sctwin.adapters.cache import CachingAdapter
from sctwin.adapters.open_meteo import OpenMeteoWeatherAdapter
from sctwin.app.cells import cells_in_bbox
from sctwin.forecast.baselines import GBMForecaster
from sctwin.forecast.features import FEATURE_COLS, build_supervised
from sctwin.reason.environment import ReasoningEnvironment
from sctwin.reason.grpo import make_reward_fn, prompt_for
from sctwin.registry import Registry
from sctwin.verify.results import verification_frame

from presets import PRESETS
from twin import _synth_load


def build_dataset(city: str, date: str, days: int, res: int):
    """A (prompt, actual, lo, hi) row per held-out test cell-hour, built from the same
    SP4 features + SP5 verification the twin uses. Returns (hf_dataset, environment)."""
    from datasets import Dataset

    p = PRESETS[city]
    cells = cells_in_bbox(p["south"], p["west"], p["north"], p["east"], res)
    start = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
    reg = Registry()
    reg.register(CachingAdapter(OpenMeteoWeatherAdapter(), ".cache/open-meteo"))
    wx = reg.get("weather.t2m", cells, start, start + timedelta(days=days - 1))

    supervised = build_supervised(_synth_load(wx, res), wx)
    results = verification_frame(GBMForecaster(), supervised, FEATURE_COLS)
    rows = supervised.join(results.select("cell", "time", "y_true", "lo", "hi"), on=["cell", "time"]).to_dicts()

    data = {
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
    return Dataset.from_dict(data), ReasoningEnvironment(results)


def main(
    city: Annotated[str, typer.Option(help="preset region")] = "london",
    date: Annotated[str, typer.Option(help="YYYY-MM-DD start")] = "2020-01-15",
    days: Annotated[int, typer.Option(help="days of history")] = 7,
    res: Annotated[int, typer.Option(help="H3 resolution")] = 8,
    model: Annotated[str, typer.Option(help="base model (Unsloth)")] = "unsloth/Qwen2.5-1.5B-Instruct",
    algo: Annotated[str, typer.Option(help="gspo (sequence-level) or grpo (token-level)")] = "gspo",
    steps: Annotated[int, typer.Option(help="training steps")] = 200,
    lora_rank: Annotated[int, typer.Option(help="LoRA rank")] = 16,
    num_generations: Annotated[int, typer.Option(help="rollouts per prompt")] = 8,
) -> None:
    """RLVR-train the urban reasoner against the verifiable environment (GSPO/GRPO)."""
    if city not in PRESETS:
        raise typer.BadParameter(f"--city must be one of {', '.join(sorted(PRESETS))}")
    if algo not in ("gspo", "grpo"):
        raise typer.BadParameter("--algo must be gspo or grpo")

    from trl import GRPOConfig, GRPOTrainer  # CUDA box: pip install unsloth trl
    from unsloth import FastLanguageModel

    dataset, env = build_dataset(city, date, days, res)
    print(f"dataset: {len(dataset)} held-out cell-hours | algo: {algo}")

    fast_model, tokenizer = FastLanguageModel.from_pretrained(
        model, max_seq_length=2048, load_in_4bit=True, fast_inference=True
    )
    fast_model = FastLanguageModel.get_peft_model(fast_model, r=lora_rank, lora_alpha=lora_rank * 2)

    config = GRPOConfig(
        output_dir=f"outputs/reasoner-{city}-{algo}",
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
        reward_funcs=[make_reward_fn(env)],  # the verifiable reward — same for GSPO/GRPO/DGPO
        args=config,
        train_dataset=dataset,
    )
    base = env.rollout(lambda q: (q.lo + q.hi) / 2)  # forecast-centre baseline the LLM must beat
    print(f"baseline (interval-centre) mean reward: {base['mean_reward']:.3f}")
    trainer.train()


if __name__ == "__main__":
    typer.run(main)
