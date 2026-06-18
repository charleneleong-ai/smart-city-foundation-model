"""Establish the pre-RLVR baseline for the reasoning foundation model.

Scores reference policies on a verifiable environment, and — with --model — the base LLM
*zero-shot*. That zero-shot number is the bar RLVR has to beat. Run before apps/train_reasoner.py.

Two tasks (--task), matching train_reasoner:
- forecast: oracle ceiling, the GBM forecast (interval centre), a constant-mean floor.
- intervention: oracle ceiling vs the no-effect floor — the references the interventional
  reasoner must beat.

Run (reference baselines, CPU):
    uv run --extra forecast python apps/eval_reasoner.py --city london --task intervention
Run (also base-model zero-shot, GPU box):
    uv run --extra forecast python apps/eval_reasoner.py --city london --model unsloth/Qwen2.5-1.5B-Instruct
"""

import os
from functools import partial
from typing import Annotated

import typer

from sctwin.reason.baseline import evaluate, reference_policies
from sctwin.reason.grpo import make_reward_fn
from sctwin.reason.intervention_policy import make_reward_fn as iv_reward_fn
from sctwin.reason.intervention_policy import oracle_effect, zero_effect

from presets import PRESETS
from train_reasoner import build_intervention_samples, build_samples


def _generate(model: str, prompts: list[str], max_new_tokens: int) -> list[str]:
    """Base model, no LoRA, no training: one greedy answer per prompt — the foundation model's
    output before any RLVR. GPU box only (vLLM-backed Unsloth)."""
    from unsloth import FastLanguageModel
    from vllm import SamplingParams

    llm, _ = FastLanguageModel.from_pretrained(
        model, max_seq_length=2048, load_in_4bit=True, fast_inference=True
    )
    outs = llm.fast_generate(
        prompts, sampling_params=SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    )
    return [o.outputs[0].text for o in outs]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def main(
    city: Annotated[str, typer.Option(help="preset region")] = "london",
    date: Annotated[str, typer.Option(help="YYYY-MM-DD start")] = "2020-01-15",
    days: Annotated[int, typer.Option(help="days of history")] = 7,
    res: Annotated[int, typer.Option(help="H3 resolution")] = 8,
    task: Annotated[str, typer.Option(help="forecast or intervention")] = "forecast",
    kind: Annotated[str, typer.Option(help="intervention lever: retrofit or tariff")] = "retrofit",
    factor: Annotated[float, typer.Option(help="intervention intensity 0..1")] = 0.3,
    model: Annotated[
        str | None, typer.Option(help="also score this base model zero-shot (GPU box)")
    ] = None,
    source: Annotated[
        str, typer.Option(help="open-meteo, or era5 (gridded; needs CDS key)")
    ] = "open-meteo",
    max_new_tokens: Annotated[
        int, typer.Option(help="generation budget for the zero-shot pass")
    ] = 512,
) -> None:
    """Print the baseline reward table for the reasoning environment (run before training)."""
    if city not in PRESETS:
        raise typer.BadParameter(f"--city must be one of {', '.join(sorted(PRESETS))}")
    if task not in ("forecast", "intervention"):
        raise typer.BadParameter("--task must be forecast or intervention")
    if source == "era5":
        os.environ["WEATHER_SOURCE"] = "era5"

    if task == "intervention":
        cols, env = build_intervention_samples(city, date, days, res, kind=kind, factor=factor)
        scores = {
            "oracle (perfect)": env.rollout(oracle_effect)["mean_reward"],
            "no-effect (floor)": env.rollout(zero_effect)["mean_reward"],
        }
        reward = partial(iv_reward_fn(), true_delta=cols["true_delta"], scale=cols["scale"])
        unit = f"{kind} interventions"
    else:
        cols, env = build_samples(city, date, days, res)
        scores = evaluate(env, reference_policies(env))
        reward = partial(make_reward_fn(env), actual=cols["actual"], lo=cols["lo"], hi=cols["hi"])
        unit = "cell-hours"

    if model:  # the base model's pre-RLVR reward — the bar training has to beat
        scores[f"base zero-shot ({model})"] = _mean(
            reward(_generate(model, cols["prompt"], max_new_tokens))
        )

    print(f"\nbaseline mean reward — {len(env.questions())} held-out {unit} ({city} {date})")
    for name, score in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"  {name:34s} {score:.3f}")


if __name__ == "__main__":
    typer.run(main)
