"""Establish the pre-RLVR baseline for the reasoning foundation model.

Scores reference policies (oracle ceiling, the GBM forecast = interval centre, a constant-mean
floor) on the verifiable environment, and — with --model — the base LLM *zero-shot*. That
zero-shot number is the bar RLVR has to beat; the forecast row is the strong reference it's
trying to approach. Run this before apps/train_reasoner.py.

Run (reference baselines, CPU):
    uv run --extra forecast python apps/eval_reasoner.py --city london
Run (also base-model zero-shot, GPU box):
    uv run --extra forecast python apps/eval_reasoner.py --city london --model unsloth/Qwen2.5-1.5B-Instruct
"""

import os
from typing import Annotated

import typer

from sctwin.reason.baseline import evaluate, reference_policies
from sctwin.reason.grpo import make_reward_fn

from presets import PRESETS
from train_reasoner import build_samples


def _zero_shot_reward(model: str, cols: dict, env, max_new_tokens: int) -> float:
    """Base model, no LoRA, no training: generate one answer per prompt and score it — the
    foundation model's reward before any RLVR. GPU box only (vLLM-backed Unsloth)."""
    from unsloth import FastLanguageModel
    from vllm import SamplingParams

    llm, _ = FastLanguageModel.from_pretrained(model, max_seq_length=2048, load_in_4bit=True, fast_inference=True)
    outs = llm.fast_generate(cols["prompt"], sampling_params=SamplingParams(temperature=0.0, max_tokens=max_new_tokens))
    completions = [o.outputs[0].text for o in outs]
    rewards = make_reward_fn(env)(completions, actual=cols["actual"], lo=cols["lo"], hi=cols["hi"])
    return sum(rewards) / len(rewards)


def main(
    city: Annotated[str, typer.Option(help="preset region")] = "london",
    date: Annotated[str, typer.Option(help="YYYY-MM-DD start")] = "2020-01-15",
    days: Annotated[int, typer.Option(help="days of history")] = 7,
    res: Annotated[int, typer.Option(help="H3 resolution")] = 8,
    model: Annotated[str | None, typer.Option(help="also score this base model zero-shot (GPU box)")] = None,
    source: Annotated[str, typer.Option(help="open-meteo, or era5 (gridded; needs CDS key)")] = "open-meteo",
    max_new_tokens: Annotated[int, typer.Option(help="generation budget for the zero-shot pass")] = 512,
) -> None:
    """Print the baseline reward table for the reasoning environment (run before training)."""
    if city not in PRESETS:
        raise typer.BadParameter(f"--city must be one of {', '.join(sorted(PRESETS))}")
    if source == "era5":
        os.environ["WEATHER_SOURCE"] = "era5"

    cols, env = build_samples(city, date, days, res)
    scores = evaluate(env, reference_policies(env))
    if model:
        scores[f"base zero-shot ({model})"] = _zero_shot_reward(model, cols, env, max_new_tokens)

    print(f"\nbaseline mean reward — {len(env.questions())} held-out cell-hours ({city} {date})")
    for name, reward in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"  {name:34s} {reward:.3f}")


if __name__ == "__main__":
    typer.run(main)
