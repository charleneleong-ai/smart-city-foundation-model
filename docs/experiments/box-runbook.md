# GPU-box runbook — producing the §7 numbers

Turnkey commands to fill every `[TBD]` in the paper's §7. Each step names the result table it
produces. Run from the repo root on a CUDA box. Grounded in the committed entrypoints as of
2026-06-18.

## 0. Prereqs

```bash
# repo deps (CPU-side: forecasting, verification, oracles, the FastAPI app)
uv sync --extra dev --extra forecast --extra tsfm --extra app

# RL stack is CUDA-specific and NOT in the extras — install on the box:
uv pip install unsloth trl vllm

# optional tokens (only if you use those demand sources; London+NSW below need none)
export ENTSOE_TOKEN=...        # EU demand
export EIA_API_KEY=...         # US demand
export ELECTRICITYMAPS_TOKEN=... # global live load
mkdir -p logs data outputs
```

Smoke-test the pure stack before burning GPU time:

```bash
uv run pytest -q                 # full suite must pass
uv run mypy src                  # the gate
```

## 1. Data

| Data | How | Needed for |
| --- | --- | --- |
| Weather (Open-Meteo) | auto-fetched + cached to `.cache/` on first run | all |
| London LCL load, AU AEMO load | auto-fetched by the adapters (HF parquet / aemo.com.au) | RQ1, RQ3 |
| **LCL dToU trial CSV (~760 MB, CC-BY)** | **manual** → save as `data/lcl.csv` ([London Datastore](https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households/)) | RQ2 (tariff) |
| **UK NEED anonymised sample** | **manual** → save as `data/need.csv` ([gov.uk](https://www.gov.uk/government/collections/national-energy-efficiency-data-need-framework)) | RQ2 (retrofit) |

After downloading, confirm the LCL column names (`stdorToU`, `DateTime`, `KWH/hh (per half hour) `)
and the NEED measure/consumption columns against the data dictionary — the parsers take them as
params.

## 2. RQ1 — Forecasting crossover table (`§7.2`)

```bash
uv run --extra forecast --extra tsfm python apps/eval_compare.py \
    --freqs hour,day,week,month --meters 8 --device cuda
```

Prints `region × freq` GBM-MAE vs Chronos-MAE with the winner — copy straight into the §7.2 table.
(Loads `amazon/chronos-2` once; CPU also works but is slow at hourly.)

## 3. RQ2/RQ3 — Reference baselines + base-model zero-shot (`§7.3`, `§7.4`)

CPU reference tables (oracle / forecast / floor), both tasks:

```bash
uv run --extra forecast python apps/eval_reasoner.py --city london --task forecast
uv run --extra forecast python apps/eval_reasoner.py --city london --task intervention --kind retrofit
```

Add the **base-model zero-shot** row (GPU) — the bar RLVR must clear:

```bash
uv run --extra forecast python apps/eval_reasoner.py --city london --task intervention \
    --kind retrofit --model unsloth/Qwen2.5-1.5B-Instruct
```

## 4. Train the reasoner — detached daemon (`§7.3`, `§7.5`)

Long runs must survive SSH/Claude-session death (PPID=1). Forecast task:

```bash
setsid nohup uv run --extra forecast python apps/train_reasoner.py \
    --city london --task forecast --algo gspo --model unsloth/Qwen2.5-1.5B-Instruct --steps 200 \
    </dev/null >>logs/train_forecast_$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 & disown
```

Interventional task (the headline policy + the §7.5 ablation reward):

```bash
setsid nohup uv run --extra forecast python apps/train_reasoner.py \
    --city london --task intervention --kind retrofit --factor 0.3 \
    --algo gspo --model unsloth/Qwen2.5-1.5B-Instruct --steps 200 \
    </dev/null >>logs/train_intervention_$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 & disown

# confirm it's init-owned (column 3 == PPID == 1) and watch it
ps -ef | grep train_reasoner | grep -v grep
tail -f logs/train_intervention_*.log
```

LoRA adapters land in `outputs/reasoner-intervention-london-gspo/`. For the **§7.5 ablation**, repeat
the intervention run with the forecast-only reward and an SFT control, holding data/steps fixed.

## 5. RQ2 — Real interventional validity via DiD (`§7.3`)

This is the one piece not yet wired into an app (the `--oracle real` flag is an out-of-scope
follow-up). The script below produces the **measured DiD ground-truth Δ** (no model needed) and the
floor/ceiling bracket; scoring the trained policy against it is the optional GPU block.

```python
# scripts/real_oracle.py  — run: uv run --extra forecast python scripts/real_oracle.py
from sctwin.adapters.demand import LCLTariffAdapter, NEEDRetrofitAdapter
from sctwin.reason.intervention import InterventionEnvironment, did_question
from sctwin.reason.intervention_policy import oracle_effect, zero_effect
from sctwin.geo import cell_of

cell = cell_of(51.5, -0.12, 7).h3  # a London label cell (LCL profiles aggregate all households)

# tariff: ToU vs Std across the pre-trial (2012) and trial (2013) years, diurnal typical-day peak
tp = LCLTariffAdapter("data/lcl.csv").did_profiles(cell)
q_tariff = did_question("tariff", *tp, cell=cell, metric="peak")

# retrofit: measure vs non-measure homes, pre/post annual mean (set cols to the NEED dictionary)
need = NEEDRetrofitAdapter("data/need.csv", measure_col="LOFT_FLAG", pre_col="Econ2010", post_col="Econ2013")
q_retro = did_question("retrofit", *need.did_split(cell), cell=cell, metric="mean")

env = InterventionEnvironment.from_questions([q_tariff, q_retro])
print("measured DiD Δ — tariff peak:", q_tariff.true_delta, " retrofit annual:", q_retro.true_delta)
print("no-effect floor:", env.rollout(zero_effect)["mean_reward"])
print("oracle ceiling :", env.rollout(oracle_effect)["mean_reward"])  # ~1.0 by construction

# --- optional (GPU): score the trained reasoner against the real DiD targets ---
# from unsloth import FastLanguageModel
# from vllm import SamplingParams
# from sctwin.reason.intervention_policy import llm_policy
# llm, _ = FastLanguageModel.from_pretrained("outputs/reasoner-intervention-london-gspo", fast_inference=True)
# gen = lambda p: llm.fast_generate([p], sampling_params=SamplingParams(temperature=0.0, max_tokens=512))[0].outputs[0].text
# print("RLVR reasoner reward:", env.rollout(llm_policy(gen))["mean_reward"])
```

## 6. RQ3 — Conformal coverage (`§7.4`)

Coverage falls out of any `verification_frame`; this prints it for the London energy forecast:

```python
# run: uv run --extra forecast python -c "$(cat <<'PY'
from datetime import datetime, timezone
from sctwin.adapters.cache import CachingAdapter
from sctwin.adapters.open_meteo import OpenMeteoWeatherAdapter
from sctwin.app.cells import cells_in_bbox
from sctwin.forecast.baselines import GBMForecaster
from sctwin.forecast.features import FEATURE_COLS, build_supervised
from sctwin.registry import Registry
from sctwin.verify.results import verification_frame
from sctwin.verify.drift import drift_flags
from twin import _synth_load   # apps/ on path; or use a real demand adapter
import sys; sys.path.insert(0, "apps")
cells = cells_in_bbox(51.40,-0.25,51.60,0.05,7)[:8]
s,e = datetime(2020,1,1,tzinfo=timezone.utc), datetime(2020,2,1,tzinfo=timezone.utc)
reg = Registry(); reg.register(CachingAdapter(OpenMeteoWeatherAdapter(), ".cache/open-meteo"))
wx = reg.get("weather.t2m", cells, s, e)
res = verification_frame(GBMForecaster(), build_supervised(_synth_load(wx,7), wx), FEATURE_COLS, alpha=0.1)
print("target 1-a = 0.90 | empirical coverage =", round(res['covered'].mean(),3))
print("drift-flagged buckets:", drift_flags(res, target_coverage=0.90)['drift'].sum())
PY
)"
```

## 7. Where each number lands

| Step | Output → §7 table |
| --- | --- |
| 2 | §7.2 forecasting MAE (region × freq) |
| 3 | §7.3 / §7.4 reference rows (oracle, forecast/floor, zero-shot) |
| 4 | §7.3 RLVR reasoner reward; §7.5 ablation (reward variants) |
| 5 | §7.3 real DiD Δ targets + RLVR-vs-floor on real data (headline) |
| 6 | §7.4 conformal coverage + drift rate |

## 8. Daemon hygiene

- Every long run is `setsid nohup … & disown` with a timestamped log; verify PPID=1 with
  `ps -ef | grep train_reasoner`.
- `python -u` / unbuffered logs so progress isn't lost on exit.
- Kill before cleanup: `pkill -f train_reasoner; sleep 5; ps -ef | grep train_reasoner | grep -v grep`.
- One GPU run per task is the proof-of-concept; for a real result, sweep seeds and report variance.
