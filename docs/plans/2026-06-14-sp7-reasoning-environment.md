# SP7 (PR 1) — Verifiable-Reasoning Environment + Rewards — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or executing-plans. Checkbox steps.

**Goal:** Make the spec's thesis concrete and testable — *the city as a verifiable-reward
(RLVR) environment*. Build the **reward functions** and the **environment** that a reasoning
policy (later, an RLVR-trained LLM) is scored against. No GPU / LLM here: the policy is a
callable `Question -> float`; this PR is the substrate an LLM/GRPO loop plugs into.

**Why this slice:** Per the spec's Novelty section, the moat is the *verifiable-reasoning
layer*, and SP5 is "also the verifiable-reasoning environment." This PR is the SP5→SP7
bridge: turn SP5 verification results into a scored reasoning environment, with reward
components for the three strongest mechanisms (1 = checkable-answer reward, 3 = causal/
interventional reward, 4 = conservation/process reward). RLVR-World (2025) already does RLVR
on world models, so the contribution is the *physics/meter-verified urban* reward — encoded
here as scoring against held-out actuals + calibrated intervals + conservation constraints.

**Tech:** `math` (stdlib) + `polars`. Reuses SP5's `verification_frame` output.

**Depends on:** SP5. Branch `feat/sp7-reasoning` off `main`.

---

## File Structure

- `src/sctwin/reason/__init__.py`
- `src/sctwin/reason/reward.py` — `accuracy_reward`, `coverage_reward`, `interventional_reward`, `conservation_reward`
- `src/sctwin/reason/environment.py` — `Question`, `ReasoningEnvironment` (questions / reward / rollout)
- `tests/test_reward.py`, `tests/test_environment.py`

---

### Task 1: reward components (`reward.py`)

Four verifiable rewards, all returning [0, 1] (1 = perfect):
- **accuracy** — `exp(-|pred-actual|/scale)`: agreement with the held-out actual (mechanism 1).
- **coverage** — 1 iff `lo <= pred <= hi`: inside the calibrated interval.
- **interventional** — half direction (sign of the effect) + half magnitude vs the real
  before/after delta (mechanism 3, causal).
- **conservation** — 1 iff `|sum(parts)-whole| <= tol*|whole|`, else decays (mechanism 4).

- [ ] Write failing `tests/test_reward.py` (accuracy decays with error & is 1 at exact;
      coverage in/out; interventional rewards right direction+magnitude, penalises wrong
      direction; conservation 1 when balanced, <1 when not) → implement → pass → commit.

### Task 2: environment (`environment.py`)

`ReasoningEnvironment(results)` from an SP5 frame (`cell, time, y_true, lo, hi`):
- `questions() -> list[Question]` — each carries the *hidden* actual + interval.
- `reward(q, answer)` — `0.7*accuracy + 0.3*coverage` (verifiable per-step reward).
- `rollout(policy)` — run a `Question->float` policy over all questions → `{mean_reward, n}`.
  This is the RLVR return an LLM policy is optimised for.

- [ ] Write failing `tests/test_environment.py` (a good policy `lambda q: q.actual` scores
      ~1; a constant/bad policy scores lower; reward bounded [0,1]; within-interval answer
      gets the coverage bonus) → implement → pass → commit.

### Task 3: exports + gate

- [ ] `__init__.py` exports the rewards + `Question` + `ReasoningEnvironment`.
- [ ] `uv run pytest -q && uv run ruff check src tests && uv run mypy src` clean. Commit.

---

## Self-Review notes

- **Scope honesty:** this is the *environment*, not the trained reasoner. The LLM + GRPO
  loop (generate a reasoning chain → emit a number → reward) is the GPU follow-up; it plugs
  into `rollout(policy)`.
- **Discrimination test is the point:** the environment is only useful if better answers get
  higher reward — `test_environment` asserts good > bad, validating the RLVR signal.
- **Simulator oracle** (EnergyPlus) and **real natural-experiment deltas** are the eventual
  tier-1/2 oracles feeding `interventional_reward`; here they're the function's inputs.
