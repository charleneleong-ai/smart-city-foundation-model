#!/usr/bin/env bash
# Launch the interventional-reasoner RLVR training as a detached daemon on a CUDA box.
# Encodes box-runbook.md §0 (deps) + §4 (train). Survives SSH / Claude-session death (PPID=1).
#
# Usage (on a rented A100-class box, from the repo root):
#   scripts/launch_reasoner_gpu.sh                 # intervention/retrofit, GSPO, Qwen2.5-1.5B, 200 steps
#   TASK=forecast scripts/launch_reasoner_gpu.sh   # the forecast-task run
#   STEPS=400 MODEL=unsloth/Qwen2.5-3B-Instruct scripts/launch_reasoner_gpu.sh
#
# Then: adapters land in outputs/reasoner-<task>-<city>-<algo>/; eval with
#   uv run --extra forecast python apps/eval_reasoner.py --city "$CITY" --task intervention \
#       --oracle real --lcl-source data/lcl.csv --need-source data/need.csv --model <adapter-dir>
set -euo pipefail

# --- knobs (runbook defaults) ---------------------------------------------------------------
CITY="${CITY:-london}"
TASK="${TASK:-intervention}"          # forecast | intervention
KIND="${KIND:-retrofit}"              # retrofit | tariff  (intervention only)
FACTOR="${FACTOR:-0.3}"               # intervention intensity 0..1
ALGO="${ALGO:-gspo}"                  # gspo (sequence-level) | grpo (token-level)
MODEL="${MODEL:-unsloth/Qwen2.5-1.5B-Instruct}"
STEPS="${STEPS:-200}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"         # set 1 to skip the pre-flight gate

# --- pre-flight -----------------------------------------------------------------------------
command -v nvidia-smi >/dev/null 2>&1 || { echo "ERROR: no nvidia-smi — the RL stack is CUDA-only"; exit 1; }
python -c "import unsloth, trl, vllm" 2>/dev/null || {
    echo ">> installing the CUDA RL stack (unsloth trl vllm) — not in the repo extras ..."
    uv pip install unsloth trl vllm
}
if [ "$SKIP_SMOKE" != "1" ]; then
    echo ">> smoke gate: pytest + mypy (abort on failure) ..."
    uv run pytest -q && uv run mypy src
fi

# --- launch (detached, init-owned) ----------------------------------------------------------
mkdir -p logs
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="logs/train_${TASK}_${STAMP}.log"
ARGS=(--city "$CITY" --task "$TASK" --algo "$ALGO" --model "$MODEL" --steps "$STEPS")
[ "$TASK" = "intervention" ] && ARGS+=(--kind "$KIND" --factor "$FACTOR")

echo ">> launching: train_reasoner ${ARGS[*]}"
setsid nohup uv run --extra forecast python apps/train_reasoner.py "${ARGS[@]}" \
    </dev/null >>"$LOG" 2>&1 & disown
sleep 3

# --- verify it detached (column 3 == PPID must be 1) -----------------------------------------
echo ">> log: $LOG"
if ps -ef | grep "[t]rain_reasoner" | awk '$3 == 1 {found=1} END {exit !found}'; then
    ps -ef | grep "[t]rain_reasoner"
    echo ">> OK — init-owned (PPID=1). Watch: tail -f $LOG"
else
    echo "WARNING: no PPID=1 train_reasoner process — it may have exited early; check $LOG"
    tail -20 "$LOG" || true
    exit 1
fi
