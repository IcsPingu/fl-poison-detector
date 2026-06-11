#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_PY="python"
JUPYTER="jupyter"
SYSTEM_DIR="$ROOT/PFLlibMonza/system"
DATASET_DIR="$ROOT/PFLlibMonza/dataset"
DATASET_NAME="${DATASET_NAME:-Cifar10}"
MODEL="${MODEL:-CNN}"
GLOBAL_ROUNDS="${GLOBAL_ROUNDS:-50}"
NUM_CLIENTS="${NUM_CLIENTS:-30}"
NUM_MALICIOUS="${NUM_MALICIOUS:-9}"
JOIN_RATIO="${JOIN_RATIO:-1}"
DEVICE_ID="${DEVICE_ID:-0}"
LOCAL_STEPS="${LOCAL_STEPS:-1}"
TIMES="${TIMES:-1}"
RATE_FAKE="${RATE_FAKE:-1}"
ROUND_INIT_ATK="${ROUND_INIT_ATK:-5}"
DUMP_START_ROUND="${DUMP_START_ROUND:-$((ROUND_INIT_ATK + 1))}"

STATE_DICTS_DIR="${STATE_DICTS_DIR:-$ROOT/state_dicts_monza_cnn_cifar10}"
BERT_DIR="${BERT_DIR:-$ROOT/detector_monza_cnn_cifar10}"
MLP_DIR="${MLP_DIR:-$ROOT/detector_mlp_monza_cnn_cifar10}"
RUN_DIR="${RUN_DIR:-$ROOT/detector_runs/monza_cnn_cifar10}"
ANALYSIS_OUT="${ANALYSIS_OUT:-$ROOT/analysis_outputs}"
PUBLIC_VAL_DIR="${PUBLIC_VAL_DIR:-$DATASET_DIR/$DATASET_NAME/public_val}"
RUN_LOG="${RUN_LOG:-$ROOT/rerun_full_$(date +%Y%m%d_%H%M%S).log}"

export PUBLIC_VAL_DIR

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_monza() {
  local cc="$1"
  shift
  log "Run cc=${cc}"
  cd "$SYSTEM_DIR"
  "$VENV_PY" -u main.py \
    -m "$MODEL" \
    -data "$DATASET_NAME" \
    -lr 0.001 \
    -nmc "$NUM_MALICIOUS" \
    -nc "$NUM_CLIENTS" \
    -jr "$JOIN_RATIO" \
    -atk all \
    -ria "$ROUND_INIT_ATK" \
    -cc "$cc" \
    -gr "$GLOBAL_ROUNDS" \
    -t "$TIMES" \
    -ls "$LOCAL_STEPS" \
    -did "$DEVICE_ID" \
    -rfake "$RATE_FAKE" \
    "$@"
  cd "$ROOT"
}

main() {
  cd "$ROOT"
  log "START"
  log "ROOT=$ROOT"
  log "GIT=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  log "LOG=$RUN_LOG"

  log "Clean generated artifacts"
  rm -rf \
    "$STATE_DICTS_DIR" \
    "$BERT_DIR" \
    "$MLP_DIR" \
    "$RUN_DIR" \
    "$ANALYSIS_OUT" \
    "$ROOT/notebook_monza_analysis.executed.ipynb" \
    "$ROOT"/plot_*.png
  rm -f \
    "$SYSTEM_DIR"/f.csv \
    "$SYSTEM_DIR"/fpr_frr_results_*.csv \
    "$SYSTEM_DIR"/cc_detail_results_*.csv \
    "$SYSTEM_DIR"/cc_type_results_*.csv

  log "Validate environment"
  "$VENV_PY" - <<'PY'
import torch, pyarrow as pa
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("pyarrow", pa.__version__, "PyExtensionType", hasattr(pa, "PyExtensionType"))
PY
log "Generate ${DATASET_NAME} partition"
  rm -rf "$DATASET_DIR/$DATASET_NAME"
  cd "$DATASET_DIR"
  "$VENV_PY" generate_cifar10.py noniid - dir --num-clients "$NUM_CLIENTS"
  cd "$ROOT"
  "$VENV_PY" scripts/create_label_flip_train_mal.py \
    --dataset-dir "$DATASET_DIR/$DATASET_NAME" \
    --num-classes 10

  printf 'train=%s train_mal=%s\n' \
    "$(ls -1 "$DATASET_DIR/$DATASET_NAME/train"/*.npz 2>/dev/null | wc -l)" \
    "$(ls -1 "$DATASET_DIR/$DATASET_NAME/train_mal"/*.npz 2>/dev/null | wc -l)"
    
  printf 'public_val=%s test=%s\n' \
    "$(ls -1 "$DATASET_DIR/$DATASET_NAME/public_val"/*.npz 2>/dev/null | wc -l)" \
    "$(ls -1 "$DATASET_DIR/$DATASET_NAME/test"/*.npz 2>/dev/null | wc -l)"

  log "Dump MONZA state_dicts"
  run_monza 5 --dump_state_dicts "$STATE_DICTS_DIR" --dump_start_round "$DUMP_START_ROUND"
  ls -1 "$STATE_DICTS_DIR"/*.json 2>/dev/null | wc -l
  du -sh "$STATE_DICTS_DIR"
    
  printf 'train=%s train_mal=%s\n' \
    "$(ls -1 "$DATASET_DIR/$DATASET_NAME/train"/*.npz 2>/dev/null | wc -l)" \
    "$(ls -1 "$DATASET_DIR/$DATASET_NAME/train_mal"/*.npz 2>/dev/null | wc -l)"
    
  printf 'public_val=%s test=%s\n' \
    "$(ls -1 "$DATASET_DIR/$DATASET_NAME/public_val"/*.npz 2>/dev/null | wc -l)" \
    "$(ls -1 "$DATASET_DIR/$DATASET_NAME/test"/*.npz 2>/dev/null | wc -l)"

  log "Dump MONZA state_dicts"
  run_monza 5 --dump_state_dicts "$STATE_DICTS_DIR" --dump_start_round "$DUMP_START_ROUND"
  ls -1 "$STATE_DICTS_DIR"/*.json 2>/dev/null | wc -l
  du -sh "$STATE_DICTS_DIR"


  log "Train DistilBERT detector"
  STATE_DICTS_DIR="$STATE_DICTS_DIR" \
  PUBLIC_VAL_DIR="$PUBLIC_VAL_DIR" \
  FINAL_MODEL_DIR="$BERT_DIR" \
  RUN_DIR="$RUN_DIR" \
    "$VENV_PY" -u src/detector.py

  log "Train MLP detector"
  STATE_DICTS_DIR="$STATE_DICTS_DIR" \
  PUBLIC_VAL_DIR="$PUBLIC_VAL_DIR" \
  ARTIFACTS_DIR="$MLP_DIR" \
    "$VENV_PY" -u src/detector_mlp.py

  log "Run baseline CCs"
  run_monza 5
  run_monza 3

  log "Run detector CCs"
  run_monza 6 --detector_dir "$BERT_DIR" --bert_threshold_key threshold_label_fpr05
  run_monza 7 --detector_dir "$MLP_DIR" --mlp_threshold_key threshold_label_fpr05

  log "Execute notebook plots"
  "$JUPYTER" nbconvert \
    --to notebook \
    --execute notebook_monza_analysis.ipynb \
    --output notebook_monza_analysis.executed.ipynb

  log "Write CLI summaries"
  "$VENV_PY" scripts/plot_cc_attack_types.py \
    --system-dir PFLlibMonza/system \
    --out-dir "$ANALYSIS_OUT" \
    --tail-rounds 30
  log "DONE"
}

if [[ "${1:-}" == "--background" ]]; then
  mkdir -p "$(dirname "$RUN_LOG")"
  nohup env \
    ROOT="$ROOT" \
    VENV_PY="$VENV_PY" \
    JUPYTER="$JUPYTER" \
    DATASET_NAME="$DATASET_NAME" \
    MODEL="$MODEL" \
    GLOBAL_ROUNDS="$GLOBAL_ROUNDS" \
    NUM_CLIENTS="$NUM_CLIENTS" \
    NUM_MALICIOUS="$NUM_MALICIOUS" \
    JOIN_RATIO="$JOIN_RATIO" \
    DEVICE_ID="$DEVICE_ID" \
    LOCAL_STEPS="$LOCAL_STEPS" \
    TIMES="$TIMES" \
    RATE_FAKE="$RATE_FAKE" \
    ROUND_INIT_ATK="$ROUND_INIT_ATK" \
    DUMP_START_ROUND="$DUMP_START_ROUND" \
    STATE_DICTS_DIR="$STATE_DICTS_DIR" \
    BERT_DIR="$BERT_DIR" \
    MLP_DIR="$MLP_DIR" \
    RUN_DIR="$RUN_DIR" \
    ANALYSIS_OUT="$ANALYSIS_OUT" \
    PUBLIC_VAL_DIR="$PUBLIC_VAL_DIR" \
    RUN_LOG="$RUN_LOG" \
    "$0" >"$RUN_LOG" 2>&1 &
  printf 'Started PID %s\nLog: %s\n' "$!" "$RUN_LOG"
else
  main 2>&1 | tee "$RUN_LOG"
fi
