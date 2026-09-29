#!/usr/bin/env bash
# Train both models (resumable), then evaluate and export the browser models.
# Safe to re-run: finished steps are skipped, interrupted training resumes.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
[ -f models/spm_en.model ] || python3 src/preprocess.py
COMMON="--emb 256 --hid 256 --layers 2 --batch_size 128 --epochs 5 --time_budget 150 --threads 2"
pids=()
for m in seq2seq attention; do
  if [ ! -f "results/train_$m.json" ]; then
    python3 src/train.py --model "$m" $COMMON >> "logs/train_$m.log" 2>&1 &
    pids+=($!)
  fi
done
for p in "${pids[@]}"; do wait "$p"; done
python3 src/evaluate.py > logs/evaluate.log 2>&1
python3 src/export_onnx.py > logs/export.log 2>&1
python3 tests/test_web_parity.py > logs/parity.log 2>&1
echo PIPELINE_DONE >> logs/pipeline.log
