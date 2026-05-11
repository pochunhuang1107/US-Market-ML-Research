#!/usr/bin/env bash
set -u

mkdir -p artifacts/mlp_runs

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT="artifacts/mlp_runs/mlp_resumable_sweep_${RUN_ID}.log"
DONE="artifacts/mlp_runs/mlp_completed_keys.txt"
FAILED="artifacts/mlp_runs/mlp_failed_keys.txt"

touch "$DONE"
touch "$FAILED"

echo "MLP resumable sweep started at $(date)" | tee -a "$OUT"
echo "Log: $OUT" | tee -a "$OUT"
echo "Done checkpoint: $DONE" | tee -a "$OUT"
echo "" | tee -a "$OUT"

UNIVERSES=(
  "core:artifacts/cache/universe_core_2018_2026.parquet"
  "broad_macro:artifacts/cache/universe_broad_macro_2018_2026.parquet"
  "sector:artifacts/cache/universe_sector_2018_2026.parquet"
  "style_growth:artifacts/cache/universe_style_growth_2018_2026.parquet"
  "full:artifacts/cache/universe_full_2018_2026.parquet"
)

FOLDS=(
  "fold_a:2019-01-01:2021-12-31:2022-01-01:2022-12-31:2023-01-01:2024-12-31"
  "fold_b:2019-01-01:2022-12-31:2023-01-01:2023-12-31:2024-01-01:2024-12-31"
  "fold_c:2019-01-01:2023-12-31:2024-01-01:2024-12-31:2025-01-01:2026-05-08"
)

TOPKS=(1 2 3 5)
HIDDEN_CONFIGS=("64" "128,64" "256,128" "128,64,32")
LRS=("0.001" "0.0005")
DROPOUTS=("0.0" "0.1")
SEEDS=(42 7)

for U in "${UNIVERSES[@]}"; do
  IFS=":" read -r UNIVERSE_NAME DATA_PATH <<< "$U"

  for F in "${FOLDS[@]}"; do
    IFS=":" read -r FOLD_NAME TRAIN_START TRAIN_END VAL_START VAL_END TEST_START TEST_END <<< "$F"

    for HIDDEN in "${HIDDEN_CONFIGS[@]}"; do
      for LR in "${LRS[@]}"; do
        for DROPOUT in "${DROPOUTS[@]}"; do
          for SEED in "${SEEDS[@]}"; do
            for TOPK in "${TOPKS[@]}"; do

              KEY="UNIVERSE=${UNIVERSE_NAME}|FOLD=${FOLD_NAME}|TOPK=${TOPK}|HIDDEN=${HIDDEN}|LR=${LR}|DROPOUT=${DROPOUT}|SEED=${SEED}"

              if grep -Fxq "$KEY" "$DONE"; then
                echo "SKIP completed: $KEY" | tee -a "$OUT"
                continue
              fi

              echo "==================================================" | tee -a "$OUT"
              echo "$KEY" | tee -a "$OUT"
              echo "DATA_PATH=$DATA_PATH" | tee -a "$OUT"
              echo "TRAIN=$TRAIN_START to $TRAIN_END" | tee -a "$OUT"
              echo "VAL=$VAL_START to $VAL_END" | tee -a "$OUT"
              echo "TEST=$TEST_START to $TEST_END" | tee -a "$OUT"
              echo "==================================================" | tee -a "$OUT"

              if us-market-ml-mlp-smoke \
                --data-path "$DATA_PATH" \
                --train-start "$TRAIN_START" \
                --train-end "$TRAIN_END" \
                --validation-start "$VAL_START" \
                --validation-end "$VAL_END" \
                --test-start "$TEST_START" \
                --test-end "$TEST_END" \
                --hidden-units "$HIDDEN" \
                --learning-rate "$LR" \
                --dropout "$DROPOUT" \
                --max-epochs 80 \
                --patience 10 \
                --batch-size 128 \
                --random-state "$SEED" \
                --top-k "$TOPK" 2>&1 | tee -a "$OUT"; then

                echo "$KEY" >> "$DONE"
                echo "DONE: $KEY" | tee -a "$OUT"
              else
                echo "$KEY" >> "$FAILED"
                echo "FAILED: $KEY" | tee -a "$OUT"
              fi

              echo "" | tee -a "$OUT"

            done
          done
        done
      done
    done
  done
done

echo "MLP resumable sweep finished at $(date)" | tee -a "$OUT"
echo "Output saved to $OUT"
