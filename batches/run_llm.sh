#!/bin/bash

# ===============================
# 🔹 CONFIGURATION
# ===============================
FEATURES=(
'GPT_5_4_is_not_generalizable_fewshot'
'GPT_5_4_is_context_specific_fewshot'
'GPT_5_4_detect_all_at_once_fewshot'
'GPT_5_4_is_numeric_fewshot'
'GPT_5_4_is_categorical_fewshot'
'GPT_5_4_is_datetime_fewshot'
'GPT_5_4_is_sentence_fewshot'
'GPT_5_4_is_url_fewshot'
'GPT_5_4_is_embedded_number_fewshot'
'GPT_5_4_is_list_fewshot'
'GPT_5_4_is_not_generalizable'
'GPT_5_4_is_context_specific'
'GPT_5_4_detect_all_at_once'
'GPT_5_4_is_numeric'
'GPT_5_4_is_categorical'
'GPT_5_4_is_datetime'
'GPT_5_4_is_sentence'
'GPT_5_4_is_url'
'GPT_5_4_is_embedded_number'
'GPT_5_4_is_list'
)

MAX_PARALLEL=3
LOG_DIR="../logs"
BATCH_ID_FILE="batches/batch_ids.txt"

echo "========================================="
echo "Parallel LLM feature processing"
echo "Max parallel jobs: $MAX_PARALLEL"
echo "Batch-IDs are stored in file: ${BATCH_ID_FILE}"
echo "========================================="

# ===============================
# 🔹 PARALLEL EXECUTION
# ===============================
PIDS=()

for FEATURE in "${FEATURES[@]}"
do
  (
    echo "▶ Starting: $FEATURE"

    echo "> To check the current status of the running batch, look into the file: $LOG_DIR/${FEATURE}.log"

    python3 llm_feature_runner.py "$FEATURE" \
      > "$LOG_DIR/${FEATURE}.log" 2>&1

    if [ $? -ne 0 ]; then
      echo "Failed: $FEATURE"
    else
      echo "Completed: $FEATURE"
    fi

  ) &

  PIDS+=($!)

  # throttle parallel jobs
  if [ ${#PIDS[@]} -ge $MAX_PARALLEL ]; then
    wait -n
    PIDS=($(jobs -rp))
  fi
done

wait

echo "========================================="
echo "All features finished"
echo "========================================="
