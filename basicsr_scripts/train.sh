#!/bin/bash

# -----------------------------------------------------------------------------
# Batch Training Script (Continues on Failure)
#
# Function:
# 1. Sequentially executes all training tasks defined in the `TRAINING_CONFIGS` array.
# 2. Checks if each task succeeded. If a task fails, it logs the error and
#    continues to the next task.
# 3. Checks for the existence of all required files.
# 4. Reports the total number of successful and failed tasks at the end.
# -----------------------------------------------------------------------------

# --- Script Safety Settings ---
# -u: Error when using an undefined variable
# -o pipefail: If any command in a pipe fails, the whole pipe fails
# (We've removed -e to ensure the script doesn't exit on a failed task)
set -uo pipefail

# --- 1. Configuration Constants ---

# Path to the main training script
TRAIN_SCRIPT="basicsr/train.py"

# Parameters for torch.distributed.launch
NPROC_PER_NODE=1
MASTER_PORT=4321
LAUNCHER="pytorch"

# --- 2. List of Tasks to Train ---

# Add all .yml config file paths to this array
# The script will execute them in this order
declare -a TRAINING_CONFIGS=(
    "options/train/CUSTOM/NAFMamba.yml"
    "options/train/CUSTOM/NAFMamba.yml"
    "options/train/CUSTOM/NAFMamba.yml"
)

# --- 3. Pre-run Checks and Counters ---

# Check if the training script exists
if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "❌ [Error] Training script not found: $TRAIN_SCRIPT"
    exit 1
fi

TOTAL_TASKS=${#TRAINING_CONFIGS[@]}
CURRENT_TASK=0
SUCCESS_COUNT=0
FAIL_COUNT=0

echo "🚀 Starting batch training... Total tasks: $TOTAL_TASKS."
echo "=========================================================="

# --- 4. Loop and Execute Tasks ---

for config_file in "${TRAINING_CONFIGS[@]}"; do
    CURRENT_TASK=$((CURRENT_TASK + 1))
    
    echo "🏃 [Task $CURRENT_TASK / $TOTAL_TASKS] Starting..."
    echo "   Config: $config_file"
    echo "----------------------------------------------------------"

    # Check if the config file exists
    if [[ ! -f "$config_file" ]]; then
        echo "⚠️ [Skipped] Config file not found: $config_file"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "=========================================================="
        continue # Skip this task, move to the next
    fi

    # Execute the training command
    python -m torch.distributed.launch \
        --nproc_per_node="$NPROC_PER_NODE" \
        --master_port="$MASTER_PORT" \
        "$TRAIN_SCRIPT" \
        -opt "$config_file" \
        --launcher "$LAUNCHER"
    
    # *** Vital Check ***
    # Check the exit code of the last command (python)
    if [[ $? -ne 0 ]]; then
        # Task Failed
        echo "----------------------------------------------------------"
        echo "❌ [Failed] Task $CURRENT_TASK / $TOTAL_TASKS failed!"
        echo "   Failed config: $config_file"
        echo "   Continuing to the next task..."
        FAIL_COUNT=$((FAIL_COUNT + 1))
    else
        # Task Succeeded
        echo "----------------------------------------------------------"
        echo "✅ [Success] Task $CURRENT_TASK / $TOTAL_TASKS completed."
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    fi
    
    echo "=========================================================="
done

# --- 5. Final Summary ---
echo "🎉 Batch training run finished."
echo "--- Summary ---"
echo "   Successful: $SUCCESS_COUNT / $TOTAL_TASKS"
echo "   Failed:     $FAIL_COUNT / $TOTAL_TASKS"
echo "=========================================================="

# (Optional) If you want the script to report a failure status
# (e.g., for CI/CD systems) if ANY task failed,
# keep the following lines.
if [[ $FAIL_COUNT -gt 0 ]]; then
    echo "Script exiting with error status due to task failures."
    exit 1
fi
