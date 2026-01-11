#!/bin/bash

################################################################################
# ETTm1 Case Study Experiments
#
# This script runs models on the ETTm1 dataset by calling individual model scripts.
# Use inline comments below to select which experiments to run.
#
# Features:
#   - Select experiments by commenting/uncommenting lines in CONFIGURATION section
#   - Different models run in parallel for efficiency
#   - Same model's pred_len experiments run sequentially (in model script)
#   - Collects all results to output/case_study/ETTm1.txt
#   - Renames and copies pred.npz files to output/case_study/record/
#
# Usage:
#   ./scripts/case_study/run_case_study.sh
#
# To select experiments:
#   - Add # at beginning of line to skip an experiment (comment out)
#   - Remove # to enable an experiment (uncomment)
#   - Multiple uncommented experiments will run in parallel
#
# Example configuration:
#   EXPERIMENTS+=("MODE.sh MODE")              # Run MODE model
#   # EXPERIMENTS+=("S-Mamba.sh S_Mamba")       # Skip S_Mamba model
#
################################################################################

set -e

# Create directories
mkdir -p logs
mkdir -p temp/results
mkdir -p output/case_study/record

# =============================================================================
# CONFIGURATION: Select experiments to run (comment/uncomment lines)
# =============================================================================

# Read the configuration section and parse uncommented lines
EXPERIMENTS=()
EXPERIMENTS+=("MODE.sh MODE")
EXPERIMENTS+=("S-Mamba.sh S_Mamba")
EXPERIMENTS+=("iTransformer.sh iTransformer")
EXPERIMENTS+=("PatchTST.sh PatchTST")

# =============================================================================
# END OF CONFIGURATION
# =============================================================================

# Set case study mode environment variable for output directory
export CASE_STUDY_MODE=1

# Parse experiments into arrays
EXPERIMENT_SCRIPTS=()
EXPERIMENT_MODELS=()
for exp in "${EXPERIMENTS[@]}"; do
    EXPERIMENT_SCRIPTS+=("${exp%% *}")  # Get script name (before space)
    EXPERIMENT_MODELS+=("${exp##* }")   # Get model name (after space)
done

# Check if any experiments are selected
if [ ${#EXPERIMENT_SCRIPTS[@]} -eq 0 ]; then
    echo "❌ No experiments selected!"
    echo "Please uncomment at least one experiment in the CONFIGURATION section."
    exit 1
fi

# Verify all scripts exist
for i in "${!EXPERIMENT_SCRIPTS[@]}"; do
    SCRIPT="${EXPERIMENT_SCRIPTS[$i]}"
    MODEL="${EXPERIMENT_MODELS[$i]}"
    SCRIPT_PATH="scripts/case_study/$SCRIPT"

    if [ ! -f "$SCRIPT_PATH" ]; then
        echo "❌ Script not found: $SCRIPT_PATH"
        echo "Please check the script name for $MODEL."
        exit 1
    fi
done

# Clear and initialize output file
OUTPUT_FILE="output/case_study/ETTm1.txt"
> "$OUTPUT_FILE"
echo "ETTm1 Dataset Experiments" >> "$OUTPUT_FILE"
echo "=========================" >> "$OUTPUT_FILE"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUTPUT_FILE"
echo "Models to run: ${EXPERIMENT_MODELS[*]}" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Display configuration
echo "================================================================================"
echo "ETTm1 Case Study Experiments"
echo "================================================================================"
echo "Selected models (${#EXPERIMENT_MODELS[@]}): ${EXPERIMENT_MODELS[*]}"
echo "================================================================================"
echo ""
echo "Starting experiments in parallel..."
echo ""

# Run selected experiments in parallel
PIDS=()

for i in "${!EXPERIMENT_SCRIPTS[@]}"; do
    SCRIPT="${EXPERIMENT_SCRIPTS[$i]}"
    MODEL="${EXPERIMENT_MODELS[$i]}"

    # Record start time
    START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
    echo "Model: $MODEL | Started: $START_TIME" >> "$OUTPUT_FILE"

    # Run in background for parallel execution
    echo "▶ Starting $MODEL (running in background)..."
    LOG_FILE="logs/train_${MODEL}_ETTm1.log"

    (
        bash "scripts/case_study/$SCRIPT" > "$LOG_FILE" 2>&1
        EXIT_CODE=$?

        if [ $EXIT_CODE -eq 0 ]; then
            # Process prediction files for each prediction length
            for pred_len in 96 192 384; do
                RESULT_DIRS=(./temp/results/*${MODEL}*ETTm1_*${pred_len}*)
                for RESULT_DIR in "${RESULT_DIRS[@]}"; do
                    if [ -d "$RESULT_DIR" ] && [ -f "${RESULT_DIR}/pred.npy" ]; then
                        NEW_NAME="ETTm1_${MODEL}_${pred_len}_pred.npy"
                        cp "${RESULT_DIR}/pred.npy" "output/case_study/record/${NEW_NAME}"
                    fi
                done
            done
        fi

        exit $EXIT_CODE
    ) &

    # Store PID
    PIDS+=($!)
done

# Wait for all background processes
echo ""
echo "Waiting for all experiments to complete..."
echo "================================================================================"
echo ""

FAILED_MODELS=()
SUCCESSFUL_MODELS=()

for i in "${!PIDS[@]}"; do
    MODEL="${EXPERIMENT_MODELS[$i]}"
    PID="${PIDS[$i]}"

    wait $PID
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ $MODEL completed successfully"
        SUCCESSFUL_MODELS+=("$MODEL")
        echo "Status: SUCCESS | Completed: $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUTPUT_FILE"
    else
        echo "✗ $MODEL failed with exit code $EXIT_CODE"
        FAILED_MODELS+=("$MODEL")
        echo "Status: FAILED | Exit Code: $EXIT_CODE | Log: logs/train_${MODEL}_ETTm1.log" >> "$OUTPUT_FILE"
    fi

echo "" >> "$OUTPUT_FILE"
done

# Convert results to NPZ format
echo ""
echo "Converting results to NPZ format..."
echo "================================================================================"

if [ -f "scripts/convert_npy_to_npz.py" ]; then
    python scripts/convert_npy_to_npz.py --results_dir ./temp/results --verify
    echo "✓ Results conversion completed"
else
    echo "⚠ Converter script not found, skipping conversion"
fi

# Final summary
echo ""
echo "================================================================================"
echo "Experiment Complete!"
echo "================================================================================"
echo ""
echo "Results Summary:"
echo "  📊 Results: $OUTPUT_FILE"
echo "  📁 Record files: output/case_study/record/"
echo "  📝 Logs: logs/train_*.log"
echo ""
echo "Execution Summary:"
echo "  Successful models (${#SUCCESSFUL_MODELS[@]}): ${SUCCESSFUL_MODELS[*]}"
echo "  Failed models (${#FAILED_MODELS[@]}): ${FAILED_MODELS[*]}"
echo ""
echo "Generated prediction files:"
ls -lh output/case_study/record/*.npy 2>/dev/null || echo "No prediction files found"
echo ""
echo "================================================================================"

# Clear environment variable
unset CASE_STUDY_MODE
