#!/bin/bash

# =============================================================================
# MODE Increasing Lookback Experiments Runner
# =============================================================================
# This script runs lookback experiments for different models across datasets.
# Tests model performance with varying input sequence lengths.
#
# Usage:
#   bash run-lookback.sh [dataset] [model]
#     dataset: Dataset name (ECL, ETTm1, Traffic, Weather, PEMS_04) or 'all' (default: all)
#     model:   Optional model name. If not specified, runs all models for the dataset
#
# Examples:
#   bash run-lookback.sh              # Run all models on all datasets
#   bash run-lookback.sh ETTm1        # Run all models on ETTm1 dataset
#   bash run-lookback.sh ETTm1 MODE   # Run only MODE model on ETTm1 dataset
#   bash run-lookback.sh all MODE     # Run MODE model on all datasets
#   bash run-lookback.sh Weather      # Run all models on Weather dataset
# =============================================================================

set -e  # Exit on error

# Change to the repository root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

echo "MODE Increasing Lookback Experiments Runner"
echo "Repository root: $REPO_ROOT"
echo ""

# Configuration
# =============================================================================
# Use line comments to select which datasets and models to include in experiments
# Simply comment out (add #) any dataset/model you want to exclude
# =============================================================================

# Available datasets (comment out to disable)
DATASETS=(
    "ECL"          # Electricity Consuming Load (321 variables)
    "ETTm1"        # Electricity Transformer Temperature (7 variables, 15 min)
    # "Traffic"      # California highway occupancy (862 variables)
    # "PEMS_04"      # Traffic flow data (307 variables)
    "Weather"      # Weather measurements (21 variables)
)

# Available models (comment out to disable)
MODELS=(
    "MODE"           # MODE: ECL, ETTm1, Weather
    "S_Mamba"        # S_Mamba: ECL, ETTm1, Weather
    "Informer_M"     # Informer_M: ECL, ETTm1, Weather
    "Transformer_M"  # Transformer_M: ECL, ETTm1, Weather
    "Reformer_M"     # Reformer_M: ECL, ETTm1, Weather
    "Informer"       # Informer: ECL, ETTm1, Weather
    "iTransformer"   # iTransformer: ECL, ETTm1, Weather
    "Reformer"       # Reformer: ECL, ETTm1, Weather
    "Transformer"    # Transformer:
)

# Validate dataset
validate_dataset() {
    local dataset=$1
    local found=0
    for d in "${DATASETS[@]}"; do
        if [[ "$d" == "$dataset" ]]; then
            found=1
            break
        fi
    done
    if [[ $found -eq 0 ]]; then
        echo -e "${RED}Error: Unknown dataset: $dataset${NC}"
        echo "Available datasets: ${DATASETS[*]}"
        exit 1
    fi
}

# Get model script path
get_model_script() {
    local dataset=$1
    local model=$2
    echo "./scripts/increasing_lookback/${dataset}/${model}.sh"
}

# Validate model
validate_model() {
    local model=$1
    local found=0
    for m in "${MODELS[@]}"; do
        if [[ "$m" == "$model" ]]; then
            found=1
            break
        fi
    done
    if [[ $found -eq 0 ]]; then
        echo "Error: Unknown model: $model"
        echo "Available models: ${MODELS[*]}"
        exit 1
    fi
}

# Main execution
main() {
    local dataset_arg="${1:-all}"
    local model_arg="${2:-}"

    # Show help if requested
    if [[ "$dataset_arg" =~ ^(-h|--help|help)$ ]]; then
        echo "Usage: $0 [dataset] [model]"
        echo ""
        echo "Arguments:"
        echo "  dataset   Dataset to test: ${DATASETS[*]}, or 'all' (default: all)"
        echo "  model     Optional model name. If not specified, runs all models for the dataset"
        echo ""
        echo "Examples:"
        echo "  $0                    # Run all models on all datasets"
        echo "  $0 ETTm1              # Run all models on ETTm1 dataset only"
        echo "  $0 ETTm1 MODE         # Run only MODE model on ETTm1 dataset"
        echo "  $0 all MODE           # Run MODE model on all datasets"
        echo "  $0 all                 # Run all models on all datasets"
        exit 0
    fi

    # Determine datasets to process
    local datasets_to_run=()
    if [[ "$dataset_arg" == "all" ]]; then
        datasets_to_run=("${DATASETS[@]}")
    else
        validate_dataset "$dataset_arg"
        datasets_to_run=("$dataset_arg")
    fi

    # Determine models to process
    local models_to_run=()
    if [[ -z "$model_arg" ]]; then
        # No model specified, run all models
        models_to_run=("${MODELS[@]}")
    else
        # Model specified, validate and use only that model
        validate_model "$model_arg"
        models_to_run=("$model_arg")
    fi

    echo "Running lookback experiments"
    echo "Datasets: ${datasets_to_run[*]}"
    echo "Models: ${models_to_run[*]}"
    echo ""

    # Set output directory for lookback experiments
    export LOOKBACK_OUTPUT_DIR="./output/lookback"

    # Process combinations
    local completed_experiments=0
    local failed_experiments=0
    local total_experiments=$((${#datasets_to_run[@]} * ${#models_to_run[@]}))
    local current_experiment=0

    for dataset in "${datasets_to_run[@]}"; do
        for model in "${models_to_run[@]}"; do
            ((current_experiment = current_experiment + 1))
            local script=$(get_model_script "$dataset" "$model")

            if [[ ! -f "$script" ]]; then
                echo "[$current_experiment/$total_experiments] Warning: Script not found: $script"
                echo "Skipping $dataset for model $model"
                ((failed_experiments++))
                continue
            fi

            echo "================================================================="
            echo "Experiment $current_experiment of $total_experiments"
            echo "Dataset: ${dataset^^}"
            echo "Model: $model"
            echo "Script: $script"
            echo "================================================================="

            # Make script executable
            chmod +x "$script"

            # Run directly with error handling to prevent set -e from exiting
            echo "Running experiment..."
            # Ensure LOOKBACK_OUTPUT_DIR is passed to the script
            if bash -c "export LOOKBACK_OUTPUT_DIR='$LOOKBACK_OUTPUT_DIR' && $script"; then
                echo "✓ ${dataset} - ${model} completed successfully"
                ((completed_experiments++))
            else
                exit_code=$?
                echo "✗ ${dataset} - ${model} failed with exit code $exit_code"
                ((failed_experiments++))
                echo "Continuing with next experiment (due to error handling)..."
            fi

            # Wait a bit before starting next experiment
            if [[ $current_experiment -lt $total_experiments ]]; then
                echo "Waiting 5 seconds before next experiment..."
                sleep 5
            fi

            echo ""
        done
    done

    echo "================================================================="
    echo "Lookback Experiments Summary"
    echo "================================================================="
    echo "Datasets: ${datasets_to_run[*]}"
    echo "Models: ${models_to_run[*]}"
    echo "Completed: $completed_experiments / $total_experiments"
    echo "Failed: $failed_experiments"

    if [[ $failed_experiments -eq 0 ]]; then
        echo "All experiments completed successfully! ✓"
        exit 0
    else
        echo "$failed_experiments experiment(s) failed"
        exit 1
    fi
}

# Run main function
main "$@"
