#!/bin/bash

# =============================================================================
# MODE Ablation Study Runner
# =============================================================================
# This script runs ablation experiments for different datasets and configurations.
# It executes all experiment scripts for a dataset sequentially.
#
# Usage: bash scripts/ablation/run-ablation.sh [dataset]
#   If no dataset is specified, runs all datasets (ETTm1, Weather, ECL)
# =============================================================================

set -e  # Exit on error

# Set ablation output directory
export ABLATION_OUTPUT_DIR="./output/ablation"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Change to the repository root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

echo -e "${BLUE}MODE Ablation Study Runner${NC}"
echo "Repository root: $REPO_ROOT"
echo ""

# Function to run a single experiment script with error handling
run_experiment() {
    local script_path="$1"
    local config_name="$2"

    echo -e "${YELLOW}Running${NC} $config_name..."

    if [[ ! -f "$script_path" ]]; then
        echo -e "${RED}Error${NC}: Script not found: $script_path"
        return 1
    fi

    # Make script executable
    chmod +x "$script_path"

    # Run the experiment
    if bash "$script_path"; then
        echo -e "${GREEN}✓${NC} $config_name completed successfully"
        return 0
    else
        echo -e "${RED}✗${NC} $config_name failed"
        return 1
    fi
}

# Function to run all experiments for a dataset
run_dataset_experiments() {
    local dataset="$1"
    local dataset_dir="./scripts/ablation/${dataset}"

    echo -e "\n${BLUE}=================================================================${NC}"
    echo -e "${BLUE}Processing dataset: ${dataset^^}${NC}"
    echo -e "${BLUE}=================================================================${NC}"

    # Check if dataset directory exists
    if [[ ! -d "$dataset_dir" ]]; then
        echo -e "${RED}Error${NC}: Dataset directory not found: $dataset_dir"
        return 1
    fi

    # Define all experiment configurations
    local experiments=(
        # "${dataset}_baseline"
        # "${dataset}_no_hippo"
		# "${dataset}_no_ode"
        # "${dataset}_static_ode"
        "${dataset}_s_mamba"
        # "${dataset}_attention_ffn"
        # "${dataset}_linear"
        # "${dataset}_r_rank_small"
        # "${dataset}_r_rank_large"
        # "${dataset}_r_rank_full"
    )

    local failed_experiments=0
    local completed_experiments=0

    # Run each experiment
    for exp_name in "${experiments[@]}"; do
        local script_path="${dataset_dir}/${exp_name}.sh"

        if [[ -f "$script_path" ]]; then
            if run_experiment "$script_path" "$exp_name"; then
                ((completed_experiments++))
                # Small delay between experiments to avoid GPU memory issues
                sleep 2
            else
                ((failed_experiments++))
            fi
        else
            echo -e "${YELLOW}Warning${NC}: Script not found: $script_path (skipping)"
        fi
    done

    echo ""
    echo -e "${GREEN}$completed_experiments${NC} experiments completed, ${RED}$failed_experiments${NC} failed"

    if [[ $failed_experiments -gt 0 ]]; then
        echo -e "${YELLOW}Warning${NC}: Some experiments failed"
    fi

    return $failed_experiments
}

# Function to process a dataset (run experiments only)
process_dataset() {
    local dataset="$1"

    # Run experiments
    run_dataset_experiments "$dataset"
    return $?
}

# Main execution
main() {
    local datasets=()

    # Get datasets from command line or use defaults
    if [[ $# -gt 0 ]]; then
        # Run specific datasets
        for dataset in "$@"; do
            datasets+=("$dataset")
        done
    else
        # Run all datasets
        datasets=("ETTm1" "Weather" "ECL" "PEMS08")
    fi

    echo -e "${BLUE}Running ablation study for datasets: ${datasets[*]}${NC}"
    echo ""

    local failed_datasets=0

    # Process each dataset sequentially (avoid GPU conflicts)
    for dataset in "${datasets[@]}"; do
        if process_dataset "$dataset"; then
            echo -e "\n${GREEN}✓ Dataset ${dataset} completed successfully${NC}"
        else
            echo -e "\n${RED}✗ Dataset ${dataset} failed${NC}"
            ((failed_datasets++))
        fi

        # Delay between datasets
        sleep 5
    done

    # Summary
    echo ""
    echo -e "${BLUE}=================================================================${NC}"
    echo -e "${BLUE}Ablation Study Complete${NC}"
    echo -e "${BLUE}=================================================================${NC}"

    if [[ $failed_datasets -eq 0 ]]; then
        echo -e "${GREEN}All datasets completed successfully!${NC}"
    else
        echo -e "${YELLOW}$failed_datasets dataset(s) had failures${NC}"
    fi
}

# Run main function with all arguments
main "$@"
