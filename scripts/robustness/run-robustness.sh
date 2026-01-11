#!/bin/bash

# =============================================================================
# MODE Robustness Testing Runner (Simplified)
# =============================================================================
# This script runs robustness experiments for different datasets with Gaussian noise.
# It tests model performance under various noise levels.
#
# Usage: bash scripts/robustness/run-robustness.sh [dataset]
#   dataset: Optional - specify dataset (ETTm1, Weather, ECL, etc.)
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Change to the repository root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

echo -e "${BLUE}MODE Robustness Testing Runner${NC}"
echo "Repository root: $REPO_ROOT"
echo ""

# Configuration
DATASETS=("ETTm1" "Weather" "ECL")
NOISE_LEVELS=(0.1 0.2 0.0)  # Test with 10% and 20% noise

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

# Get script path for dataset
get_dataset_script() {
    local dataset=$1
    echo "./scripts/robustness/${dataset}.sh"
}

# Main execution
main() {
    local dataset_arg="${1:-all}"

    # Show help if requested
    if [[ "$dataset_arg" =~ ^(-h|--help|help)$ ]]; then
        echo "Usage: $0 [dataset]"
        echo ""
        echo "Arguments:"
        echo "  dataset        Dataset to test: ${DATASETS[*]}, or 'all' (default: all)"
        echo ""
        echo "Examples:"
        echo "  $0                    # Test all datasets with all noise levels"
        echo "  $0 ETTm1              # Test ETTm1 only"
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

    echo -e "${YELLOW}Testing datasets: ${datasets_to_run[*]}${NC}"
    echo -e "${YELLOW}Noise levels: ${NOISE_LEVELS[*]}${NC}"
    echo ""

    # Process datasets and noise levels
    local total_experiments=0
    local failed_experiments=0

    for dataset in "${datasets_to_run[@]}"; do
        local script=$(get_dataset_script "$dataset")

        if [[ ! -f "$script" ]]; then
            echo -e "${RED}Error: Script not found: $script${NC}"
            continue
        fi

        # Test each noise level
        for noise_level in "${NOISE_LEVELS[@]}"; do
            echo -e "${BLUE}=================================================================${NC}"
            echo -e "${BLUE}Dataset: ${dataset^^} | Noise Level: ${noise_level}${NC}"
            echo -e "${BLUE}=================================================================${NC}"

            echo -e "${CYAN}Running robustness experiments with noise_level=$noise_level...${NC}"

            # Export noise_level as environment variable
            export noise_level=$noise_level

            if ! bash "$script"; then
                echo -e "${RED}✗ ${dataset} experiments failed (noise_level=$noise_level)${NC}"
                ((failed_experiments++))
            else
                echo -e "${GREEN}✓ ${dataset} experiments completed (noise_level=$noise_level)${NC}"
            fi
            ((total_experiments++))

            echo -e "${YELLOW}Waiting 2 seconds before next experiment...${NC}"
            sleep 2
        done

        if [[ ${#datasets_to_run[@]} -gt 1 ]]; then
            echo -e "\n${YELLOW}Waiting before next dataset...${NC}"
            sleep 5
        fi
    done

    # Summary
    echo -e "\n${BLUE}=================================================================${NC}"
    echo -e "${BLUE}Robustness Testing Summary${NC}"
    echo -e "${BLUE}=================================================================${NC}"
    echo -e "Total experiment groups: ${total_experiments}"
    echo -e "Completed: ${GREEN}$((total_experiments - failed_experiments))${NC}"
    echo -e "Failed: ${RED}${failed_experiments}${NC}"

    # Clear environment variable
    unset noise_level

    if [[ $failed_experiments -eq 0 ]]; then
        echo -e "\n${GREEN}All tests completed successfully! ✓${NC}"
        exit 0
    else
        echo -e "\n${YELLOW}${failed_experiments} experiment group(s) failed${NC}"
        exit 1
    fi
}

# Run main function
main "$@"
