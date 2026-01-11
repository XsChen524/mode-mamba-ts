#!/usr/bin/env python3
"""
Convert result_long_term_forecast.txt to CSV format.
Each dataset occupies 5 rows: header + 4 experiment results (96, 192, 336, 720).
"""

import sys
import csv
from pathlib import Path
from datetime import datetime

# Configuration
RESULTS_FILE = Path(__file__).parent.parent / "result_long_term_forecast.txt"
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "csv"
DATASETS = [
    "ETTm1",
    "ETTm2",
    "ETTh1",
    "ETTh2",
    "Electricity",
    "Exchange",
    "Weather",
    "Solar-Energy",
    # "PEMS04",
    "PEMS08",
]

# Prediction lengths for different dataset types
PRED_LENS_STANDARD = ["96", "192", "336", "720"]  # For most datasets
PRED_LENS_PEMS = ["12", "24", "48", "96"]  # For PEMS datasets

# Mapping for dataset names in the results file
DATASET_MAPPING = {
    "ETTm1": "ETTm1",
    "ETTm2": "ETTm2",
    "ETTh1": "ETTh1",
    "ETTh2": "ETTh2",
    "ECL": "Electricity",
    "Exchange": "Exchange",
    "Weather": "Weather",
    "solar": "Solar-Energy",
    # "PEMS04": "PEMS04",
    "PEMS08": "PEMS08",
}


def get_pred_lens_for_dataset(dataset):
    """Get prediction lengths for a specific dataset."""
    if dataset.startswith("PEMS"):
        return PRED_LENS_PEMS
    else:
        return PRED_LENS_STANDARD


def parse_results_for_dataset(file_path, target_dataset):
    """Parse the results file and extract MSE/MAE values for a specific dataset and all prediction lengths."""
    pred_lens = get_pred_lens_for_dataset(target_dataset)
    # Initialize with None to distinguish between "not found" and "found but is nan"
    results = {pred_len: {"mse": None, "mae": None} for pred_len in pred_lens}

    if not file_path.exists():
        print(f"Error: Results file '{file_path}' not found.")
        return results

    with open(file_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines
        if not line:
            i += 1
            continue

        # Check if this is a configuration line (contains experiment settings)
        if "mse:" not in line and "mae:" not in line:
            # Extract dataset name from the configuration
            # Format: {dataset}_96_{pred_len}_...
            parts = line.split("_")
            if len(parts) > 3:
                raw_dataset = parts[0]
                pred_len = parts[2]

                # Map to standardized dataset name
                dataset = DATASET_MAPPING.get(raw_dataset)

                # Check if this is the target dataset and valid prediction length
                if dataset == target_dataset and pred_len in pred_lens:
                    # Next line should contain the results
                    if i + 1 < len(lines):
                        result_line = lines[i + 1].strip()
                        if "mse:" in result_line and "mae:" in result_line:
                            # Parse MSE and MAE values
                            try:
                                # Extract mse and mae values
                                mse_part = (
                                    result_line.split("mse:")[1].split(",")[0].strip()
                                )
                                mae_part = result_line.split("mae:")[1].strip()

                                mse_val = float(mse_part)
                                mae_val = float(mae_part)

                                # Round to 3 decimal places (rounding from 4th decimal)
                                mse_rounded = round(mse_val, 3)
                                mae_rounded = round(mae_val, 3)

                                results[pred_len] = {
                                    "mse": mse_rounded,
                                    "mae": mae_rounded,
                                }
                            except (ValueError, IndexError) as e:
                                print(
                                    f"Warning: Could not parse results for {dataset}_{pred_len}: {e}"
                                )
                                results[pred_len] = {"mse": "nan", "mae": "nan"}
                    # If next line doesn't contain results but line still matches, skip
                    elif i + 1 < len(lines):
                        result_line = lines[i + 1].strip()
                        if "mse:" not in result_line and "mae:" not in result_line:
                            # No valid results found for this entry
                            results[pred_len] = {"mse": None, "mae": None}

        i += 1

    return results


def create_csv(all_results):
    """Create CSV file from parsed results."""
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp: mode_MMDD_HHMMSS.csv
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"mode_{timestamp}.csv"

    # Prepare CSV data - each dataset occupies 5 rows
    csv_data = []

    for dataset in DATASETS:
        # First row: dataset header (MSE first, then MAE)
        header_row = [f"{dataset}_MSE", f"{dataset}_MAE"]
        csv_data.append(header_row)

        # Next 4 rows: results for prediction lengths (different for PEMS vs others)
        results = all_results.get(dataset, {})
        pred_lens = get_pred_lens_for_dataset(dataset)
        for pred_len in pred_lens:
            values = results.get(pred_len, {"mae": None, "mse": None})

            # Handle different value types:
            # - None: not found in results (keep empty)
            # - 'nan' or 'NAN': explicitly a nan value
            # - other values: use as is
            def normalize_value(val):
                if val is None:
                    return ""
                elif isinstance(val, str) and val.lower() == "nan":
                    return "nan"
                else:
                    return val

            mse_val = normalize_value(values["mse"])
            mae_val = normalize_value(values["mae"])
            values_row = [mse_val, mae_val]
            csv_data.append(values_row)

    # Write CSV file
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)

    print(f"CSV file created: {output_file}")
    print(f"\nDataset coverage:")
    for dataset in DATASETS:
        results = all_results.get(dataset, {})
        pred_lens = get_pred_lens_for_dataset(dataset)
        found_results = sum(
            1 for pl in pred_lens if results.get(pl, {}).get("mae") is not None
        )
        total_results = len(pred_lens)
        status = (
            "✓" if found_results == total_results else "~" if found_results > 0 else "✗"
        )
        print(
            f"  {status} {dataset}: {found_results}/{total_results} experiment values found"
        )

    return output_file


def main():
    """Main execution function."""
    print("Parsing results file for all datasets...")
    all_results = {}

    for dataset in DATASETS:
        print(f"  Extracting results for {dataset}...")
        all_results[dataset] = parse_results_for_dataset(RESULTS_FILE, dataset)

    print("\nDataset summary:")
    for dataset in DATASETS:
        results = all_results[dataset]
        pred_lens = get_pred_lens_for_dataset(dataset)
        print(f"\n{dataset}:")
        for pred_len in pred_lens:
            values = results.get(pred_len, {})
            mae = values.get("mae")
            mse = values.get("mse")
            # Check if value is None (not found) vs 'nan' (found but is nan) vs actual value
            if mae is None:
                print(f"  {pred_len}: (missing)")
            elif isinstance(mae, str) and mae.lower() == "nan":
                print(f"  {pred_len}: MSE=nan, MAE=nan")
            else:
                print(f"  {pred_len}: MSE={mse}, MAE={mae}")

    print("\nCreating CSV file...")
    output_file = create_csv(all_results)

    print(f"\nDone! CSV saved to: {output_file}")


if __name__ == "__main__":
    main()
