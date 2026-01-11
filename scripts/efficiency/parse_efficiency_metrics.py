#!/usr/bin/env python3
"""
Script to parse training output and extract efficiency metrics.
Extracts: MSE, training time (ms/iter), GPU memory (Allocated, Cached, Total)
"""

import re
import sys
import argparse

def parse_training_output(log_content):
    """Parse training log content and extract metrics."""
    metrics = {
        'mse': 0.0,
        'mae': 0.0,
        'train_time_ms_per_iter': 0.0,
        'avg_gpu_mem_allocated_mb': 0.0
    }

    # Extract MSE and MAE (format: mse:X, mae:Y)
    mse_mae_match = re.search(r'mse:([0-9\.e\-]+),\s*mae:([0-9\.e\-]+)', log_content)
    if mse_mae_match:
        metrics['mse'] = float(mse_mae_match.group(1))
        metrics['mae'] = float(mse_mae_match.group(2))

    # Extract training time (speed: Xs/iter)
    speed_matches = re.findall(r'speed: ([0-9\.]+)s/iter', log_content)
    if speed_matches:
        # Convert to ms/iter and take average
        times_ms = [float(t) * 1000 for t in speed_matches]
        metrics['train_time_ms_per_iter'] = sum(times_ms) / len(times_ms)

    # Extract GPU memory details - simple format: allocated_memory: X
    gpu_matches = re.findall(r'allocated_memory: ([0-9\.]+)', log_content)
    if gpu_matches:
        allocated_values = []
        for alloc in gpu_matches:
            allocated_mb = float(alloc) * 1024  # Convert GB to MB
            allocated_values.append(allocated_mb)
        metrics['avg_gpu_mem_allocated_mb'] = sum(allocated_values) / len(allocated_values)

    return metrics

def main():
    parser = argparse.ArgumentParser(description="Parse training output and extract efficiency metrics")
    parser.add_argument("results_file", help="Path to results file")
    parser.add_argument("model_name", help="Model name")
    parser.add_argument("dataset", help="Dataset name")
    parser.add_argument("timestamp", help="Timestamp for the experiment")
    parser.add_argument("model_id", help="Model ID in format DATASET_SEQ_LEN_PRED_LEN")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--train_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--d_model", type=int, help="Model dimension")
    parser.add_argument("--e_layers", type=int, help="Number of encoder layers")
    parser.add_argument("--seq_len", type=int, help="Sequence length")
    parser.add_argument("--pred_len", type=int, help="Prediction length")

    args = parser.parse_args()

    # Read from stdin
    log_content = sys.stdin.read()

    try:
        metrics = parse_training_output(log_content)

        # Write results to file
        with open(args.results_file, 'a') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Model: {args.model_name}, Dataset: {args.dataset}, Model ID: {args.model_id}, Timestamp: {args.timestamp}\n")
            f.write("-" * 80 + "\n")
            f.write("Configuration:\n")
            f.write(f"  - Model: {args.model_name}\n")
            f.write(f"  - Dataset: {args.dataset}\n")
            f.write(f"  - Model ID: {args.model_id}\n")
            f.write(f"  - Batch Size: {args.batch_size}\n")
            f.write(f"  - Train Epochs: {args.train_epochs}\n")
            f.write(f"  - Learning Rate: {args.learning_rate}\n")
            f.write(f"  - D Model: {args.d_model}\n")
            f.write(f"  - E Layers: {args.e_layers}\n")
            f.write(f"  - Seq Len: {args.seq_len}\n")
            f.write(f"  - Pred Len: {args.pred_len}\n")
            f.write("-" * 80 + "\n")
            f.write("Results:\n")
            f.write(f"  - MSE: {metrics['mse']:.8f}\n")
            f.write(f"  - MAE: {metrics['mae']:.8f}\n")
            f.write(f"  - Training Time: {metrics['train_time_ms_per_iter']:.2f} ms/iter\n")
            f.write(f"  - Avg Allocated GPU Memory: {metrics['avg_gpu_mem_allocated_mb']:.2f} MB\n")
            f.write("=" * 80 + "\n\n")

        print(f"Successfully parsed metrics for {args.model_name} on {args.dataset}")
        print(f"  Model ID: {args.model_id}")
        print(f"  MSE: {metrics['mse']:.8f}")
        print(f"  MAE: {metrics['mae']:.8f}")
        print(f"  Training Time: {metrics['train_time_ms_per_iter']:.2f} ms/iter")
        print(f"  Avg Allocated GPU Memory: {metrics['avg_gpu_mem_allocated_mb']:.2f} MB")
        print(f"Results appended to {args.results_file}")

    except Exception as e:
        print(f"Error parsing output: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
