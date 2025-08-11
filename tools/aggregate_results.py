#!/usr/bin/env python3

import numpy as np
import glob
import argparse
from pathlib import Path

def aggregate_observable_files(pattern, output_file):
    """Aggregate observable data from multiple runs."""
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"Warning: No files found matching pattern: {pattern}")
        return

    all_data = []
    for f in files:
        try:
            data = np.loadtxt(f)
            all_data.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")
            continue

    if not all_data:
        print(f"Error: No valid data files found for pattern: {pattern}")
        return

    # Combine all bins from all runs
    combined = np.concatenate(all_data)

    # Calculate statistics
    mean = np.mean(combined, axis=0)
    std = np.std(combined, axis=0) / np.sqrt(len(combined))

    # Save aggregated results
    try:
        np.savetxt(output_file, np.column_stack([mean, std]))
        print(f"Aggregated {len(files)} files -> {output_file}")
    except Exception as e:
        print(f"Error: Failed to save {output_file}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='Aggregate DQMC results from parallel runs'
    )
    parser.add_argument('--input-dir', required=True,
                       help='Directory containing individual run results')
    parser.add_argument('--output-dir', required=True,
                       help='Directory to save aggregated results')
    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Aggregate different observable types
    observables = ['swave', 'dos', 'greens', 'dss']

    for obs in observables:
        pattern = f"{args.input_dir}/{obs}_*.out"
        output_file = f"{args.output_dir}/{obs}_aggregated.out"
        aggregate_observable_files(pattern, output_file)

if __name__ == "__main__":
    main()
