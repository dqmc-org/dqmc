#!/bin/bash

# Configuration
N_RUNS=4
CONFIG_FILE="../example/config.toml"
OUTPUT_DIR="../example/output_parallel"
EXE="../build/main"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate run sequence and execute in parallel
echo "Starting $N_RUNS parallel DQMC runs..."
seq 0 $((N_RUNS-1)) | \
parallel -j $(nproc) \
  "$EXE --config $CONFIG_FILE --output $OUTPUT_DIR --run-id {}"

# Aggregate results
echo "Aggregating results..."
python3 ../tools/aggregate_results.py \
  --input-dir "$OUTPUT_DIR" \
  --output-dir "$OUTPUT_DIR"

echo "Parallel execution complete. Results aggregated in $OUTPUT_DIR"