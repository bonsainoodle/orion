#!/bin/bash

# Script to run all ResNet examples sequentially using uv run
# Logs are saved to logs/ directory with timestamps

set -e  # Exit on error

# Create logs directory if it doesn't exist
LOGS_DIR="logs"
mkdir -p "$LOGS_DIR"

# Get timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# List of scripts to run
SCRIPTS=(
    "run_resnet18_relu_cifar10.py"
    "run_resnet18_relu_imagenet.py"
    "run_resnet18_relu_tiny.py"
    "run_resnet20_relu_cifar10.py"
    "run_resnet20_relu_tiny.py"
    "run_resnet20_silu_cifar10.py"
    "run_resnet34_relu_cifar10.py"
    "run_resnet34_relu_imagenet.py"
    "run_resnet34_relu_tiny.py"
)

echo "========================================="
echo "Starting ResNet benchmark suite"
echo "Timestamp: $TIMESTAMP"
echo "Logs will be saved to: $LOGS_DIR/"
echo "========================================="
echo

# Run each script
for script in "${SCRIPTS[@]}"; do
    echo "Running: $script"
    log_file="$LOGS_DIR/${script%.py}_${TIMESTAMP}.log"

    if uv run "$script" 2>&1 | tee "$log_file"; then
        echo "✓ Completed: $script"
        echo "  Log: $log_file"
    else
        echo "✗ Failed: $script (exit code: $?)"
        echo "  Log: $log_file"
        echo "  Continuing to next script..."
    fi
    echo
done

echo "========================================="
echo "Benchmark suite completed"
echo "All logs saved to: $LOGS_DIR/"
echo "========================================="
