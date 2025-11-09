# Benchmark Logging System

This directory contains a comprehensive JSON-based logging system for Orion FHE benchmarks.

## Overview

All benchmark scripts now automatically create detailed JSON logs in the `logs/` directory with:
- Complete metadata (timestamp, hostname, platform info)
- Model and dataset information
- Per-phase timing breakdowns
- Final results (MAE, precision, outputs)
- Error information (if failed)

## Log Structure

Each log file contains:

```json
{
  "metadata": {
    "benchmark_name": "resnet20_relu_cifar10",
    "timestamp_start": "2025-01-09T12:34:56",
    "timestamp_end": "2025-01-09T12:45:23",
    "hostname": "server-01",
    "platform": "Linux-5.15.0",
    "python_version": "3.10.12",
    "pytorch_version": "2.0.1",
    "cuda_available": false,
    "config_file": "../configs/resnet.yml"
  },
  "model": {
    "name": "ResNet-20",
    "architecture": "ResNet20",
    "dataset": "CIFAR-10",
    "batch_size": 1,
    "seed": 42
  },
  "phases": {
    "initialization": {"duration_seconds": 0.5, "status": "success"},
    "data_loading": {"duration_seconds": 2.3, "status": "success"},
    "cleartext_inference": {"duration_seconds": 0.1, "status": "success"},
    "fit": {"duration_seconds": 5.2, "status": "success"},
    "compile": {"duration_seconds": 3.7, "input_level": 14, "status": "success"},
    "encode": {"duration_seconds": 0.3, "status": "success"},
    "encrypt": {"duration_seconds": 1.2, "status": "success"},
    "fhe_inference": {"duration_seconds": 45.6, "status": "success"},
    "decrypt": {"duration_seconds": 0.8, "status": "success"},
    "decode": {"duration_seconds": 0.2, "status": "success"}
  },
  "results": {
    "mae": 0.000123,
    "precision_bits": 12.67,
    "cleartext_output_sample": [0.123, 0.456, 0.789, ...],
    "fhe_output_sample": [0.124, 0.455, 0.788, ...]
  },
  "timing": {
    "total_duration_seconds": 59.9,
    "fhe_inference_only": 45.6,
    "preprocessing_total": 13.5
  },
  "status": "success",
  "error": null
}
```

## Files

### Core Components

- **`benchmark_logger.py`**: Main logging utility class
  - `BenchmarkLogger`: Context manager for structured logging
  - Automatic file creation with timestamps
  - Phase-based timing tracking
  - Error handling and status tracking

### Updated Benchmark Scripts

All scripts now use the logging system:
- `run_resnet20_relu_cifar10.py`
- `run_resnet18_relu_tiny.py`
- `run_resnet34_relu_imagenet.py`

### Analysis Tool

- **`analyze_logs.py`**: Command-line tool for analyzing logs

## Usage

### Running Benchmarks

Simply run any benchmark script as usual:

```bash
cd /home/orion/examples
uv run python run_resnet20_relu_cifar10.py
```

Logs are automatically saved to `logs/resnet20_relu_cifar10_YYYYMMDD_HHMMSS.json`

### Analyzing Logs

View all benchmark results:
```bash
python analyze_logs.py
```

Show only the latest run of each benchmark:
```bash
python analyze_logs.py --latest
```

Filter by benchmark name:
```bash
python analyze_logs.py --benchmark resnet20
```

Show detailed phase breakdown:
```bash
python analyze_logs.py --verbose
```

Export summary to JSON:
```bash
python analyze_logs.py --export summary.json
```

### Programmatic Access

```python
from benchmark_logger import BenchmarkLogger

# Create logger
logger = BenchmarkLogger("my_benchmark")
logger.set_config("../configs/my_config.yml")
logger.set_model_info(
    name="MyModel",
    architecture="MyArch",
    dataset="MyDataset",
    batch_size=1,
    seed=42
)

try:
    # Track phases
    logger.start_phase("preprocessing")
    # ... do work ...
    logger.end_phase("preprocessing")

    logger.start_phase("inference")
    # ... do inference ...
    logger.end_phase("inference")

    # Record results
    logger.set_results(mae, precision, cleartext_out, fhe_out)
    logger.finalize(status="success")

except Exception as e:
    logger.finalize(status="failed", error=str(e))
    raise
```

## Log File Naming

Format: `{benchmark_name}_{YYYYMMDD_HHMMSS}.json`

Examples:
- `resnet20_relu_cifar10_20250109_123456.json`
- `resnet18_relu_tiny_20250109_145623.json`

## Integration with run_all_benchmarks.sh

The `/home/run_all_benchmarks.sh` script automatically benefits from this logging:
- Each Orion benchmark creates its own detailed JSON log
- Logs persist even if tmux session is lost
- Easy to track progress by checking the logs directory
- Can analyze partial results even if benchmarks fail

## Tips

1. **Monitor progress**: Check the logs directory to see which benchmarks have completed
2. **Debug failures**: Failed runs include error messages and stack traces in the JSON
3. **Compare runs**: Use timestamps in filenames to track performance over time
4. **Aggregate data**: Use `analyze_logs.py --export` to create summary reports

## Example Analysis Output

```
================================================================================
BENCHMARK SUMMARY - 3 log(s)
================================================================================

✓ RESNET20_RELU_CIFAR10
  ────────────────────────────────────────────────────────────────────────────
  Model:       ResNet-20 (ResNet20)
  Dataset:     CIFAR-10
  Status:      SUCCESS
  Started:     2025-01-09T12:34:56
  Duration:    59.90s (1.00m)
  FHE Time:    45.60s
  MAE:         0.000123
  Precision:   12.67 bits

✓ RESNET18_RELU_TINY
  ────────────────────────────────────────────────────────────────────────────
  Model:       ResNet-18 (ResNet18)
  Dataset:     Tiny ImageNet
  Status:      SUCCESS
  Started:     2025-01-09T13:45:12
  Duration:    125.34s (2.09m)
  FHE Time:    98.23s
  MAE:         0.000234
  Precision:   11.89 bits

================================================================================
STATISTICS
================================================================================
  Total benchmarks:     3
  Successful:           2
  Failed:               1
  Total time:           185.24s (3.09m)
  Total FHE time:       143.83s (2.40m)
  Avg time/benchmark:   92.62s
```
