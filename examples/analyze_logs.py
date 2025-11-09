#!/usr/bin/env python3
"""
Analyze and summarize benchmark logs from the logs/ directory.

Usage:
    python analyze_logs.py                  # Analyze all logs
    python analyze_logs.py --latest         # Show only most recent run of each benchmark
    python analyze_logs.py --benchmark resnet20  # Filter by benchmark name
    python analyze_logs.py --export summary.json # Export summary to file
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import sys


def load_logs(log_dir: Path = Path("logs")) -> List[Dict[str, Any]]:
    """Load all JSON log files from the logs directory."""
    if not log_dir.exists():
        print(f"Error: Log directory '{log_dir}' does not exist")
        return []

    logs = []
    for log_file in sorted(log_dir.glob("*.json")):
        try:
            with open(log_file, 'r') as f:
                data = json.load(f)
                data['_log_file'] = str(log_file)
                logs.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {log_file}: {e}", file=sys.stderr)

    return logs


def filter_latest(logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only the most recent log for each benchmark."""
    benchmark_map = {}
    for log in logs:
        name = log['metadata']['benchmark_name']
        timestamp = log['metadata']['timestamp_start']
        if name not in benchmark_map or timestamp > benchmark_map[name]['metadata']['timestamp_start']:
            benchmark_map[name] = log
    return list(benchmark_map.values())


def print_summary(logs: List[Dict[str, Any]], verbose: bool = False):
    """Print a formatted summary of benchmark results."""
    if not logs:
        print("No logs found.")
        return

    print(f"\n{'='*80}")
    print(f"BENCHMARK SUMMARY - {len(logs)} log(s)")
    print(f"{'='*80}\n")

    for log in sorted(logs, key=lambda x: x['metadata']['timestamp_start']):
        meta = log['metadata']
        model = log['model']
        timing = log.get('timing', {})
        results = log.get('results', {})
        status = log['status']

        # Status indicator
        status_symbol = "✓" if status == "success" else "✗"

        print(f"{status_symbol} {meta['benchmark_name'].upper()}")
        print(f"  {'─'*76}")
        print(f"  Model:       {model.get('name', 'N/A')} ({model.get('architecture', 'N/A')})")
        print(f"  Dataset:     {model.get('dataset', 'N/A')}")
        print(f"  Status:      {status.upper()}")
        print(f"  Started:     {meta['timestamp_start']}")

        if status == "success":
            print(f"  Duration:    {timing.get('total_duration_seconds', 0):.2f}s ({timing.get('total_duration_seconds', 0)/60:.2f}m)")
            print(f"  FHE Time:    {timing.get('fhe_inference_only', 0):.2f}s")
            print(f"  MAE:         {results.get('mae', 'N/A'):.6f}")
            print(f"  Precision:   {results.get('precision_bits', 'N/A'):.2f} bits")
        else:
            print(f"  Error:       {log.get('error', 'Unknown error')}")

        if verbose:
            print(f"\n  Phase Breakdown:")
            phases = log.get('phases', {})
            for phase_name, phase_data in phases.items():
                duration = phase_data.get('duration_seconds', 0)
                status_str = phase_data.get('status', 'unknown')
                print(f"    {phase_name:20s} {duration:8.2f}s  [{status_str}]")

        print()

    # Overall statistics
    successful = [l for l in logs if l['status'] == 'success']
    if successful:
        print(f"{'='*80}")
        print(f"STATISTICS")
        print(f"{'='*80}")
        print(f"  Total benchmarks:     {len(logs)}")
        print(f"  Successful:           {len(successful)}")
        print(f"  Failed:               {len(logs) - len(successful)}")

        total_time = sum(l.get('timing', {}).get('total_duration_seconds', 0) for l in successful)
        fhe_time = sum(l.get('timing', {}).get('fhe_inference_only', 0) for l in successful)

        print(f"  Total time:           {total_time:.2f}s ({total_time/60:.2f}m)")
        print(f"  Total FHE time:       {fhe_time:.2f}s ({fhe_time/60:.2f}m)")
        print(f"  Avg time/benchmark:   {total_time/len(successful):.2f}s")
        print()


def export_summary(logs: List[Dict[str, Any]], output_file: Path):
    """Export a compact summary to JSON."""
    summary = {
        "generated_at": datetime.now().isoformat(),
        "total_benchmarks": len(logs),
        "successful": len([l for l in logs if l['status'] == 'success']),
        "failed": len([l for l in logs if l['status'] == 'failed']),
        "benchmarks": []
    }

    for log in logs:
        summary['benchmarks'].append({
            "name": log['metadata']['benchmark_name'],
            "model": log['model'].get('name'),
            "dataset": log['model'].get('dataset'),
            "status": log['status'],
            "timestamp": log['metadata']['timestamp_start'],
            "total_time_seconds": log.get('timing', {}).get('total_duration_seconds', 0),
            "fhe_time_seconds": log.get('timing', {}).get('fhe_inference_only', 0),
            "mae": log.get('results', {}).get('mae'),
            "precision_bits": log.get('results', {}).get('precision_bits'),
            "log_file": log.get('_log_file')
        })

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark logs")
    parser.add_argument('--log-dir', type=Path, default=Path('logs'),
                        help='Directory containing log files (default: logs/)')
    parser.add_argument('--latest', action='store_true',
                        help='Show only the most recent log for each benchmark')
    parser.add_argument('--benchmark', type=str,
                        help='Filter by benchmark name (partial match)')
    parser.add_argument('--export', type=Path,
                        help='Export summary to JSON file')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed phase breakdown')

    args = parser.parse_args()

    # Load logs
    logs = load_logs(args.log_dir)

    if not logs:
        sys.exit(1)

    # Apply filters
    if args.benchmark:
        logs = [l for l in logs if args.benchmark.lower() in l['metadata']['benchmark_name'].lower()]
        if not logs:
            print(f"No logs found matching '{args.benchmark}'")
            sys.exit(1)

    if args.latest:
        logs = filter_latest(logs)

    # Display or export
    if args.export:
        export_summary(logs, args.export)

    print_summary(logs, verbose=args.verbose)


if __name__ == '__main__':
    main()
