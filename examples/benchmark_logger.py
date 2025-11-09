"""
Benchmark logging utility for Orion FHE experiments.

Creates structured JSON logs with comprehensive metadata, metrics, and timing information.
"""

import json
import time
import socket
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import torch


class BenchmarkLogger:
    """
    Logger for FHE benchmark experiments that outputs structured JSON.

    Log structure:
    {
        "metadata": {
            "benchmark_name": str,
            "timestamp_start": str (ISO format),
            "timestamp_end": str (ISO format),
            "hostname": str,
            "platform": str,
            "python_version": str,
            "pytorch_version": str,
            "cuda_available": bool,
            "config_file": str
        },
        "model": {
            "name": str,
            "architecture": str,
            "dataset": str,
            "batch_size": int,
            "seed": int
        },
        "phases": {
            "data_loading": {"duration_seconds": float, "status": str},
            "cleartext_inference": {"duration_seconds": float, "status": str},
            "fit": {"duration_seconds": float, "status": str},
            "compile": {"duration_seconds": float, "input_level": int, "status": str},
            "encode": {"duration_seconds": float, "status": str},
            "encrypt": {"duration_seconds": float, "status": str},
            "fhe_inference": {"duration_seconds": float, "status": str},
            "decrypt": {"duration_seconds": float, "status": str},
            "decode": {"duration_seconds": float, "status": str}
        },
        "results": {
            "mae": float,
            "precision_bits": float,
            "cleartext_output_sample": list,
            "fhe_output_sample": list
        },
        "timing": {
            "total_duration_seconds": float,
            "fhe_inference_only": float,
            "preprocessing_total": float
        },
        "status": "success" | "failed",
        "error": Optional[str]
    }
    """

    def __init__(self, benchmark_name: str, log_dir: str = "logs"):
        self.benchmark_name = benchmark_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.start_time = time.time()
        self.start_timestamp = datetime.now().isoformat()

        self.log_data = {
            "metadata": {
                "benchmark_name": benchmark_name,
                "timestamp_start": self.start_timestamp,
                "timestamp_end": None,
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "pytorch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "config_file": None
            },
            "model": {},
            "phases": {},
            "results": {},
            "timing": {},
            "status": "running",
            "error": None
        }

        self.phase_timers = {}

    def set_config(self, config_file: str):
        """Set the configuration file path."""
        self.log_data["metadata"]["config_file"] = config_file

    def set_model_info(self, name: str, architecture: str, dataset: str,
                       batch_size: int = 1, seed: int = 42):
        """Set model metadata."""
        self.log_data["model"] = {
            "name": name,
            "architecture": architecture,
            "dataset": dataset,
            "batch_size": batch_size,
            "seed": seed
        }

    def start_phase(self, phase_name: str):
        """Start timing a phase."""
        self.phase_timers[phase_name] = time.time()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting {phase_name}...", flush=True)

    def end_phase(self, phase_name: str, status: str = "success", **kwargs):
        """
        End timing a phase and record results.

        Args:
            phase_name: Name of the phase
            status: "success" or "failed"
            **kwargs: Additional phase-specific data to record
        """
        if phase_name not in self.phase_timers:
            raise ValueError(f"Phase {phase_name} was not started")

        duration = time.time() - self.phase_timers[phase_name]

        phase_data = {
            "duration_seconds": round(duration, 4),
            "status": status
        }
        phase_data.update(kwargs)

        self.log_data["phases"][phase_name] = phase_data
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Completed {phase_name} in {duration:.2f}s", flush=True)

    def set_results(self, mae: float, precision_bits: float,
                    cleartext_output: torch.Tensor, fhe_output: torch.Tensor,
                    sample_size: int = 5):
        """Record final results."""
        # Convert tensors to lists for JSON serialization
        clear_sample = cleartext_output.flatten()[:sample_size].tolist()
        fhe_sample = fhe_output.flatten()[:sample_size].tolist()

        self.log_data["results"] = {
            "mae": round(mae, 6),
            "precision_bits": round(precision_bits, 4),
            "cleartext_output_sample": [round(x, 6) for x in clear_sample],
            "fhe_output_sample": [round(x, 6) for x in fhe_sample]
        }

    def finalize(self, status: str = "success", error: Optional[str] = None):
        """Finalize the log and write to disk."""
        end_time = time.time()
        total_duration = end_time - self.start_time

        self.log_data["metadata"]["timestamp_end"] = datetime.now().isoformat()
        self.log_data["status"] = status
        self.log_data["error"] = error

        # Calculate timing summary
        fhe_inference_time = self.log_data["phases"].get("fhe_inference", {}).get("duration_seconds", 0)
        preprocessing_phases = ["data_loading", "cleartext_inference", "fit", "compile", "encode", "encrypt"]
        preprocessing_total = sum(
            self.log_data["phases"].get(p, {}).get("duration_seconds", 0)
            for p in preprocessing_phases
        )

        self.log_data["timing"] = {
            "total_duration_seconds": round(total_duration, 4),
            "fhe_inference_only": round(fhe_inference_time, 4),
            "preprocessing_total": round(preprocessing_total, 4)
        }

        # Write to file
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{self.benchmark_name}_{timestamp_str}.json"
        log_path = self.log_dir / log_filename

        with open(log_path, 'w') as f:
            json.dump(self.log_data, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Benchmark completed: {status.upper()}")
        print(f"Total time: {total_duration:.2f}s ({total_duration/60:.2f}m)")
        if status == "success":
            print(f"MAE: {self.log_data['results']['mae']:.6f}")
            print(f"Precision: {self.log_data['results']['precision_bits']:.2f} bits")
        print(f"Log saved to: {log_path}")
        print(f"{'='*60}\n")

        return log_path

    def get_data(self) -> Dict[str, Any]:
        """Get the current log data."""
        return self.log_data
