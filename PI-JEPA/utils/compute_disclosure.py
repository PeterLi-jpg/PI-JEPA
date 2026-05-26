"""
Compute disclosure helper for the NeurIPS / ICML / ICLR reproducibility
checklists.

Writes a `compute_disclosure.json` that records:
    - hardware (GPU model, count, memory, CPU, RAM)
    - wall clock time
    - estimated GPU-hours
    - optional carbon estimate (via CodeCarbon if installed)

Usage in a training script:

    from utils.compute_disclosure import ComputeRecorder
    with ComputeRecorder("outputs/run1/compute_disclosure.json") as cr:
        train(...)

The recorder writes a fresh JSON when the context exits (success or
failure). Failed runs are recorded too — important for honest GPU-hour
accounting including failed runs that consumed compute.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from contextlib import contextmanager
from typing import Optional

import torch


def _safe_subprocess(cmd) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def detect_hardware() -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(),
        "n_cpus": os.cpu_count(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()
        info["gpus"] = [
            {
                "name": torch.cuda.get_device_name(i),
                "memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2),
            }
            for i in range(torch.cuda.device_count())
        ]
    # CPU memory (best-effort, macOS-friendly)
    sysctl_mem = _safe_subprocess(["sysctl", "-n", "hw.memsize"])
    if sysctl_mem.isdigit():
        info["ram_gb"] = round(int(sysctl_mem) / 1e9, 2)
    return info


class ComputeRecorder:
    """Context manager that times a code block and writes compute_disclosure.json.

    Optional CodeCarbon integration: if `codecarbon` is importable and
    `track_carbon=True`, runs an EmissionsTracker around the timed region.
    """

    def __init__(
        self,
        output_path: str,
        track_carbon: bool = False,
        run_label: Optional[str] = None,
        extra: Optional[dict] = None,
    ):
        self.output_path = output_path
        self.track_carbon = track_carbon
        self.run_label = run_label or os.path.basename(os.path.dirname(output_path))
        self.extra = extra or {}
        self.start_time = None
        self.carbon_tracker = None
        self.failed = False
        self.error = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        self.start_time = time.time()
        if self.track_carbon:
            try:
                from codecarbon import EmissionsTracker
                self.carbon_tracker = EmissionsTracker(
                    project_name=self.run_label, save_to_file=False, log_level="error"
                )
                self.carbon_tracker.start()
            except Exception as e:
                self.carbon_tracker = None
                self.carbon_error = str(e)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        wall = time.time() - self.start_time
        if exc_type is not None:
            self.failed = True
            self.error = f"{exc_type.__name__}: {exc_val}"
        carbon_kg = None
        if self.carbon_tracker is not None:
            try:
                carbon_kg = self.carbon_tracker.stop()
            except Exception as e:
                self.carbon_error = str(e)

        hardware = detect_hardware()
        gpu_count = hardware.get("gpu_count", 0)
        gpu_hours = (wall / 3600.0) * max(gpu_count, 1)

        record = {
            "run_label": self.run_label,
            "wall_clock_seconds": wall,
            "wall_clock_hours": wall / 3600.0,
            "gpu_hours": gpu_hours,
            "hardware": hardware,
            "carbon_kgco2eq": carbon_kg,
            "failed": self.failed,
            "error": self.error,
            "extra": self.extra,
        }
        with open(self.output_path, "w") as f:
            json.dump(record, f, indent=2, default=str)
        # Don't swallow exceptions.
        return False


@contextmanager
def record_compute(output_path: str, **kwargs):
    """Functional alias for ComputeRecorder."""
    cr = ComputeRecorder(output_path, **kwargs)
    with cr:
        yield cr
