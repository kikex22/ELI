"""Informacion del equipo (solo lectura). Lo que no se puede leer queda como None."""
from __future__ import annotations

import os
import platform
import subprocess
import sys

_cache: dict | None = None


def _nvidia(index: int = 0) -> dict:
    fields = ["name", "driver_version", "compute_cap", "memory.total"]
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", str(index), f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return {}
    parts = [p.strip() for p in out.split(",")]
    if len(parts) != len(fields):
        return {}

    def num(text: str) -> float | None:
        try:
            return float(text)
        except ValueError:
            return None

    return {
        "name": parts[0] or None,
        "driver": parts[1] or None,
        "compute_cap": num(parts[2]),
        "vram_total_mib": num(parts[3]),
    }


def _cpu_model() -> str | None:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _ram_total_mib() -> float | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except (OSError, ValueError):
        pass
    return None


def hardware_info(refresh: bool = False) -> dict:
    global _cache
    if _cache is None or refresh:
        _cache = {
            "gpu": _nvidia(),
            "cpu": {"model": _cpu_model(), "threads": os.cpu_count()},
            "ram_total_mib": _ram_total_mib(),
            "os": platform.platform(),
            "python": sys.version.split()[0],
        }
    return _cache
