"""GIA (GPU & Inference Analyzer): muestreo de la GPU con nvidia-smi.

Sin dependencias: un proceso `nvidia-smi -lms` de larga duracion produce una
lectura cada INTERVAL_MS; un hilo la guarda en un historial y la reparte a los
clientes suscritos. Otro hilo consulta la memoria por proceso cada segundo.
"""
from __future__ import annotations

import queue
import statistics
import subprocess
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass

FIELDS = [
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw",
    "power.limit",
    "temperature.gpu",
    "clocks.sm",
    "clocks.mem",
    "pstate",
    "clocks_event_reasons.active",
]
INTERVAL_MS = 250

# Bits de clocks_event_reasons que significan que la GPU esta frenada.
# 0x1 (GPU inactiva), 0x2 y 0x100 (relojes fijados) no son frenados.
THROTTLE_BITS = {
    0x4: "limite de potencia",
    0x8: "frenado por hardware",
    0x20: "limite termico (software)",
    0x40: "limite termico (hardware)",
    0x80: "freno de potencia (hardware)",
}


@dataclass(frozen=True)
class Sample:
    ts: float  # epoch, segundos
    util: float | None  # % de la GPU
    mem_util: float | None  # % del controlador de memoria
    mem_used: float | None  # MiB
    mem_total: float | None  # MiB
    power: float | None  # W
    power_limit: float | None  # W
    temp: float | None  # C
    sm_clock: float | None  # MHz
    mem_clock: float | None  # MHz
    pstate: str | None
    throttle: tuple[str, ...]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["throttle"] = list(self.throttle)
        return data


def _num(text: str) -> float | None:
    text = text.strip()
    if not text or text.startswith("[") or text.lower() in {"n/a", "not supported"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def decode_throttle(text: str) -> tuple[str, ...]:
    try:
        value = int(text.strip(), 16)
    except ValueError:
        return ()
    return tuple(name for bit, name in THROTTLE_BITS.items() if value & bit)


def parse_line(line: str, ts: float) -> Sample | None:
    parts = [p.strip() for p in line.strip().split(",")]
    if len(parts) != len(FIELDS):
        return None
    pstate = parts[9] if parts[9] and not parts[9].startswith("[") else None
    return Sample(
        ts=ts,
        util=_num(parts[0]),
        mem_util=_num(parts[1]),
        mem_used=_num(parts[2]),
        mem_total=_num(parts[3]),
        power=_num(parts[4]),
        power_limit=_num(parts[5]),
        temp=_num(parts[6]),
        sm_clock=_num(parts[7]),
        mem_clock=_num(parts[8]),
        pstate=pstate,
        throttle=decode_throttle(parts[10]),
    )


def parse_processes(text: str) -> list[dict]:
    rows = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3 or not parts[0].isdigit():
            continue
        name = parts[1].rsplit("/", 1)[-1]
        rows.append({"pid": int(parts[0]), "name": name, "mem_mib": _num(parts[2])})
    return rows


def window_stats(
    samples: list[Sample],
    t0: float,
    t1: float,
    baseline_power: float | None = None,
) -> dict | None:
    """Resumen de la GPU entre t0 y t1 (segundos epoch).

    La energia es potencia media x duracion. Es la potencia de TODA la GPU, no
    solo la del modelo; `energy_net_wh` resta el consumo base de reposo.
    """
    if not samples or t1 <= t0:
        return None

    def values(name: str) -> list[float]:
        return [v for s in samples if (v := getattr(s, name)) is not None]

    duration = t1 - t0
    util, power, temp, mem, clocks = values("util"), values("power"), values("temp"), values("mem_used"), values("sm_clock")
    stats: dict = {
        "samples": len(samples),
        "duration_s": round(duration, 3),
        "vram_start_mib": mem[0] if mem else None,
        "vram_end_mib": mem[-1] if mem else None,
        "vram_peak_mib": max(mem) if mem else None,
        "util_mean": round(statistics.fmean(util), 1) if util else None,
        "util_max": max(util) if util else None,
        "power_mean_w": round(statistics.fmean(power), 1) if power else None,
        "power_peak_w": max(power) if power else None,
        "temp_max_c": max(temp) if temp else None,
        "sm_clock_mean_mhz": round(statistics.fmean(clocks)) if clocks else None,
        "throttle": sorted({r for s in samples for r in s.throttle}),
    }
    if power:
        energy = statistics.fmean(power) * duration / 3600
        stats["energy_wh"] = round(energy, 5)
        if baseline_power is not None:
            stats["baseline_power_w"] = round(baseline_power, 1)
            stats["energy_net_wh"] = round(max(0.0, energy - baseline_power * duration / 3600), 5)
    return stats


IDLE_UTIL_MAX = 15.0  # % de uso por debajo del cual se considera reposo
IDLE_MIN_SAMPLES = 8  # al menos 2 s de lecturas en reposo


def idle_power(samples: list[Sample]) -> float | None:
    """Consumo en reposo: mediana de la potencia con la GPU casi inactiva.

    No sirve mirar solo los segundos previos a una peticion: justo despues de
    otra respuesta la GPU sigue con potencia alta y el reposo saldria inflado.
    """
    power = [s.power for s in samples if s.power is not None and s.util is not None and s.util <= IDLE_UTIL_MAX]
    return statistics.median(power) if len(power) >= IDLE_MIN_SAMPLES else None


class GPUMonitor:
    def __init__(self, interval_ms: int = INTERVAL_MS, history: int = 2400, command: list[str] | None = None):
        self.interval_ms = interval_ms
        self.command = command or [
            "nvidia-smi",
            f"--query-gpu={','.join(FIELDS)}",
            "--format=csv,noheader,nounits",
            "-lms",
            str(interval_ms),
        ]
        self.available = False
        self.error: str | None = None
        self._samples: deque[Sample] = deque(maxlen=history)
        self._subscribers: list[queue.Queue] = []
        self._processes: list[dict] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None
        self._threads: list[threading.Thread] = []

    # --- ciclo de vida -------------------------------------------------

    def start(self) -> None:
        for target in (self._run, self._run_processes):
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        self._stop.set()
        proc = self._proc
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._proc = subprocess.Popen(
                    self.command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1
                )
            except FileNotFoundError:
                self.available, self.error = False, "nvidia-smi no esta disponible en esta maquina"
                self._stop.wait(30)
                continue
            for line in self._proc.stdout:
                sample = parse_line(line, time.time())
                if sample is None:
                    continue
                self.available, self.error = True, None
                self._publish(sample)
            self._proc.stdout.close()
            self._proc.wait()
            self.available = False
            if not self._stop.is_set():
                self.error = "nvidia-smi se detuvo; reintentando"
                self._stop.wait(5)

    def _run_processes(self) -> None:
        command = [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            try:
                out = subprocess.run(command, capture_output=True, text=True, timeout=5).stdout
                self._processes = parse_processes(out)
            except (OSError, subprocess.SubprocessError):
                self._processes = []
            self._stop.wait(1)

    def _publish(self, sample: Sample) -> None:
        with self._lock:
            self._samples.append(sample)
            subscribers = list(self._subscribers)
        for q in subscribers:
            try:
                q.put_nowait(sample)
            except queue.Full:
                pass

    # --- consultas -----------------------------------------------------

    def latest(self) -> Sample | None:
        with self._lock:
            return self._samples[-1] if self._samples else None

    def window(self, t0: float, t1: float) -> list[Sample]:
        with self._lock:
            return [s for s in self._samples if t0 <= s.ts <= t1]

    def history(self, seconds: float) -> list[Sample]:
        return self.window(time.time() - seconds, float("inf"))

    def processes(self) -> list[dict]:
        return list(self._processes)

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=200)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)
