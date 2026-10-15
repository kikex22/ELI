"""Persistencia de ejecuciones. Nunca sobrescribe ni borra datos historicos.

runs/benchmarks/<run_id>/
    run.json        configuracion, suite, modelos, hardware (se escribe una vez)
    status.json     estado y contadores (lo unico que cambia; escritura atomica)
    results.jsonl   un registro por linea, solo se anade (model_start, warmup, result, model_end)

Los chats se guardan como archivos sueltos en runs/ con nombre unico.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import secrets
import threading
from datetime import datetime
from pathlib import Path
from typing import Iterator

RUN_ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")

CSV_COLUMNS = [
    "run_id", "model", "test_id", "instance_id", "category", "difficulty", "repetition", "num_ctx_requested",
    "prompt_tokens_processed", "outcome", "reason", "gen_status", "gen_tokens", "decode_tok_s", "ttft_s", "total_s",
    "load_s", "cold", "vram_peak_mib", "util_mean", "power_mean_w", "temp_max_c", "energy_net_wh", "wh_per_token",
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def atomic_write(path: Path, text: str) -> None:
    """Escribe a un temporal y lo mueve encima: nunca queda un archivo a medias."""
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_unique(directory: Path, stem: str, suffix: str, text: str) -> Path:
    """Crea `stem+suffix` (o stem-2, stem-3...) sin pisar nunca un archivo existente."""
    directory.mkdir(parents=True, exist_ok=True)
    tmp = directory / f".{secrets.token_hex(6)}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    try:
        for n in range(1, 1000):
            final = directory / (f"{stem}{suffix}" if n == 1 else f"{stem}-{n}{suffix}")
            try:
                os.link(tmp, final)  # falla si ya existe: es la proteccion contra colisiones
                return final
            except FileExistsError:
                continue
        raise OSError(f"no se pudo crear un nombre unico para {stem}")
    finally:
        tmp.unlink(missing_ok=True)


def _csv_safe(value):
    """Evita que una hoja de calculo interprete texto como formula."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@"):
        return "'" + value
    return value


class RunHandle:
    def __init__(self, run_id: str, directory: Path):
        self.run_id = run_id
        self.dir = directory
        self._lock = threading.Lock()
        self._status: dict = json.loads((directory / "status.json").read_text(encoding="utf-8"))

    def append(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock, open(self.dir / "results.jsonl", "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    def set_status(self, **fields) -> dict:
        with self._lock:
            self._status.update(fields)
            atomic_write(self.dir / "status.json", json.dumps(self._status, ensure_ascii=False, indent=1))
            return dict(self._status)

    def status(self) -> dict:
        with self._lock:
            return dict(self._status)


class Store:
    def __init__(self, runs_dir: Path, subdir: str = "benchmarks"):
        self.runs_dir = Path(runs_dir)
        self.root = self.runs_dir / subdir

    # --- ejecuciones ---------------------------------------------------

    def create_run(self, run_doc: dict) -> RunHandle:
        self.root.mkdir(parents=True, exist_ok=True)
        for _ in range(50):
            run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"
            directory = self.root / run_id
            try:
                os.mkdir(directory)  # exclusivo: dos ejecuciones nunca comparten carpeta
                break
            except FileExistsError:
                continue
        else:
            raise OSError("no se pudo crear una carpeta unica para la ejecucion")
        doc = {"run_id": run_id, "created": now_iso(), **run_doc}
        (directory / "run.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        (directory / "results.jsonl").touch()
        status = {"state": "running", "started": doc["created"], "finished": None, "done": 0,
                  "total": run_doc.get("planned", {}).get("total", 0),
                  "counts": {"passed": 0, "failed": 0, "ungraded": 0, "error": 0, "timeout": 0, "cancelled": 0},
                  "models": {}, "message": None}
        atomic_write(directory / "status.json", json.dumps(status, ensure_ascii=False, indent=1))
        return RunHandle(run_id, directory)

    def _dir(self, run_id: str) -> Path | None:
        if not isinstance(run_id, str) or not RUN_ID_RE.match(run_id):
            return None  # tambien impide rutas como ../
        directory = self.root / run_id
        return directory if directory.is_dir() else None

    @staticmethod
    def _read(path: Path) -> dict | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def list_runs(self) -> list[dict]:
        """Resumenes ordenados del mas reciente al mas antiguo. Solo lee run.json y status.json."""
        rows = []
        if not self.root.is_dir():
            return rows
        for directory in self.root.iterdir():
            if not RUN_ID_RE.match(directory.name):
                continue
            run = self._read(directory / "run.json")
            if run is None:
                continue
            status = self._read(directory / "status.json") or {"state": "unknown"}
            # el id solo tiene resolucion de segundos: la hora exacta de creacion desempata
            order = (run.get("created") or "", (directory / "run.json").stat().st_mtime_ns)
            rows.append((order, self._summary(run, status)))
        return [row for _, row in sorted(rows, key=lambda r: r[0], reverse=True)]

    @staticmethod
    def _summary(run: dict, status: dict) -> dict:
        return {
            "run_id": run["run_id"], "created": run.get("created"), "state": status.get("state"),
            "finished": status.get("finished"), "models": [m.get("name") for m in run.get("models", [])],
            "suite": run.get("suite", {}), "mode": run.get("config", {}).get("mode"),
            "repetitions": run.get("config", {}).get("repetitions"),
            "done": status.get("done", 0), "total": status.get("total", 0), "counts": status.get("counts", {}),
            "message": status.get("message"),
        }

    def load_run(self, run_id: str, with_results: bool = True) -> dict | None:
        directory = self._dir(run_id)
        if directory is None:
            return None
        run = self._read(directory / "run.json")
        if run is None:
            return None
        status = self._read(directory / "status.json") or {"state": "unknown"}
        doc = {"run": run, "status": status, "summary": self._summary(run, status)}
        if with_results:
            doc["results"], doc["events"] = [], []
            for record in self.iter_records(run_id):
                (doc["results"] if record.get("type") == "result" else doc["events"]).append(record)
        return doc

    def iter_records(self, run_id: str) -> Iterator[dict]:
        directory = self._dir(run_id)
        if directory is None:
            return
        try:
            with open(directory / "results.jsonl", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except ValueError:
                        continue  # linea a medias de una ejecucion interrumpida
        except OSError:
            return

    def mark_interrupted(self, active: set[str]) -> list[str]:
        """Las ejecuciones que quedaron 'running' sin un proceso vivo pasan a 'interrupted'."""
        fixed = []
        if not self.root.is_dir():
            return fixed
        for directory in self.root.iterdir():
            if not RUN_ID_RE.match(directory.name) or directory.name in active:
                continue
            status = self._read(directory / "status.json")
            if status and status.get("state") == "running":
                status.update(state="interrupted", finished=now_iso(),
                              message="La ejecucion se interrumpio (ELI se cerro o fallo antes de terminar).")
                atomic_write(directory / "status.json", json.dumps(status, ensure_ascii=False, indent=1))
                fixed.append(directory.name)
        return fixed

    def latest_models(self) -> list[dict]:
        """Por modelo, su evaluacion mas reciente y los metadatos guardados (sirve aunque el modelo ya no este instalado)."""
        seen: dict[str, dict] = {}
        for row in self.list_runs():  # del mas reciente al mas antiguo
            run = self._read(self.root / row["run_id"] / "run.json") or {}
            for meta in run.get("models", []):
                name = meta.get("name")
                if name and name not in seen:
                    seen[name] = {"name": name, "run_id": row["run_id"], "created": row.get("created"), "state": row.get("state"),
                                  "has_results": bool(row.get("done")), "meta": meta}
        return sorted(seen.values(), key=lambda m: m["name"])

    def observed_vram(self) -> dict[str, dict]:
        """VRAM maxima observada por modelo en evaluaciones (de status.json; incluye modelos ya borrados)."""
        observed: dict[str, dict] = {}
        for row in self.list_runs():
            status = self._read(self.root / row["run_id"] / "status.json") or {}
            for name, info in (status.get("models") or {}).items():
                cur = observed.setdefault(name, {"runs": 0, "vram_used_mib": None, "vram_gpu_peak_mib": None, "model_vram_bytes": None})
                cur["runs"] += 1
                for key in ("vram_used_mib", "vram_gpu_peak_mib", "model_vram_bytes"):
                    if info.get(key) is not None and (cur[key] is None or info[key] > cur[key]):
                        cur[key] = info[key]
        return observed

    # --- exportacion ---------------------------------------------------

    def export_json(self, run_id: str) -> str | None:
        doc = self.load_run(run_id)
        if doc is None:
            return None
        return json.dumps({"export": "eli-run", "format_version": 1, **doc}, ensure_ascii=False, indent=1)

    def export_csv(self, run_id: str) -> str | None:
        doc = self.load_run(run_id)
        if doc is None:
            return None
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for r in doc["results"]:
            m = r.get("metrics") or {}
            gpu = (m.get("gpu") or {}).get("request") or {}
            decode = (m.get("gpu") or {}).get("decode") or {}
            row = {
                "run_id": run_id, "model": r.get("model"), "test_id": r.get("test_id"), "instance_id": r.get("instance_id"),
                "category": r.get("category"), "difficulty": r.get("difficulty"), "repetition": r.get("repetition"),
                "num_ctx_requested": r.get("num_ctx_requested"), "prompt_tokens_processed": m.get("prompt_tokens"),
                "outcome": r.get("outcome"), "reason": (r.get("grade") or {}).get("reason") or r.get("error") or "",
                "gen_status": r.get("gen_status"), "gen_tokens": m.get("gen_tokens"), "decode_tok_s": m.get("decode_tok_s"),
                "ttft_s": m.get("ttft_s"), "total_s": m.get("total_s"), "load_s": m.get("load_s"), "cold": r.get("cold"),
                "vram_peak_mib": gpu.get("vram_peak_mib"), "util_mean": gpu.get("util_mean"),
                "power_mean_w": gpu.get("power_mean_w"), "temp_max_c": gpu.get("temp_max_c"),
                "energy_net_wh": decode.get("energy_net_wh"), "wh_per_token": m.get("wh_per_token"),
            }
            writer.writerow({k: _csv_safe(v) for k, v in row.items()})
        return out.getvalue()

    # --- chats ---------------------------------------------------------

    def save_chat(self, record: dict, model: str, t_start: float) -> Path | None:
        stamp = datetime.fromtimestamp(t_start).strftime("%Y%m%d-%H%M%S")
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model)[:60]
        try:
            return write_unique(self.runs_dir, f"{stamp}_{safe}_chat", ".json", json.dumps(record, ensure_ascii=False, indent=1))
        except OSError:
            return None

    def list_chats(self, limit: int = 100) -> list[dict]:
        rows = []
        if not self.runs_dir.is_dir():
            return rows
        for path in sorted(self.runs_dir.glob("*_chat*.json"), reverse=True)[:limit]:
            data = self._read(path)
            if not data:
                continue
            m = data.get("metrics") or {}
            rows.append({"file": path.name, "started": data.get("started"), "model": data.get("model"),
                         "status": data.get("status", "ok"), "gen_tokens": m.get("gen_tokens"),
                         "decode_tok_s": m.get("decode_tok_s")})
        return rows
