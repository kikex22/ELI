"""Ejecucion aislada de codigo generado por un modelo.

Usa bubblewrap (`bwrap`): sin red, sin ver el /home ni el resto del disco, /usr
de solo lectura, /tmp efimero, entorno vacio y limites de CPU, memoria y
archivos. Si bwrap no esta disponible NO se ejecuta nada: la prueba queda como
no calificada. Aislamiento contra errores y accidentes del modelo; no pretende
ser una frontera de seguridad contra codigo deliberadamente hostil.
"""
from __future__ import annotations

import json
import os
import resource
import secrets
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

MAX_OUTPUT = 16 * 1024

HARNESS = r'''
import json, os, signal, sys

TOKEN = os.environ["ELI_TOKEN"]
spec = json.load(open("/work/cases.json"))
out = {"error": None, "cases": []}

def done():
    sys.stdout.flush()
    print("\n" + TOKEN + json.dumps(out), flush=True)

class CaseTimeout(Exception):
    pass

def on_alarm(signum, frame):
    raise CaseTimeout()

signal.signal(signal.SIGALRM, on_alarm)
ns = {"__name__": "candidate"}
try:
    exec(compile(open("/work/candidate.py").read(), "candidate.py", "exec"), ns)
    fn = ns.get(spec["function"])
    if not callable(fn):
        out["error"] = "el codigo no define la funcion " + spec["function"]
except BaseException as exc:
    out["error"] = "error al cargar el codigo: " + type(exc).__name__ + ": " + str(exc)[:200]
    fn = None

def norm(value):
    return json.loads(json.dumps(value))

if fn is not None:
    for case in spec["cases"]:
        item = {"args": case["args"], "expected": case["expected"], "ok": False, "got": None, "error": None}
        try:
            signal.setitimer(signal.ITIMER_REAL, spec["case_timeout_s"])
            got = norm(fn(*case["args"]))
            signal.setitimer(signal.ITIMER_REAL, 0)
            item["got"] = got
            item["ok"] = got == case["expected"]
        except CaseTimeout:
            item["error"] = "tiempo agotado"
        except BaseException as exc:
            signal.setitimer(signal.ITIMER_REAL, 0)
            item["error"] = type(exc).__name__ + ": " + str(exc)[:200]
        out["cases"].append(item)
done()
'''

_available: str | None | bool = False  # False = sin comprobar


def _system_binds() -> list[str]:
    """Enlaces de solo lectura del sistema base, solo con las rutas que existen."""
    args: list[str] = ["--ro-bind", "/usr", "/usr"]
    for name in ("lib", "lib64", "bin", "sbin"):
        path = Path("/" + name)
        if path.is_symlink():
            args += ["--symlink", os.readlink(path), "/" + name]
        elif path.is_dir():
            args += ["--ro-bind", str(path), "/" + name]
    return args


def _base_command(workdir: str) -> list[str]:
    return [
        "bwrap", "--unshare-all", "--die-with-parent", "--new-session",
        *_system_binds(),
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
        "--ro-bind", workdir, "/work", "--chdir", "/work",
        "--clearenv", "--setenv", "PATH", "/usr/bin:/bin",
        "--setenv", "HOME", "/tmp", "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
    ]


def available() -> str | None:
    """'bwrap' si se puede aislar de verdad (se comprueba una vez), o None."""
    global _available
    if _available is False:
        _available = None
        if shutil.which("bwrap"):
            with tempfile.TemporaryDirectory() as tmp:
                try:
                    probe = subprocess.run(
                        _base_command(tmp) + ["/usr/bin/true"], capture_output=True, timeout=10
                    )
                    if probe.returncode == 0:
                        _available = "bwrap"
                except (OSError, subprocess.SubprocessError):
                    pass
    return _available  # type: ignore[return-value]


def info() -> dict:
    kind = available()
    return {
        "available": kind is not None,
        "kind": kind,
        "isolation": "sin red, sin acceso al /home, /usr de solo lectura, entorno vacio" if kind else None,
        "limits": {"memory_mib": 512, "cpu_s": "timeout + 2", "file_mib": 1},
    }


@dataclass
class SandboxResult:
    ran: bool
    error: str | None = None
    timed_out: bool = False
    passed: int = 0
    total: int = 0
    cases: list[dict] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""


def _limits(timeout_s: float, memory_mib: int):
    def apply() -> None:
        cpu = int(timeout_s) + 2
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        mem = memory_mib * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    return apply


def run_function_tests(
    code: str, function: str, cases: list[dict], timeout_s: float = 10.0, case_timeout_s: float = 2.0, memory_mib: int = 512
) -> SandboxResult:
    """Carga `code` en el sandbox, llama a `function(*args)` por caso y compara con `expected` (JSON)."""
    if available() is None:
        return SandboxResult(ran=False, error="No hay aislamiento disponible (bubblewrap): el codigo no se ejecuto.")
    token = "ELI-RESULT-" + secrets.token_hex(8) + ":"
    with tempfile.TemporaryDirectory(prefix="eli-sbx-") as workdir:
        Path(workdir, "candidate.py").write_text(code, encoding="utf-8")
        Path(workdir, "main.py").write_text(HARNESS, encoding="utf-8")
        Path(workdir, "cases.json").write_text(
            json.dumps({"function": function, "cases": cases, "case_timeout_s": case_timeout_s}), encoding="utf-8"
        )
        command = _base_command(workdir) + ["--setenv", "ELI_TOKEN", token, "/usr/bin/python3", "-I", "-S", "/work/main.py"]
        try:
            proc = subprocess.run(
                command, capture_output=True, timeout=timeout_s, preexec_fn=_limits(timeout_s, memory_mib),
                start_new_session=True,
            )
        except subprocess.TimeoutExpired as exc:
            return SandboxResult(
                ran=True, timed_out=True, total=len(cases), error=f"Tiempo maximo de {timeout_s:g} s superado.",
                stdout=(exc.stdout or b"")[:MAX_OUTPUT].decode(errors="replace"),
            )
        except OSError as exc:
            return SandboxResult(ran=False, error=f"No se pudo iniciar el sandbox: {exc}")

    stdout = proc.stdout[:MAX_OUTPUT].decode(errors="replace")
    stderr = proc.stderr[:MAX_OUTPUT].decode(errors="replace")
    line = next((ln for ln in reversed(stdout.splitlines()) if ln.startswith(token)), None)
    if line is None:
        reason = "el proceso termino sin entregar resultados"
        if proc.returncode < 0:
            reason += f" (senal {-proc.returncode}: limite de memoria o CPU)"
        return SandboxResult(ran=True, total=len(cases), error=reason, stdout=stdout, stderr=stderr)
    try:
        data = json.loads(line[len(token):])
    except ValueError:
        return SandboxResult(ran=True, total=len(cases), error="resultado ilegible", stdout=stdout, stderr=stderr)
    results = data.get("cases", [])
    return SandboxResult(
        ran=True,
        error=data.get("error"),
        passed=sum(1 for c in results if c.get("ok")),
        total=len(cases),
        cases=results,
        stdout=stdout.replace(line, "")[:MAX_OUTPUT],
        stderr=stderr,
    )
