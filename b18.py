#!/usr/bin/env python3
"""ELI (Evaluation Lab for Inference): abre el dashboard en el navegador."""
from __future__ import annotations

import argparse
import os
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from eli.gpu import GPUMonitor  # noqa: E402
from eli.ollama import DEFAULT_URL, Ollama  # noqa: E402
from eli.server import App, make_server  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="ELI: dashboard local para evaluar modelos.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("ELI_PORT", "8766")))
    parser.add_argument("--ollama-url", default=os.environ.get("ELI_OLLAMA_URL", DEFAULT_URL))
    parser.add_argument("--no-browser", action="store_true", help="no abrir el navegador")
    parser.add_argument("--runs-dir", type=Path, default=Path(os.environ.get("ELI_RUNS_DIR", ROOT / "runs")),
                        help="carpeta donde se guardan chats y ejecuciones (por defecto ./runs)")
    args = parser.parse_args()

    gpu = GPUMonitor()
    gpu.start()
    app = App(Ollama(args.ollama_url), gpu, args.runs_dir, args.port)
    try:
        server = make_server(app, args.port)
    except OSError as exc:
        print(f"No se pudo abrir el puerto {args.port}: {exc}", file=sys.stderr)
        gpu.stop()
        return 1
    url = f"http://127.0.0.1:{args.port}"
    print(f"ELI listo en {url}  (Ctrl+C para salir)")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nCerrando ELI.")
    finally:
        server.server_close()
        gpu.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
