# Plan: evaluacion de modelos de embeddings en ELI (traspaso)

Estado a 15-oct-2026. Rama `feat/eli-v0.2`, **sin commit** (el usuario decide cuando). El usuario hace las pruebas con modelos
reales; no lanzar evaluaciones reales (una llamada minima a `/api/embed` para ver su forma si hace falta).

## Decisiones del usuario
- Pestana propia **Embeddings** (no dentro de Benchmark).
- Calidad con **banco incluido (ES/EN) y tambien con sus propios documentos** (carpeta).
- Solo hay instalado `nomic-embed-text` (768 dims, contexto 2048). Otros (bge-m3, mxbai-embed-large) los instala el usuario; ELI no descarga.

## Que se mide (todo verificable, sin modelo juez)
1. **Recuperacion** (banco: 30 documentos, 40 consultas: 30 monolingues + 10 entre idiomas): Recall@1/3/5 y MRR.
   Ranking *pesimista* con empates (un modelo que da el mismo vector a todo no gana por azar).
2. **Similitud**: 18 tripletes (es/en/cross); acierto si sim(ancla, parafrasis) > sim(ancla, negativo).
3. **Estabilidad**: mismo texto repetido y dentro de un lote -> coseno minimo.
4. **Rendimiento**: latencia (texto corto/medio/largo, mediana de N) y embeddings/s con lotes 1/8/32; VRAM y energia con GIA.
5. **Corpus propio**: consulta = titulo de seccion Markdown; correcto = fragmentos de esa seccion sin el titulo
   (+ `eli_queries.jsonl` opcional con `{"query","relevant_files"}`). Huella (sha256) del corpus en cada ejecucion.

## Reglas de diseno
- `truncate=False` en `/api/embed`: un texto mas largo que el contexto es error explicito, no recorte silencioso.
- **Prefijos de tarea** (nomic: `search_query: `/`search_document: `, e5, mxbai, snowflake) opcionales; se registran los usados.
  Por defecto sin prefijos. Reglas versionadas en `eli/data/embedding_prefixes.json`.
- Mismas condiciones para todos: comparar solo modelos con la misma suite (hash), mismo modo de prefijos y misma huella de corpus;
  si difieren, avisar y dejar esa metrica en blanco. Clasificar solo con >= 2 modelos con datos validos.
- Ejecucion: un modelo tras otro, bajo el mismo bloqueo de GPU que chat y benchmark; calentamiento aparte; cancelacion entre
  llamadas; confirmacion antes de descargar otros modelos de la VRAM; descargar cada modelo al terminar y esperar a que Ollama confirme.
- Persistencia en `runs/embeddings/<run_id>/` con la misma clase `Store(runs_dir, "embeddings")`: run.json (una vez), status.json,
  results.jsonl (solo se anade). No guardar el contenido de los documentos del usuario, solo rutas, consultas y posiciones.
- Seguridad del corpus: solo .md/.txt/.rst, sin enlaces simbolicos, sin carpetas ocultas ni nombres sensibles, topes de archivos,
  tamano y carpetas, rechazar `/`, `/proc`, `/sys`, `/dev`.

## Hecho (sin probar con pruebas automaticas todavia)
- `eli/ollama.py`: `embed()`; `_json(..., timeout)`.
- `eli/bench/store.py`: `Store(runs_dir, subdir="benchmarks")`.
- `eli/inventory.py`: `inspect_installed(..., extra_stores=())` suma la VRAM observada de las evaluaciones de embeddings.
- `eli/emb/metrics.py` (coseno, rango pesimista, recall/MRR, tripletes, estabilidad).
- `eli/emb/bank.py` + `eli/emb/suites/emb-core-1.0.0.json` (generado con un script; validado al cargar) + prefijos.
- `eli/emb/corpus.py`: probado a mano sobre `~/Documents/CLU/docs` (17 fragmentos, 14 consultas); falta cubrirlo con tests.

## Pendiente, en este orden
1. `eli/emb/runner.py` (`EmbManager`, como `BenchManager`): pasos warmup -> retrieval -> retrieval_user -> similarity -> stability ->
   performance; registros `model_start/warmup/step/model_end`; estado y contadores; GPU con `gpu_summary`.
2. `eli/emb/compare.py`: por modelo, ultima ejecucion completa; condiciones iguales; rankings (recuperacion, entre idiomas,
   similitud, velocidad, VRAM) y frentes de Pareto con `bench.compare.pareto_front`.
3. Servidor: `GET /api/emb/suite`, `/api/emb/models` (capability `embedding`), `POST /api/emb/corpus-preview`, `POST /api/emb/start`,
   `GET /api/emb/status`, `POST /api/emb/cancel`, `GET /api/emb/history[/<id>[/export?format=json|csv]]`, `GET /api/emb/compare`.
   Anadir el JS a `WEB_FILES` en `server.py` y el boton/seccion en `index.html`.
4. UI `eli/web/embeddings.js` (seleccion de modelos, banco/carpeta/prefijos, progreso y cancelacion, tabla, graficas con
   `barChart`/`scatterChart`, detalle de consultas falladas, historial y exportacion). Usar `put()` y `h()` (nunca `append(null)`).
5. Tests: metricas, banco, corpus (carpeta temporal: titulos, ocultos, symlinks, sensibles, duplicados, manual, topes), runner con un
   `/api/embed` falso (vectores por hash de palabras; modelo degenerado de vector constante), cancelacion, errores, prefijos
   registrados, endpoints, compatibilidad de la UI (los tests existentes ya revisan ids y rutas del JS).
6. README: seccion Embeddings y limites. Fakes en `tests/fakes.py` (anadir `/api/embed`).

## Objetivo final: elegir modelos para convertir a TensorRT (anadido 15-oct-2026)
La evaluacion en Ollama es el *filtro*; la conversion es un paso posterior y aparte. Para que el filtro sirva hay que registrar
ademas, por modelo (campo `trt_fit` en la ficha, informativo, sin ejecutar nada):
- **Arquitectura**: encoder (BERT/XLM-R: bge-m3, e5, nomic-bert, mxbai) vs decoder (Qwen3-Embedding, gte-Qwen2). Cambia la ruta:
  - encoder -> ONNX + TensorRT clasico (`trtexec` ya esta en el equipo, `/usr/bin/trtexec`); TRT-LLM solo trae ejemplo de BERT/RoBERTa.
  - decoder -> TRT-LLM (checkpoint HF -> `convert_checkpoint` -> `trtllm-build`), con pooling de ultimo token hecho fuera del motor.
- **Pooling y normalizacion** (cls / mean / last-token; L2) y **dimension**/Matryoshka: hay que replicarlos tras la conversion.
- **Checkpoint original**: TRT no carga GGUF de Ollama (igual que con los LLM); hace falta el repo HF -> guardar `hf_repo` por modelo.
- **Referencia para validar**: guardar en cada ejecucion los vectores de un conjunto fijo de textos (hash en `run.json`) para que,
  tras convertir, se compare coseno(ollama, trt) >= 0.99 y se repitan recuperacion y rendimiento con el mismo banco.
- Tabla de compatibilidad en `eli/data/trtllm_rules.json` (ya existe para LLM): extender con familias de embeddings.
Fase 2 (despues de tener la pestana): adaptador `eli/emb/backends/trt.py` con la misma interfaz `embed(texts)`; asi el runner y
el comparador no cambian y se ve Ollama vs TRT en la misma tabla.

## Trampas ya conocidas
- En este shell `cd` esta redirigido: usar `builtin cd /home/adjor/Documents/ELI`.
- `pkill -f` con un patron que aparece en el propio comando mata el shell: anclar el patron (`'^python3 b18.py ...'`).
- Capturas del panel del navegador pueden ir un paso atrasadas; confirmar el estado con JS.
- Comparar usa el razonamiento *solicitado*, no el efectivo; preferir ejecuciones completas sobre parciales.
- Tests: `PYTHONPATH=tests python3 -W error::ResourceWarning -m unittest discover -s tests` (192 pasan hoy).
