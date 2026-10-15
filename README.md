# ELI v0.2 - Evaluation Lab for Inference

Aplicacion local para **inspeccionar, evaluar, comparar y administrar** modelos de Ollama en tu propia GPU.
Es independiente de CLU y no importa nada de el. No es un agente: no ejecuta herramientas ni toma decisiones por ti.
Plan por partes en [docs/roadmap_eli.md](docs/roadmap_eli.md).

## Uso

```bash
B18                                   # alias en ~/.bashrc; abre http://127.0.0.1:8766
python3 b18.py --no-browser --port 8766
python3 b18.py --runs-dir /otra/carpeta   # donde se guardan chats y ejecuciones (por defecto ./runs)
```

Variables: `ELI_PORT` (8766), `ELI_OLLAMA_URL` (`http://127.0.0.1:11434`), `ELI_RUNS_DIR`.
Solo escucha en `127.0.0.1` y rechaza otros `Host`/`Origin`. Sin dependencias: Python 3.12 y la libreria estandar.

## Secciones

| Seccion | Que hace |
|---|---|
| **Dashboard** | Sistema (GPU, CPU, RAM, versiones, sandbox), modelos, evaluacion en curso y ejecuciones recientes. |
| **Models** | Inventario con **Model Inspector** (formato, cuantizacion y arquitectura por separado; lo que Ollama no informa sale como *desconocido*), analisis informativo de **TensorRT-LLM** y administracion: descargar de VRAM o eliminar. |
| **Benchmark** | Eliges uno o varios modelos, rapido/completo, categorias, repeticiones, parametros y tamanos de contexto. Progreso real y cancelacion. |
| **Compare** | Tablas, clasificaciones, graficas y filtros por tamano. Solo se comparan las pruebas hechas por todos en las mismas condiciones. |
| **History** | Ejecuciones anteriores con prompts, respuestas, validaciones y metricas; exportacion JSON y CSV. |
| **Chat** | El chat con metricas de la v0.1, para pruebas manuales. |

El panel **GIA** (VRAM, uso, consumo, temperatura, relojes, frenado y procesos) esta siempre visible arriba.

## El benchmark

- Banco versionado: `eli/bench/suites/eli-core-1.0.0.json` (66 pruebas, 28 en el modo rapido; hash SHA-256 guardado en cada ejecucion).
  Categorias: razonamiento logico, matematicas, programacion, comprension de textos, instrucciones, formatos estructurados,
  espanol/ingles, generacion larga (velocidad) y contexto largo. Cada prueba lleva id, categoria, dificultad, prompt, criterio
  verificable, parametros y version.
- **Calificacion solo con criterios verificables** (respuesta exacta, numero, JSON con esquema, restricciones medibles, o codigo
  ejecutado en un sandbox). Nunca un modelo como juez. Lo que no se puede calificar con fiabilidad queda **sin calificar**
  (respuesta cortada por el limite de tokens, ambigua, sin sandbox...): no cuenta ni como acierto ni como fallo.
- **Codigo generado:** se ejecuta con bubblewrap (sin red, sin ver tu `/home`, `/usr` de solo lectura, entorno vacio, limites de
  CPU, memoria y archivos). Si no esta disponible no se ejecuta nada y la prueba queda sin calificar. Es aislamiento contra
  errores del modelo, no una frontera de seguridad contra codigo deliberadamente hostil.
- Modelos **uno tras otro**, cada prueba en una **conversacion independiente**, mismas pruebas y condiciones para todos,
  calentamiento aparte (la carga se mide por separado), tiempo maximo por prueba, cancelacion, y corte automatico de un modelo
  tras 3 errores seguidos. Antes de empezar pide confirmacion para descargar de la VRAM otros modelos cargados.
- No descarga modelos ni borra nada.

## Como leer las cifras

- Cada metrica indica su origen: *medido por Ollama*, *calculado* de valores de Ollama (tok/s), *reloj de ELI* (primer token),
  *nvidia-smi* (GPU) o *estimacion* (energia).
- **num_ctx** es lo que se pidio; el contexto **comprobado** es el mayor prompt (en tokens procesados) con el que el modelo
  recupero el dato. No son lo mismo.
- **VRAM:** pico de la GPU menos lo que ya habia antes de cargar el modelo. **Energia:** potencia media de toda la GPU por la
  duracion, menos el reposo (mediana de las lecturas con uso <= 15 % de los ultimos 10 min); solo se informa con >= 5 mediciones.
- **tok/s** usa solo respuestas con >= 16 tokens y con el modelo caliente.
- **Calidad** = superadas / calificadas. Se muestran tambien las sin calificar y los errores.
- **No hay un indicador unico.** Se muestran calidad, velocidad, VRAM, tamano y energia, con sus frentes de Pareto. Las
  clasificaciones exigen >= 3 pruebas calificadas por categoria y >= 2 modelos con datos; si no, dicen «datos insuficientes».
  «Mejor eficiencia general» = mas frentes de Pareto, solo entre modelos con >= 60 % de la calidad del mejor.
- **Razonamiento:** por defecto el benchmark lo desactiva (comparacion estable y rapida). Algunos modelos (p. ej. `gemma4:e4b`)
  mezclan su borrador de razonamiento con la respuesta si se deja «por defecto».

## Datos guardados

```
runs/benchmarks/<run_id>/
    run.json        configuracion, suite, metadatos de los modelos, hardware (se escribe una vez)
    status.json     estado y contadores (lo unico que cambia)
    results.jsonl   un registro por linea, solo se anade
runs/<fecha>_<modelo>_chat*.json   chats (nombre unico: nunca se pisan)
```
Los resultados historicos no se alteran ni se borran (ELI no tiene ninguna operacion para ello) y se pueden consultar aunque el
modelo ya no este instalado. Una ejecucion que quedo «en curso» tras un cierre inesperado se marca como *interrumpida*.

## TensorRT-LLM (informativo)

El Inspector aplica reglas versionadas (`eli/data/trtllm_rules.json`) y cada hallazgo cita su regla y su fuente. Nunca dice
«compatible»: dice *arquitectura potencialmente compatible*, *requiere conversion*, *requiere checkpoint original*, *restricciones*
o *no determinada*. No instala ni convierte nada.

## Pruebas

```bash
PYTHONPATH=tests python3 -m unittest discover -s tests          # 190+ pruebas, solo con Ollama y GPU falsos
ELI_INTEGRATION=1 PYTHONPATH=tests python3 -m unittest tests.test_integration_ollama -v   # real: usa tu GPU y un modelo pequeno
```

## Limites conocidos

- Una sola GPU (`ELI_GPU_INDEX`/indice 0) y una evaluacion a la vez; maximo 8 modelos, 10 repeticiones y contextos de hasta 65 536.
- Las pruebas de contexto largo usan un unico tipo de tarea (recuperar un dato escondido) y el prompt ocupa ~60 % del contexto.
- Las pruebas de idioma usan una heuristica de palabras funcionales; las de programacion son funciones pequenas.
- `nvidia-smi` dice si la GPU esta ocupada, no cuanto aprovecha sus nucleos ni el ancho de banda de memoria.
- No hay graficas de la distribucion por repeticiones ni exportacion a otros formatos; no se pueden borrar ejecuciones.
- Benchmarks multimodales, TensorRT-LLM real, fine-tuning e integracion con CLU: pendientes (ver el roadmap).

## Estructura

```
b18.py              entrada (servidor + navegador)
eli/gpu.py          GIA: muestreo con nvidia-smi, estadisticas, energia
eli/ollama.py       cliente de Ollama (stream cancelable, borrado, metadatos)
eli/generate.py     una generacion medida (la usan el chat y el benchmark)
eli/inspector.py    Model Inspector y analisis TensorRT-LLM; eli/data/trtllm_rules.json
eli/inventory.py    inspeccion del inventario y borrado seguro
eli/hardware.py     informacion del equipo
eli/bench/          bank (banco), graders, sandbox, runner, store (persistencia), compare
eli/server.py       API JSON/SSE y proteccion de host/origen
eli/web/            interfaz (HTML, CSS y JS sin dependencias)
tests/
```
