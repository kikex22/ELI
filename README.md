# ELI - Evaluation Lab for Inference

Dashboard local para evaluar modelos de Ollama: velocidad, memoria de GPU,
consumo de energia y, mas adelante, contexto y calidad. Es independiente de CLU
y no importa nada de el. El plan completo esta en [docs/roadmap_eli.md](docs/roadmap_eli.md).

**Estado:** Parte 1 construida. GIA (panel de GPU), ficha del modelo y chat con
metricas.

## Uso

```bash
B18                      # alias en ~/.bashrc; abre http://127.0.0.1:8766
python3 b18.py --no-browser --port 8766
```

Variables: `ELI_PORT` (8766) y `ELI_OLLAMA_URL` (`http://127.0.0.1:11434`).
El servidor solo escucha en `127.0.0.1` y rechaza otros `Host` y `Origin`.

## Que mide

- **GIA** lee `nvidia-smi` 4 veces por segundo: VRAM, uso de la GPU, consumo (W),
  temperatura, relojes, motivo de frenado y VRAM por proceso.
- **Cada respuesta** muestra tokens generados, tok/s, tiempo al primer token,
  carga del modelo, lectura del prompt, resumen de GPU durante la generacion,
  energia y si el modelo cabe entero en la GPU. Se guarda en `runs/`.

## Como leer las cifras

- **Energia:** potencia media de toda la GPU por la duracion, menos el consumo en
  reposo. El reposo es la mediana de las lecturas con uso <= 15 % de los ultimos
  10 minutos. Sin al menos 2 s de reposo medido, solo se informa la energia bruta.
- **Tokens evaluados:** es lo que Ollama proceso en esa peticion. Si reutiliza la
  cache de la conversacion, es menor que el contexto real.
- **Primer token:** incluye la carga del modelo si estaba descargado.
- **VRAM pico** (de `nvidia-smi`) es mayor que el tamano del modelo que informa
  Ollama: incluye el contexto CUDA y la cache KV.
- **Razonamiento "Por defecto":** con algunos modelos (por ejemplo `gemma4:e4b`)
  el borrador de razonamiento aparece dentro de la respuesta y cuenta en los
  tokens generados. Usa "Activado" o "Desactivado" para separarlo o evitarlo.

## Pruebas

```bash
PYTHONPATH=tests python3 -m unittest discover -s tests
```

Usan un Ollama y una GPU falsos; no necesitan modelos ni hardware.

## Estructura

```
b18.py            entrada (servidor + navegador)
eli/gpu.py        GIA: muestreo con nvidia-smi, estadisticas, energia
eli/ollama.py     cliente de Ollama (solo libreria estandar)
eli/metrics.py    metricas de una respuesta
eli/server.py     API JSON/SSE y proteccion de host/origen
eli/web/          pagina (HTML, CSS, JS sin dependencias)
runs/             una corrida por respuesta (no se versiona)
tests/
```
