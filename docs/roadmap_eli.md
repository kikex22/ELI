# Roadmap de ELI (Evaluation Lab for Inference)

Estado (15 de octubre de 2026): **v0.2 construida**. Cubre la Parte 1 y, ademas, buena parte de las Partes 2 a 6 en una
primera version funcional: benchmark automatico (velocidad, VRAM y contexto), calidad verificable, comparacion e historial,
Model Inspector con analisis TensorRT-LLM informativo y administracion de modelos. Lo que esta en el README como «limites
conocidos» sigue pendiente; el resto de este documento es el plan original.

Pendiente del roadmap tras la v0.2: Parte 3 en profundidad (prueba de calor sostenida), Parte 6 (vision con tus frames y
etiquetas), Parte 7 (adaptador TensorRT-LLM real), Parte 8 (optimizar y volver a medir), Parte 9 (decision sobre entrenamiento)
y Parte 10 (conectar con CLU).

Estado anterior (Parte 1):
Resultados de la Parte 1 que condicionan las siguientes:

- Las lecturas de GIA coinciden con una captura independiente de `nvidia-smi`
  (VRAM, potencia pico, uso y temperatura).
- El consumo en reposo no se puede medir en los segundos previos a una peticion
  (la GPU sigue alta tras la respuesta anterior): se usa la mediana de las
  lecturas con uso <= 15 % de los ultimos 10 minutos.
- Para la Parte 2: usar prompts unicos (la cache de Ollama reduce los tokens
  evaluados) y fijar el razonamiento de forma explicita (en `gemma4:e4b` el modo
  por defecto mete el borrador dentro de la respuesta y cuenta como tokens).
El nombre es provisional. ELI es el dashboard; GIA (GPU & Inference Analyzer)
es su panel de GPU, siempre visible.

## Para que existe

Evaluar como rinde un modelo **solo, fuera de CLU**: velocidad, memoria de GPU,
energia, contexto y calidad. Con esos numeros se decide que modelo usar, se
prueba si optimizarlo (cuantizacion, TensorRT-LLM) vale la pena, y despues se
integra a CLU. CLU no se modifica en ninguna de las primeras partes.

## Reglas de trabajo

- **Por partes.** Cada parte termina con algo que ya se puede usar. No se empieza
  la siguiente hasta que Angel valide la anterior.
- **Aislado de CLU.** Proyecto `~/Documents/ELI/`, con su propio git. No importa nada de CLU: ni `clu_chat.py`, el Controller
  ni el Router.
- **Solo local.** El servidor escucha en `127.0.0.1`. Puerto **8766**: el 8765 ya
  lo usa el dashboard de EMMA.
- **Sin dependencias nuevas.** Servidor con la libreria estandar de Python y
  pagina con HTML/JS simple. Graficas dibujadas sin librerias externas.
- **Medir bien o no medir.** Temperatura 0 y semilla fija; una pasada de
  calentamiento que se descarta; mediana de varias repeticiones; `num_ctx`
  fijado en cada prueba y verificado (Ollama recorta el prompt en silencio si no
  cabe); consumo en reposo medido antes y restado.
- **Una prueba de GPU a la vez.** Antes de cada benchmark se descargan los
  modelos de Ollama que ocupen VRAM.
- **Cada corrida se guarda** como JSON con fecha en `eli/runs/`, con la serie
  completa de lecturas de GPU, para volver a graficarla.
- **Adaptadores de backend desde el inicio.** Ollama primero; TensorRT-LLM despues
  sera otro adaptador con la misma interfaz, no una reescritura.

## Lo que ya existe y se reutiliza

`~/Documents/CLU_TensorRT_Test` (8/10/2026): comparacion TensorRT-LLM 0.19 contra
Ollama con Qwen 2.5 1.5B en BF16, con `scripts/benchmark.py`, prompts de
tres tamanos, resultados y conclusiones (`docs/2026-10-08-findings.md`).
Conclusiones que condicionan este plan:

- TRT-LLM dio mejor tiempo al primer token y mucho mas rendimiento con varios
  clientes, pero uso unos 5 GB mas de VRAM que Ollama.
- Gemma 4 no corrio en TRT-LLM: las versiones con soporte piden CUDA 13 y no
  esta confirmado que el driver 570 con una GeForce lo permita.
- TRT-LLM no carga los GGUF de Ollama; necesita el checkpoint original de
  Hugging Face. Un modelo de 26B no cabe en 16 GB.

## Las partes

### Parte 1 - ELI minimo con GIA
**Queda funcionando:** `B18` abre una pagina local donde eliges un modelo de
Ollama y ves:
- **Panel GIA en vivo (4 lecturas por segundo):** VRAM usada y libre, utilizacion
  de GPU, consumo en vatios contra el limite, temperatura, relojes, motivo de
  frenado y VRAM por proceso.
- **Ficha del modelo:** parametros, cuantizacion, contexto anunciado, capacidades.
- **Chat con metricas** en cada respuesta: tokens de entrada y salida, tok/s,
  tiempo al primer token, tiempo de carga, y resumen de GPU durante la
  generacion (VRAM pico, utilizacion media, vatios, temperatura, Wh por token).

**Verificacion:** una respuesta real de `gemma4:e4b` con sus cifras, y las
lecturas de GIA coherentes con `nvidia-smi` corrido aparte.
**No incluye:** benchmarks automaticos, comparaciones, TensorRT.

### Parte 2 - Velocidad y VRAM contra contexto
**Queda funcionando:** un benchmark que mide tok/s, tiempo al primer token y
VRAM con 500, 2.000, 8.000 y 32.000 tokens de contexto, con curvas. Incluye
la prueba "cabe todo en la GPU?" (tamano del modelo contra VRAM realmente
cargada; avisa si una parte se fue a CPU). Estima el tiempo antes de correr y
pide confirmacion en las pruebas largas.
**Verificacion:** la curva de `gemma4:e4b` y la de `qwen3-coder:30b`, y que se
vea de forma clara que el 30B no cabe entero en 16 GB.
**Decision al terminar:** cuales contextos son realmente usables en esta PC.

### Parte 3 - Eficiencia y prueba de calor
**Queda funcionando:** tokens por vatio y por GB de VRAM; una carga sostenida de
varios minutos que muestra si la GPU se frena por temperatura y cuanto cae la
velocidad.

### Parte 4 - Calidad con respuesta verificable
**Queda funcionando:** pruebas que se califican solas:
- aguja en un pajar (dato escondido en texto largo, en distintas posiciones);
- herramientas (reutiliza los 25 casos de `benchmarks/xlam_routing_suite.json`,
  pero sin CLU en medio);
- formato (JSON valido, respuesta en una linea, solo en espanol);
- codigo y cuentas cortas con resultado comprobable;
- repetibilidad (misma pregunta varias veces con la misma semilla).

**Nota:** las pruebas abiertas no se califican solas; eso es la Parte 5.

### Parte 5 - Comparar e historial
**Queda funcionando:** dos modelos lado a lado con la misma pregunta, votacion a
ciegas, tabla comparativa y un historial de corridas para ver si algo mejoro o
empeoro entre una medicion y otra.

### Parte 6 - Vision con tus datos
**Queda funcionando:** frames de `arm_cam` a un modelo con vision (`gemma4:e4b`),
preguntando por lo que hay en la imagen y comparando contra las etiquetas YOLO
ya existentes, que sirven como respuesta correcta.

### Parte 7 - Adaptador TensorRT-LLM
**Queda funcionando:** el mismo panel y los mismos benchmarks contra un servidor
TRT-LLM en contenedor, con un modelo que quepa en 16 GB (hasta unos 8B).
**Antes de empezar hay que verificar:** que el modelo elegido tenga soporte en la
version de TRT-LLM disponible, y cuanta VRAM reserva su cache KV.
**Regla de comparacion:** misma precision cuando se pueda; la velocidad nunca
se reporta sin la calidad de la Parte 4 al lado.

### Parte 8 - Optimizar y volver a medir
Cuantizacion (por ejemplo FP8, que la RTX 4060 Ti soporta) y ajustes de contexto,
con el mismo benchmark antes y despues. Mide cuanta calidad se pierde por cuanta
velocidad o memoria se gana.

### Parte 9 - Decidir sobre entrenamiento
Punto de decision, no de construccion. Solo se plantea ajuste fino (LoRA) si las
partes anteriores muestran un fallo concreto que el entrenamiento pueda corregir
y se dispone de datos. Fuera del alcance hasta entonces.

### Parte 10 - Conectar con CLU
**Depende de la modularizacion de CLU.** Extraer primero el cliente del modelo
(hoy habla directo con Ollama), para que pueda apuntar a un endpoint compatible
con OpenAI. Cada modelo queda con una ficha en JSON (contexto maximo que cabe,
VRAM, velocidad, eficiencia, calidad con herramientas) que CLU lee para elegir
modelo. Las pruebas de ELI pasan a ser la prueba de regresion al cambiar de
modelo.

## Resumen

| Parte | Entrega | Riesgo |
|---|---|---|
| 1 | Pagina, panel GIA, chat con metricas | Bajo |
| 2 | Curvas de velocidad y VRAM, "cabe en GPU" | Bajo |
| 3 | Eficiencia y prueba de calor | Bajo |
| 4 | Calidad automatica | Medio: elegir buenos casos |
| 5 | Comparar e historial | Bajo |
| 6 | Vision con tus datos | Bajo |
| 7 | Adaptador TensorRT-LLM | Medio: contenedor de 33 GB, versiones |
| 8 | Optimizar y re-medir | Medio |
| 9 | Decidir sobre entrenamiento | Solo decision |
| 10 | Conectar con CLU | Depende de modularizar CLU |

Las partes 1 y 2 ya dan el valor principal: saber como se comporta cada modelo en
esta PC y cuanta GPU consume.

## Decisiones pendientes

1. Nombre definitivo (ELI y GIA propuestos).
2. Que modelos se evaluan primero. Instalados: `gemma4:e4b`,
   `qwen3-coder:30b`, `xLAM-2-3b`, GLM 4.7 flash, y los Qwen 2.5 7B.
3. "Mejorar" un modelo: solo optimizarlo, o tambien entrenarlo (Parte 9).

## Riesgos conocidos

- Con 16 GB de VRAM no se maximizan velocidad, contexto y tamano a la vez.
- Las lecturas de `nvidia-smi` dicen si la GPU esta ocupada, no que tan bien
  aprovecha los nucleos ni el ancho de banda de memoria.
- Cambiar flash attention o la cuantizacion de la cache KV en Ollama exige
  reiniciarlo con otras variables de entorno; lo haria el usuario.
- Pruebas con contextos de 128K pueden tardar mucho y llenar la VRAM.
