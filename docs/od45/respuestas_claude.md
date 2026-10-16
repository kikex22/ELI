# OD45: respuestas de referencia (Claude Code)

Fecha: 16 de octubre de 2026. Código leído: `/home/adjor/Emma/src/od` (versión `OD V3.6`), más
`src/bridge/core/{r2,rgbout}.py` y `scripts/orin.sh`.

**Importante: este archivo NO debe copiarse dentro de `/home/adjor/Emma`.** Si entra al corpus, los
embeddings encontrarían las respuestas aquí en vez de en el código, y la prueba dejaría de medir nada.

## Cómo se usa

Por cada pregunta, este archivo da lo mismo que debería entregar un buen buscador, y algo más:

1. **Archivos**: los archivos que hay que leer para responder, del más importante al menos importante.
   Están en `/home/adjor/Emma/eli_queries.jsonl` con `"kind": "od45"`. ELI compara contra ellos el top-k de
   cada modelo de embeddings (Recall, cobertura por archivo).
2. **Respuesta**: lo que un asistente debería poder decir después de leer esos archivos. Sirve para el
   paso siguiente (un LLM responde con lo que recuperó el embedding, y se compara con esta respuesta).

Las rutas son relativas a `src/od/` salvo que empiecen por `src/bridge/` o `scripts/`.

Tipo de pregunta (cambia lo que un embedding puede lograr):

- **D**: directa; la respuesta está en 1 o 2 archivos.
- **M**: necesita juntar varios archivos (es la prueba de cobertura).
- **C**: conteo o inventario. Un embedding no cuenta; solo puede traer el archivo que permite contar.
- **R**: diagnóstico u opinión. El código da las pistas, pero la respuesta requiere razonar.

### Limitaciones conocidas de la referencia

- `scripts/orin.sh` y `tools/*.sh` no están en el corpus (ELI no indexa `.sh`). La parte de la transmisión de
  video que vive ahí (pregunta 8) no se puede recuperar con ningún modelo.
- Las declaraciones sin cuerpo no generan fragmentos. Por eso `src/pre/pre.hpp`, el único lugar donde aparece
  la entrada de 416×416 (como valor por defecto de `preprocess`), no está en el corpus. Las preguntas 7 y 30
  dependen de que el modelo encuentre `pre.cpp` y de que el lector sepa leer `cv::Size(input_w, input_h)`.
- La referencia la escribí yo leyendo todo el código. Comparar a Claude Code contra su propia referencia
  daría 100 % por definición. Para medir a Claude Code como participante, otra sesión sin acceso a este
  archivo debe responder las 45 preguntas y calificarse igual que los embeddings.
- Algunas respuestas R son hipótesis razonadas desde el código, no comportamientos medidos en el robot.
  Están marcadas con "hipótesis".

---

## A. Arquitectura y funcionamiento

### 1. ¿Cómo está estructurado OD? — M
**Archivos:** `src/main.cpp`, `CMakeLists.txt`, `src/post/post.cpp`, `src/trt/trt.cpp`, `src/pre/pre.cpp`, `arm_cam/arm_cam.cpp`

El paquete ROS 2 `od` genera tres ejecutables (`CMakeLists.txt`):
- `trt_pipeline`: el OD principal. Su nodo se llama `od_node` y compila todos los `src/*.cpp` menos `benchmark.cpp`.
- `arm_cam_udp`: la cámara del brazo (`arm_cam/arm_cam.cpp` + `target_policy.cpp` + los módulos compartidos).
- `od_benchmark`: evaluación offline (mAP, precisión, recall) con imágenes etiquetadas.

El código se divide en módulos:
- `pre/`: preprocesado.
- `trt/`: motor TensorRT.
- `post/`: decodificación, NMS, tracker, revividor de IDs y compensación de movimiento de cámara.
- `monitor/`: grabación, overlay del bin, telemetría y métricas.
- `config/`: rutas locales.
- `msg/`: `Detection2D.msg` y `Distance.msg`.

`main.cpp` orquesta todo en un único bucle de un solo hilo. `trt_pipeline` y `arm_cam_udp` son procesos
independientes: cada uno tiene su propio engine y su propio estado del tracker.

### 2. ¿Cuántos archivos de código tiene OD? — C
**Archivos:** `CMakeLists.txt`

En `src/od/src` hay 42 archivos C++ (20 `.cpp` y 22 `.hpp`, unas 5 200 líneas), y en `arm_cam/` hay 6 más.
Aparte están `msg/` (2 `.msg`), `launch/od.launch.py`, `CMakeLists.txt`, y en `tools/` 2 scripts Python y 3 de shell.

`trt_pipeline` compila por glob 19 de los 20 `.cpp` (todos menos `benchmark.cpp`). Hay código compilado
que el bucle principal no usa: `cam/cam.cpp` (clase `Camera` por V4L2), `post/depth_mask.cpp` y
`post/snapshot.hpp`. `arm_cam/inference.cpp` no aparece en `CMakeLists.txt`.

### 3. ¿Cómo funciona TensorRT en OD? — M
**Archivos:** `src/trt/trt.cpp`, `src/trt/trt.hpp`, `src/main.cpp`, `src/pre/pre.cpp`

`TRTInfer` lee el archivo `.engine` (un engine ya serializado; el programa no construye engines) y hace estos pasos:
1. `createInferRuntime` → `deserializeCudaEngine` → `createExecutionContext`.
2. Lee las formas de los tensores `input`, `output_boxes` y `output_scores`.
3. Reserva memoria fija (pinned) en el host y buffers en la GPU, y enlaza las direcciones con `setTensorAddress`.

En cada frame, `infer()` hace estos pasos:
1. Copia el tensor NCHW a la memoria pinned.
2. `cudaMemcpyAsync` al dispositivo.
3. `enqueueV3` entre dos eventos CUDA, que miden `gpu_ms`.
4. Copia de vuelta cajas y scores.
5. `cudaStreamSynchronize`.

La llamada es síncrona: el bucle espera a la GPU. `main.cpp` le pasa la salida de `preprocess()` y
entrega cajas y scores a `PostProcessor`.

### 4. ¿Cómo está estructurada la implementación de TensorRT? — D
**Archivos:** `src/trt/trt.hpp`, `src/trt/trt.cpp`, `CMakeLists.txt`

Es una clase, `TRTInfer`. Su constructor recibe la ruta del engine y los nombres de los tensores (por
defecto `input`, `output_boxes` y `output_scores`). El destructor libera todo. El único método público de
trabajo es `infer(const float*)`, que devuelve `TRTOutputs{boxes, scores, gpu_ms}`.

Miembros:
- Runtime, engine y contexto.
- Formas, número de elementos y bytes de cada tensor.
- Punteros host (pinned) y device.
- Un `cudaStream_t` y dos `cudaEvent_t`.

El `Logger` está silenciado: la línea que imprime está comentada, así que los errores de TensorRT no se ven.
`CMakeLists.txt` busca `NvInfer.h`, `libnvinfer`, CUDA y NVML. La conversión ONNX → engine está fuera del
programa, en `tools/onnx_to_trt_ui.sh` (`trtexec`, `--fp16` opcional).

### 5. ¿Cómo se graba el feed de video? — M
**Archivos:** `src/monitor/record.cpp`, `src/monitor/recording_control.hpp`, `src/monitor/record.hpp`, `src/main.cpp`

`VideoRecorder` (`record.cpp`) usa `cv::VideoWriter` con códec `mp4v` a 10 FPS (`RECORD_FPS`) y 640×480.
Graba en un archivo temporal `.temp_recording_*_<fecha>.mp4`. Al guardar lo renombra a
`astra_annotated_<fecha>.mp4` o `astra_raw_<fecha>.mp4`; al cancelar lo borra. La ruta es
`<video_directory>/EMMA/astra/<AAAA-MM-DD>/`.

`write()` repite frames si hace falta para que el video dure lo mismo que el tiempo real. Si el bucle va
lento, el video no se acelera.

Se controla de dos formas:
- Teclado: `r`/`s` para el video anotado, `o`/`p` para el crudo.
- ROS: `RecordingController` escucha `/od/recording_cmd` (`start_both`, `start_annotated`, `start_raw`,
  `save`, `cancel`) y publica `/od/recording_status_json` cada 500 ms.

Exige 2 GB libres para empezar. Al cerrar el programa guarda lo que se estuviera grabando. La grabación se
hace en el mismo hilo del bucle.

### 6. ¿Cuántos feeds se pueden grabar simultáneamente? — M
**Archivos:** `src/main.cpp`, `src/monitor/recording_control.hpp`, `src/monitor/record.cpp`, `arm_cam/arm_cam.cpp`

Cada proceso graba **dos** a la vez: el anotado (con cajas y overlay) y el crudo (`raw_frame`, la copia sin
dibujar). `start_both` arranca los dos con la misma marca de tiempo. `arm_cam_udp` tiene su propio par de
grabadores y su propio `/arm_cam/recording_cmd`, así que con los dos procesos son **cuatro**.

### 7. ¿En qué formato y resolución se realizan las inferencias? — D
**Archivos:** `src/pre/pre.cpp`, `src/main.cpp`, `src/trt/trt.cpp` (y `src/pre/pre.hpp`, que no está en el corpus)

El frame llega en BGR 640×480; si trae otro tamaño, `main.cpp` lo redimensiona a 640×480. `preprocess()`
hace estos pasos:
1. Redimensiona directo a **416×416**, sin letterbox: deforma la proporción 4:3 a 1:1.
2. BGR → RGB.
3. Escala a float32 en [0, 1].
4. Reordena a NCHW (1×3×416×416).

Los buffers del programa son float32. La precisión interna del engine (FP32/FP16) no se ve en el código;
depende de cómo se generó con `trtexec`.

### 8. ¿Cómo se transmite el feed de video? — M
**Archivos:** `src/main.cpp`, `arm_cam/arm_cam.cpp`, `src/bridge/core/rgbout.py`, `src/monitor/video_telemetry.hpp`
(y `scripts/orin.sh`, que no está en el corpus)

1. El robot envía H.264 por RTP/UDP al puerto **5600**.
2. `scripts/orin.sh` corre en la PC una tubería `gst-launch` que decodifica con `avdec_h264` (en CPU):
   `udpsrc → rtpjitterbuffer latency=100 → rtph264depay → avdec_h264 → BGR 640×480 → shmsink /tmp/udp_cam`.
3. `main.cpp` lee de esa memoria compartida con
   `shmsrc socket-path=/tmp/udp_cam ! ... ! appsink drop=true sync=false`.

La cámara del brazo usa su propio `udpsrc port=5601` dentro de `arm_cam.cpp`.

El video anotado no se retransmite: se muestra localmente con `cv::imshow`. Hacia afuera salen solo los
datos, por tópicos ROS 2, y `rgbout.py` reenvía `/detections_2d` a ROS 1 por rosbridge (192.168.68.71:9091).

### 9. ¿Cómo está estructurado el main? — D
**Archivos:** `src/main.cpp`

1. **Ayudas** (líneas 55–199): `loadClassNames`, la zona izquierda/centro/derecha, la forma del bbox,
   `--video` y `--config`, y el callback de distancia.
2. **Arranque** (201–329):
   - ROS: nodo `od_node`, 5 publicadores y la suscripción a `/object_distance`.
   - Telemetría de video.
   - Configuración local, clases, `TRTInfer`, telemetría de inferencia y `PostProcessor`.
   - `VideoCapture`: GStreamer en vivo o un archivo de video.
   - Dos grabadores, `RecordingController`, `BinOverlay` y `LiveRunMetrics`.
   - Constantes fijas: `CONF_THRESH = 0.30`, `IOU_THRESH = 0.40`, 640×480, 30 FPS, grabación a 10 FPS.
3. **Bucle `while (rclcpp::ok())`** (331–801):
   - Lectura y redimensionado a 640×480.
   - Flujo óptico para el movimiento de cámara.
   - Preprocesado, TensorRT y postprocesado con tracking.
   - `spin_some` y caducidad de las distancias.
   - Por cada track: dibujo, JSON y `Detection2D`.
   - `target_hint`, `detection_frame_json`, overlay del bin y `/bin_alignment`.
   - Etiqueta de versión, grabación, teclado, `imshow` y métricas.
4. **Cierre**: guarda las grabaciones, escribe el informe de métricas y hace `rclcpp::shutdown()`.

### 10. ¿Qué tracker utiliza OD? — M
**Archivos:** `src/post/post.cpp`, `src/post/tracker_model.cpp`, `src/post/track_reviver.cpp`, `src/post/camera_motion.cpp`, `src/post/post.hpp`

Es un tracker propio, en `PostProcessor`. Los comentarios lo llaman "ByteTrack-lite" y "Kalman-lite", pero
no usa la librería ByteTrack ni un filtro de Kalman completo. Tiene cuatro partes:
- **Asociación en dos pasadas**, al estilo de ByteTrack. Primero las detecciones de confianza alta
  (≥ 0,30), que pueden crear tracks. Después las de confianza baja (0,08–0,30), que solo mantienen tracks
  existentes. La puntuación de cada emparejamiento es 0,65·IoU + 0,35·cercanía, con compuertas
  adaptativas de distancia, IoU y razón de áreas.
- **Modelo de movimiento** (`tracker_model.cpp`): velocidad constante amortiguada y suavizado exponencial
  (EMA) del tamaño y de la caja.
- **Revividor** (`track_reviver.cpp`): guarda los tracks muertos durante 15 frames para recuperar su ID.
- **Compensación de movimiento de cámara** (`camera_motion.cpp`): flujo óptico Lucas–Kanade.

### 11. ¿De qué color son los bounding boxes? — D
**Archivos:** `src/main.cpp`, `src/monitor/bin_overlay.cpp`

Los colores están en BGR, como los usa OpenCV.
- **Detecciones normales**: rectángulo **verde** `(0,255,0)` de 2 px, etiqueta verde y un punto amarillo
  `(0,255,255)` en el centro (`main.cpp` 490–491 y 613–621). El color es el mismo para todas las clases.
- **Textos extra** del `bin` (coordenadas) y de las botellas (forma del bbox): amarillos.
- **Overlay del bin**, cuando está activo:
  - Caja del bin suavizada: amarilla.
  - Línea de soltar: magenta `(255,0,255)`.
  - Rango del brazo: líneas azules `(255,0,0)`.
  - Tolerancia: cian `(255,255,0)`.
  - `BIN_LOST`: rojo.
- **Indicadores de grabación**: "REC" en rojo y "RAW REC" en naranja.

### 12. ¿Cómo se activa el overlay? — D
**Archivos:** `src/main.cpp`, `src/monitor/bin_overlay.cpp`, `src/monitor/bin_overlay.hpp`

Se activa con la tecla **`b` o `B`** en la ventana de OpenCV (`main.cpp` 753–755 → `bin_overlay.toggle()`).
Empieza desactivado (`enabled_(false)`). No hay tópico ROS para activarlo.

Aunque esté apagado, el estado de alineación se calcula igual en cada frame y se publica en `/bin_alignment`
con `"enabled": false`.

### 13. ¿De dónde obtiene el overlay la información que muestra? — M
**Archivos:** `src/monitor/bin_overlay.cpp`, `src/main.cpp`, `src/monitor/bin_overlay.hpp`

Usa los **tracks del frame actual**, ya postprocesados. `main.cpp` arma `overlay_detections` (clase, caja y
score de cada track) y se lo pasa a `bin_overlay.draw()`. El overlay hace estos pasos:
1. Filtra la clase `"bin"`.
2. Elige el mejor candidato: confianza + área + 2·IoU con la caja suavizada anterior.
3. Suaviza la caja con EMA (0,85; 0,95 si el área salta más del 35 %) y la mantiene hasta 5 frames si se pierde.
4. Compara su centro con una línea de soltar fija en x = 320, con rango de 90 px y tolerancia de 35 px.

De ahí salen los estados `BIN_BASE_LEFT/RIGHT`, `BIN_ARM_LEFT/RIGHT`, `BIN_CENTER` y `BIN_LOST`. No usa
profundidad.

### 14. ¿Cómo utiliza OD OpenCV para generar y procesar las imágenes? — M
**Archivos:** `src/pre/pre.cpp`, `src/post/camera_motion.cpp`, `src/main.cpp`, `src/monitor/record.cpp`, `src/monitor/bin_overlay.cpp`

| Tarea | Funciones de OpenCV | Dónde |
| --- | --- | --- |
| Captura | `cv::VideoCapture` con backend GStreamer, o un archivo con `--video` | `main.cpp` |
| Preprocesado | `resize`, `cvtColor` (BGR→RGB), `convertTo` (float), `split` (NCHW) | `pre.cpp` |
| Movimiento de cámara | `cvtColor` a gris, `goodFeaturesToTrack` (300 puntos), `calcOpticalFlowPyrLK`; la mediana del desplazamiento es el movimiento | `camera_motion.cpp` |
| Dibujo | `rectangle`, `circle`, `putText`, `line`, `getTextSize` | `main.cpp`, `bin_overlay.cpp` |
| Visualización | `imshow`, `waitKey` (el teclado también sale de aquí) | `main.cpp` |
| Grabación | `cv::VideoWriter` en `mp4v` | `record.cpp` |

También hay morfología y momentos en `depth_mask.cpp`, pero ese archivo no se usa.

## B. Tracking, visualización y modificaciones

### 15. ¿Por qué se mueven tanto los bounding boxes? — R
**Archivos:** `src/post/tracker_model.cpp`, `src/post/post.cpp`, `src/post/tracker_model.hpp`, `src/post/camera_motion.cpp`, `src/main.cpp`

Hipótesis desde el código, de más a menos probable:
1. **Se dibujan tracks no observados.** `main.cpp` dibuja todos los tracks, también los que tienen
   `missed > 0`. Durante hasta 10 frames esas cajas avanzan solo por la velocidad predicha y luego saltan
   cuando vuelve la detección.
2. **La velocidad estimada oscila.** Se actualiza con la innovación y luego se amortigua (ver la pregunta 45).
   Las predicciones se adelantan o se quedan atrás.
3. **Ruido propio del detector**: cambios de un frame a otro. El suavizado (`box_alpha = 0.85`) solo lo atenúa.
4. **Compensación de cámara errónea.** Si el flujo óptico estima un movimiento que no hubo (escena con poca
   textura, cambio de luz), mueve todos los tracks de golpe hasta 80 px.
5. **`MIN_SEEN_FRAMES = 1`**: un track nuevo y ruidoso aparece desde su primer frame.

Para confirmarlo, compara el movimiento de las cajas en `detection_frame_json` con `observed_this_frame`.

### 16. ¿Cómo podemos evitar que se pierda el tracking cuando un objeto gira? — R
**Archivos:** `src/post/post.cpp`, `src/post/track_reviver.cpp`, `src/post/track_reviver.hpp`, `src/post/post.hpp`

Al girar, la caja cambia de forma y de área, la confianza baja y a veces cambia la clase. Lo que corta el track:
- La compuerta de razón de áreas: `AREA_RATIO_MAX = 1.70` (`area_ratio_limit`, `post.cpp` 299–309 y 340–345).
- Que el emparejamiento exija la misma clase (`post.cpp` 313).
- La compuerta de IoU.

Opciones:
1. Subir `AREA_RATIO_MAX`, o comparar por separado ancho y alto, en vez del área, cuando la velocidad es baja.
2. Permitir emparejar entre clases parecidas, con histéresis: no cambiar de clase hasta verla N frames.
3. Alargar `MAX_MISSED_FRAMES` y el TTL del revividor (15).
4. Añadir apariencia, por ejemplo un histograma de color o un embedding ReID, para decidir el emparejamiento.
5. Entrenar con más vistas rotadas del objeto.

### 17. ¿Cómo podemos cambiar el color de los bounding boxes dependiendo de su clase? — D
**Archivos:** `src/main.cpp`

El color está fijo en `main.cpp`: `cv::rectangle(frame, r, {0,255,0}, 2)` en la línea 490 y la etiqueta con
`{0,255,0}` en la 613–621. Hay que usar una paleta indexada por `t.cls`:

```cpp
static const cv::Scalar kPalette[] = {{0,255,0}, {255,128,0}, {0,0,255}, {255,0,255}};
const cv::Scalar color = kPalette[t.cls % (sizeof(kPalette) / sizeof(kPalette[0]))];
cv::rectangle(frame, r, color, 2);
// ...y el mismo color en el cv::putText de la etiqueta.
```

Si se quiere por nombre (`bin`, `bottle`...), se puede usar un `std::unordered_map<std::string, cv::Scalar>`.
Si se quiere cambiar sin recompilar, hay que leerlo de `local.yaml`, añadiendo una clave en `local_config.cpp`.

## C. Arquitectura y configuración avanzada

### 18. ¿Qué ocurre desde que la cámara captura una imagen hasta que OD detecta un objeto? — M
**Archivos:** `src/main.cpp`, `src/pre/pre.cpp`, `src/trt/trt.cpp`, `src/post/post.cpp`, `src/post/camera_motion.cpp`

1. Cámara del robot → H.264/RTP/UDP:5600 → GStreamer en la PC (jitterbuffer de 100 ms, decodificación en
   CPU) → memoria compartida `/tmp/udp_cam`.
2. `cap.read()` con `appsink drop=true`, que se queda siempre con el último frame. Redimensiona a 640×480.
3. Gris + flujo óptico LK contra el frame anterior → `setCameraMotion(dx, dy)`.
4. `preprocess()`: 416×416, RGB, [0, 1], NCHW.
5. `trt.infer()`: copia a la GPU, `enqueueV3` y copia de vuelta → cajas normalizadas y scores por clase.
6. `process_outputs()`, en este orden:
   - Clase con el mayor score. Se descarta si es < 0,08 o si la caja mide < 10 px.
   - NMS por clase con IoU 0,40.
   - Separación en confianza alta y baja.
   - Compensación de cámara y predicción de los tracks.
   - Emparejamiento de las altas, revivir IDs o crear tracks nuevos.
   - Emparejamiento de las bajas.
   - Envejecimiento y eliminación de tracks.
   - Supresión de tracks solapados (0,60).
7. Por cada track: se le une la distancia si llegó por `/object_distance`, se dibuja y se publica en
   `/detections_2d` y los JSON.

### 19. ¿Qué componentes se inician cuando arrancamos OD? — M
**Archivos:** `src/main.cpp`, `src/config/local_config.cpp`, `src/trt/trt.cpp`, `src/monitor/live_run_metrics.cpp`, `src/monitor/recording_control.hpp`, `launch/od.launch.py`

Dentro de `trt_pipeline`, en este orden:
1. `rclcpp::init` y el nodo `od_node`.
2. Los publicadores `/detections_2d`, `/object_distance_json`, `/od/detection_frame_json`, `/od/target_hint`
   y `/bin_alignment`.
3. La suscripción a `/object_distance`.
4. `VideoTelemetry` (`/od/video_status_json`).
5. `loadLocalConfig`.
6. Los nombres de clases.
7. `TRTInfer`: deserializa el engine y reserva la memoria en la GPU.
8. `InferenceTelemetry`.
9. `PostProcessor`.
10. `VideoCapture`.
11. Dos `VideoRecorder`.
12. `RecordingController`: crea la carpeta, `/od/recording_cmd` y `/od/recording_status_json`.
13. `BinOverlay`.
14. `LiveRunMetrics`, con `GPUMonitor` por NVML.

Fuera del programa: `scripts/orin.sh` (alias B10) abre tmux con el receptor GStreamer, espera 3 s y lanza
`trt_pipeline` y `bridge r2`. `launch/od.launch.py` es otra ruta de arranque: lanza `trt_pipeline` y, cuando
arranca, `bridge rgbout`.

### 20. ¿Qué parámetros podemos cambiar sin recompilar el programa? — D
**Archivos:** `src/config/local_config.cpp`, `src/config/local_config.hpp`, `src/main.cpp`

Se puede cambiar sin recompilar:
- `config/local.yaml`: `engine_path`, `names_path`, `arm_engine_path`, `arm_names_path` y `video_directory`.
- Qué archivo de configuración se usa: `--config <ruta>` o la variable `OD_CONFIG`.
- La fuente de video: `--video <archivo>` (`.mp4`, `.avi`, `.mkv` o `.mov`).
- El contenido de `obj.names` y el propio engine.
- En ejecución: la grabación (por teclado o `/od/recording_cmd`) y el overlay (`b`).

Todo lo demás está fijo en el código y exige recompilar:
- Los umbrales de confianza (0,30) e IoU (0,40).
- La resolución 640×480 y la entrada 416×416.
- Los FPS de grabación.
- Los parámetros del tracker (`post.hpp`, `tracker_model.hpp`, `track_reviver.hpp`).
- El overlay del bin.
- Los nombres de los tópicos.
- La tubería GStreamer.

No hay parámetros ROS.

### 21. ¿Qué sucede si falta un archivo de configuración? — M
**Archivos:** `src/config/local_config.cpp`, `src/config/local_config.hpp`, `src/trt/trt.cpp`, `src/main.cpp`

- **Si falta `local.yaml`:** `loadLocalConfig` prueba en orden `--config`, `OD_CONFIG` y 6 rutas fijas. Si
  no encuentra ninguna, avisa ("No se encontro config/local.yaml; usando rutas por defecto") y usa los
  valores de `local_config.hpp` (`4C_14.engine`...). Las claves desconocidas y las líneas sin `:` se ignoran
  con un aviso.
- **Si falta el engine:** el constructor de `TRTInfer` lanza "No se pudo abrir engine" y, como nadie captura
  la excepción, el proceso termina. Si el archivo existe pero no se puede deserializar, el código llama a
  `createExecutionContext()` sobre un engine nulo **antes** de comprobarlo (`trt.cpp` 63–66): eso puede
  terminar en un fallo de segmentación en vez de un mensaje claro.
- **Si falta `obj.names`:** `loadClassNames` devuelve una lista vacía, sin avisar. `process_outputs` sale al
  instante (`num_classes <= 0`). OD funciona y muestra el video, pero **nunca detecta nada** y no da ningún error.

### 22. ¿Cómo se comunican entre sí las diferentes partes de OD? — M
**Archivos:** `src/main.cpp`, `msg/Detection2D.msg`, `msg/Distance.msg`, `src/monitor/recording_control.hpp`, `src/bridge/core/r2.py`, `src/bridge/core/rgbout.py`

- **Dentro de `trt_pipeline`:** llamadas directas entre módulos en un único hilo
  (`preprocess → TRTInfer::infer → PostProcessor::process_outputs → BinOverlay/VideoRecorder`).
  Los callbacks de ROS solo corren en `exec.spin_some()`, una vez por frame.
- **Video de entrada:** UDP/RTP y después memoria compartida (`/tmp/udp_cam`).
- **Con otros nodos:** tópicos ROS 2.
  - Salen: `od/Detection2D` y JSON en `std_msgs/String`.
  - Entra `od/Distance`, desde Astra en el Orin o `r2.py`.
  - Comandos de grabación.
- **Con ROS 1:** `rgbout.py`, por rosbridge (roslibpy).
- **`arm_cam_udp`:** es otro proceso y no comparte memoria con `trt_pipeline`. Comparte el código fuente,
  pero sus datos solo viajan por sus propios tópicos (`/arm_cam/...`).

## D. Video y procesamiento

### 23. ¿Por qué OD tarda tanto en mostrar la imagen al iniciarse? — R
**Archivos:** `src/trt/trt.cpp`, `src/main.cpp`, `src/monitor/live_run_metrics.cpp`

Hipótesis; el código no mide el arranque. La espera se suma de estas partes:
1. Inicialización de ROS 2 (descubrimiento DDS).
2. Creación del contexto CUDA y deserialización del engine (`TRTInfer`). Suele ser lo más lento: de cientos
   de ms a varios segundos.
3. Inicialización de NVML (`GPUMonitor`).
4. Arranque de la tubería GStreamer. `orin.sh` además espera 3 s fijos antes de lanzar `trt_pipeline`, y el
   decodificador H.264 necesita un keyframe (I-frame) para la primera imagen.
5. Primeras inferencias más lentas (calentamiento). `LiveRunMetrics` descarta 10 frames por eso.

Para medirlo, pon marcas de tiempo alrededor de `TRTInfer` y del primer `cap.read()` exitoso.

### 24. ¿Qué sucede si la cámara se desconecta mientras OD está funcionando? — D
**Archivos:** `src/main.cpp`, `src/monitor/video_telemetry.hpp`, `src/monitor/live_run_metrics.cpp`

En vivo, `cap.read()` falla, se cuentan los fallos (`video_telemetry.readFailure()` y
`run_metrics.readFailure()`), se llama a `spin_some` y se hace `continue`. El proceso no termina, pero:
- **Nunca reabre la captura.** Si GStreamer no se recupera solo, OD queda esperando para siempre.
- **No espera entre intentos**: el bucle reintenta sin pausa y gasta CPU.
- **La ventana se queda congelada** en el último frame: `imshow` no se llama sin frame.
- **Deja de publicar detecciones.**

`/od/video_status_json` muestra `"alive": false` y sube `frame_read_failure_percent`.
En modo `--video`, el fin del archivo termina el bucle.

### 25. ¿Qué pasa si la cámara entrega imágenes más rápido de lo que OD puede procesarlas? — D
**Archivos:** `src/main.cpp`, `src/monitor/record.cpp`, `src/monitor/video_telemetry.hpp`

`appsink drop=true sync=false` descarta los frames viejos: OD siempre toma el más reciente y no se forma
cola. Se pierden frames en silencio, pero la latencia no crece. El jitterbuffer de `orin.sh` también
descarta lo que llega tarde (`drop-on-latency=true`).

Efectos:
- `annotated_fps` en `/od/video_status_json` baja a la velocidad real del bucle.
- El tracker ve saltos más grandes entre frames: las compuertas crecen con la velocidad, pero hay más
  riesgo de perder el emparejamiento.
- Los videos grabados repiten frames para mantener 10 FPS de tiempo real.

### 26. ¿Por qué el video se ve fluido, pero los bounding boxes aparecen retrasados? — R
**Archivos:** `src/post/tracker_model.cpp`, `src/post/tracker_model.hpp`, `src/post/post.cpp`, `src/main.cpp`

La caja se dibuja sobre el mismo frame en que se calcula, así que el desfase no viene del video. Viene del
**suavizado**. La salida usa la caja suavizada `bx1..by2` con `box_alpha = 0.85`: cada frame la caja avanza
solo un 15 % hacia la detección nueva.

Además la velocidad se estima a la mitad (ver la pregunta 45). Para un objeto que se mueve a velocidad
constante, la caja queda unos **3 frames atrás** (unos 100 ms a 30 FPS) y se nota como un "arrastre".

Si se ve en otra pantalla (dashboard, ROS 1), se suma la latencia del tópico. Esos consumidores reciben las
cajas por `/detections_2d` sin el video del mismo instante.

### 27. ¿Cómo podemos identificar dónde se está produciendo el retraso del video? — M
**Archivos:** `src/monitor/live_run_metrics.cpp`, `src/monitor/video_telemetry.hpp`, `src/monitor/inference_telemetry.hpp`, `src/main.cpp`, `src/monitor/live_run_metrics.hpp`

Usa la telemetría que ya existe:
- **`/od/video_status_json`** (cada segundo): FPS, jitter entre frames, latencia media y máxima de
  procesamiento, y porcentaje de fallos de lectura.
- **`/od/inference_status_json`**: tiempo de GPU medio, p95 y máximo.
- **Informe `LiveRunMetrics`**, escrito al cerrar en `<video_directory>/EMMA/metrics/od_live_<fecha>.json`.
  Da media, p95 y máximo de cada etapa: intervalo entre frames, preprocesado, inferencia en GPU,
  postprocesado y pipeline completo, junto con el uso de la GPU.

Detalles que hay que saber:
- `preprocess_ms` incluye el flujo óptico, porque el cronómetro empieza antes de `cvtColor`.
- Nada mide el transporte (UDP + jitterbuffer de 100 ms + decodificación H.264) ni el tiempo de pantalla.
  Para eso hay que poner una marca de tiempo en el emisor, o filmar a la vez un reloj y la pantalla
  (latencia "glass-to-glass").

## E. Detección de objetos

### 28. ¿De dónde obtiene OD los nombres de las clases que detecta? — D
**Archivos:** `src/main.cpp`, `src/config/local_config.hpp`, `src/config/local_config.cpp`, `src/post/post.cpp`

Del archivo `obj.names` que indica `names_path` en `config/local.yaml`. `loadClassNames` lo lee con un nombre
por línea y omite las líneas vacías. El índice de la línea es el ID de clase: `post.cpp` toma el mayor score
de `output_scores` y `main.cpp` hace `classNames[t.cls]`.

El orden tiene que coincidir con el del entrenamiento. Si el archivo tiene menos clases que el engine, el
decodificado lee las filas de scores con el tamaño equivocado (`scores_elems / num_classes`). No hay
comprobación de límites.

### 29. ¿Por qué OD detecta objetos que realmente no existen? — R
**Archivos:** `src/main.cpp`, `src/post/post.cpp`, `src/post/post.hpp`, `src/pre/pre.cpp`

La causa de fondo es el modelo (datos de entrenamiento). El código permite que esos falsos positivos lleguen
a la pantalla por cuatro vías:
1. `CONF_THRESH = 0.30` es bajo.
2. `MIN_SEEN_FRAMES = 1`: un falso positivo de un solo frame ya aparece y se publica.
3. Los tracks no observados se siguen dibujando y publicando en `/detections_2d` hasta 10 frames: son
   "cajas fantasma". `observed_this_frame` solo aparece en el JSON.
4. El redimensionado sin letterbox deforma la imagen respecto de la que vio el modelo, si se entrenó con letterbox.

Mitigaciones:
- Subir el umbral.
- Exigir `seen ≥ 2–3`.
- No dibujar ni publicar los tracks con `missed > 0`.
- Reentrenar con negativos difíciles.

### 30. ¿Por qué le cuesta detectar objetos pequeños o que están lejos? — R
**Archivos:** `src/pre/pre.cpp`, `src/post/post.cpp`, `src/bridge/core/r2.py` (y `src/pre/pre.hpp`, que no está en el corpus)

1. **Resolución**: 640×480 se reduce a 416×416, y lo pequeño queda en pocos píxeles.
2. **Filtros duros**: se descartan las detecciones de menos de 10 px de ancho o alto (`post.cpp` 167) y las
   cajas de salida de menos de 5 px (`post.cpp` 482).
3. **Modelo**: YOLOv4 (según la ruta del engine) es débil en objetos pequeños.
4. **Distancia**: más allá de 3 m tampoco hay medida, porque `r2.py` limita a 0,15–3,0 m.

Mejoras:
- Entrada de 608×608 o mayor.
- Recorte en mosaico (tiling) de la zona de interés.
- Bajar el filtro de 10 px.
- Entrenar con más ejemplos lejanos.

### 31. ¿Qué sucede cuando cambia bruscamente la iluminación? — R
**Archivos:** `src/post/camera_motion.cpp`, `src/pre/pre.cpp`, `src/post/post.cpp`, `src/post/tracker_model.cpp`

El código no compensa la iluminación: no hay normalización, CLAHE ni control de exposición, solo /255.
Consecuencias probables:
1. La confianza del detector baja. Los tracks sobreviven un tiempo gracias a la pasada de confianza baja
   (≥ 0,08) y a los 10 frames de gracia, y después se pierden o cambian de ID.
2. El flujo óptico Lucas–Kanade supone brillo constante. Un cambio brusco puede generar un "movimiento de
   cámara" falso que desplaza todos los tracks (hasta 80 px).
3. La cámara reajusta su exposición, y eso tarda unos frames.

Mejoras:
- Aumentos de brillo y contraste en el entrenamiento.
- Ecualización (CLAHE) antes de preprocesar.
- Rechazar movimientos de cámara cuando cambia mucho la intensidad media.

### 32. ¿Cómo decide OD cuáles detecciones descartar por tener poca confianza? — D
**Archivos:** `src/post/post.cpp`, `src/main.cpp`

En `process_outputs`:
1. Por cada caja toma la clase con el mayor score.
2. Si el score es **< 0,08** (`LOW_CONF_THRESH`), la descarta.
3. Si está entre **0,08 y 0,30** (`CONF_THRESH`, que viene de `main.cpp`), la marca como "baja": no crea
   tracks, solo puede mantener uno existente, con corrección suave.
4. Si es **≥ 0,30**, es "alta": crea tracks o los actualiza.

Además se aplica NMS por clase con IoU 0,40, después una supresión de tracks solapados de la misma clase
(IoU 0,60), y `prune_tracks_global` se queda con 30 tracks como máximo, ordenados por prioridad.
Para elegir el objetivo (`target_eligible`) se exige score ≥ 0,30 y al menos 2 frames de continuidad.

## F. Seguimiento de objetos

### 33. ¿Qué sucede cuando dos objetos se cruzan frente a la cámara? — R
**Archivos:** `src/post/post.cpp`, `src/post/tracker_model.cpp`, `src/post/post.hpp`

- **Si son de clases distintas**, no se afectan, porque el emparejamiento exige la misma clase.
- **Si son de la misma clase**, hay tres problemas:
  1. El emparejamiento es **voraz, no húngaro**: cada detección, en orden, toma el mejor track libre. Al
     cruzarse, los IDs pueden intercambiarse.
  2. Si uno tapa al otro, el tapado pasa a `missed` y sigue con su velocidad predicha hasta 10 frames.
  3. La supresión final (IoU > 0,60) oculta uno de los dos tracks mientras se solapan, aunque el track
     siga vivo por dentro.

  La velocidad, que se subestima (ver la pregunta 45), ayuda poco a desambiguar.

Para mejorar: asignación húngara, o apariencia (ReID).

### 34. ¿Por qué a veces un mismo objeto recibe diferentes identificadores? — M
**Archivos:** `src/post/post.cpp`, `src/post/track_reviver.cpp`, `src/post/track_reviver.hpp`, `src/post/post.hpp`

Hay cuatro causas en el código:
1. **Cambio de clase**: el detector duda entre dos clases. Como el emparejamiento exige la misma clase,
   se crea un track nuevo.
2. **Más de 10 frames sin detección** (`MAX_MISSED_FRAMES`). Pasa al revividor, que solo lo recupera si
   reaparece antes de 15 frames, a menos de 180 px, con la misma clase y una razón de áreas ≤ 3.
3. **Cambios de tamaño o forma** (giro, acercamiento rápido) que superan las compuertas de área o IoU.
4. **Detección débil**: si al reaparecer el score es < 0,30, no puede crear ni revivir un track.

Además, `arm_cam_udp` tiene su propio contador de IDs, así que el mismo objeto visto por las dos cámaras
tiene IDs distintos.

### 35. ¿Qué ocurre cuando un objeto sale de la imagen y después regresa? — M
**Archivos:** `src/post/track_reviver.cpp`, `src/post/track_reviver.hpp`, `src/post/post.cpp`, `src/post/post.hpp`

1. **Del frame 1 al 10 sin verlo**: el track sigue vivo y predicho; su caja se dibuja recortada al borde.
2. Después pasa al buffer del revividor durante **15 frames** (TTL). Ahí también se le aplica la
   compensación de movimiento de cámara.
3. **Si vuelve antes**, unos 25 frames en total (unos 0,8 s a 30 FPS), cerca de donde se fue (< 180 px),
   con la misma clase y un área compatible: **recupera su ID**. Su velocidad se recalcula a partir del
   desplazamiento.
4. **Si vuelve más tarde o más lejos**: recibe un **ID nuevo**. `main.cpp` también borra su continuidad y su
   distancia guardada.

### 36. ¿Cómo podemos suavizar los bounding boxes sin introducir demasiado retraso? — R
**Archivos:** `src/post/tracker_model.cpp`, `src/post/tracker_model.hpp`, `src/post/post.cpp`

El retraso actual viene del EMA fijo (`box_alpha = 0.85`) junto con una velocidad subestimada. Opciones:
1. **Arreglar la velocidad** (ver la pregunta 45). Con una buena predicción, el EMA ya no arrastra.
2. **Suavizado adaptativo, tipo One Euro filter**: suaviza mucho cuando el objeto está quieto y poco cuando
   se mueve. Por ejemplo, un alfa que baje con la velocidad o con el IoU entre predicción y detección.
3. **Suavizar más el tamaño que el centro**, porque el temblor suele estar en el tamaño.
4. **Un Kalman real con covarianza**, donde la ganancia sale del ruido. Se puede bajar la confianza en
   detecciones débiles en vez de usar alfas fijos.
5. Bajar `box_alpha` a 0,6–0,7 para las detecciones de confianza alta.

### 37. ¿Cómo determina OD cuándo debe dejar de seguir un objeto? — D
**Archivos:** `src/post/post.cpp`, `src/post/post.hpp`, `src/post/track_reviver.hpp`, `src/post/track_reviver.cpp`

- **Por frames sin detección**: cada frame sin emparejar suma `missed`. Con `missed > MAX_MISSED_FRAMES`
  (10; 8 en el modo de la cámara del brazo) el track pasa al revividor, que lo descarta del todo tras
  `TTL_FRAMES = 15`.
- **Por límite de tracks**: si hay más de 30 (`MAX_TRACKS_TOTAL`), se eliminan los de menor prioridad
  (`score·(1 + seen/5) − 0,2·missed`).
- **Por tamaño**: una caja de salida de menos de 5 px no se publica en ese frame.

`main.cpp` borra la continuidad y la distancia de los IDs que desaparecen.

## G. Profundidad, ROS y TensorRT

### 38. ¿Cómo calcula OD la distancia entre la cámara y un objeto? — M
**Archivos:** `src/bridge/core/r2.py`, `src/main.cpp`, `msg/Distance.msg`

**OD no la calcula.** La recibe por `/object_distance` (`od/Distance`: `track_id`, `class_name` y `distance`)
de un proveedor externo: la profundidad de la Astra en el Orin, o `bridge r2.py` en la PC.

`r2.py` hace estos pasos:
1. Toma `/detections_2d` y la imagen de profundidad (`/camera/depth/image_raw`, en mm → m).
2. Recorta el bbox y su centro (quita el 25 % de cada lado).
3. Se queda con los valores entre 0,15 y 3,0 m.
4. Usa como distancia el **percentil 15**, que es robusto al fondo.
5. Con los intrínsecos (`camera_info`), calcula el desplazamiento lateral, el ángulo y el lado.
6. Publica **solo el objeto más cercano**.

`main.cpp` guarda la distancia por `track_id`, con la hora en que llegó.

### 39. ¿Qué sucede cuando hay detecciones, pero no llegan datos de profundidad? — D
**Archivos:** `src/main.cpp`, `src/bridge/core/r2.py`

La distancia queda en −1 y todo sigue funcionando sin metros:
- La etiqueta no muestra metros.
- **No se publica `/object_distance_json`**, porque solo sale si hay distancia vigente.
- `/od/detection_frame_json` lleva `"distance_available": false` y `distance_m: null`.
- `/detections_2d` y `/od/target_hint` siguen igual, porque no dependen de la distancia.

Una distancia caduca a los **350 ms** (`DISTANCE_TIMEOUT`). En `r2.py`, sin `camera_info` válido o sin
profundidad válida en el bbox, no se publica nada.

### 40. ¿Cómo relaciona OD la información de profundidad con cada bounding box? — M
**Archivos:** `src/main.cpp`, `src/bridge/core/r2.py`, `msg/Distance.msg`, `msg/Detection2D.msg`

Por el **`track_id`**:
1. OD publica cada bbox con su `track_id` en `/detections_2d`.
2. El proveedor de profundidad mide dentro de ese bbox y responde con el mismo `track_id` y la clase.
3. `main.cpp` acepta la distancia solo si se cumplen tres condiciones: mismo `track_id`, misma
   `class_name`, y el track fue observado en este frame (`missed == 0`).
4. Caduca a los 350 ms y se borra cuando el track deja de observarse o desaparece.

Hay un riesgo: `r2.py` usa las coordenadas del bbox (640×480 de la cámara RGB) directamente sobre la imagen
de profundidad, sin registro entre cámaras. Si las dos cámaras no están alineadas o tienen otra resolución,
la profundidad se toma del lugar equivocado.

### 41. ¿Qué información envía OD a los demás nodos de ROS 2? — M
**Archivos:** `src/main.cpp`, `msg/Detection2D.msg`, `src/monitor/recording_control.hpp`, `src/monitor/video_telemetry.hpp`, `src/monitor/inference_telemetry.hpp`

| Tópico | Tipo | Contenido |
| --- | --- | --- |
| `/detections_2d` | `od/Detection2D` | Un mensaje por track y por frame: `track_id`, `class_name`, `confidence`, bbox y centro. Incluye los tracks predichos. |
| `/od/detection_frame_json` | String | Todos los tracks del frame: `observed_this_frame`, `stable_frames`, puntuaciones, distancia y edad. |
| `/od/target_hint` | String | Objetivo recomendado (`visual_score = 0.6·bottom + 0.2·centro + 0.1·conf + 0.1·estabilidad`). Solo se publica si es válido. |
| `/object_distance_json` | String | Distancia, `x_offset` normalizado, bbox, área, relación de aspecto y forma 2D. Solo con distancia vigente. |
| `/bin_alignment` | String | Estado de alineación del bin: `status`, `offset`, `ready` y centro. |
| `/od/video_status_json` | String | Telemetría de video, cada segundo. |
| `/od/inference_status_json` | String | Telemetría de la GPU, cada segundo. |
| `/od/recording_status_json` | String | Estado de grabación, cada 500 ms. |

Recibe `/object_distance` y `/od/recording_cmd`.

### 42. ¿Por qué TensorRT podría dejar de funcionar al cargar un modelo nuevo? — R
**Archivos:** `src/trt/trt.cpp`, `src/trt/trt.hpp`, `src/post/post.cpp`, `src/config/local_config.cpp`

Hay seis motivos posibles, en orden de probabilidad:
1. **El engine se generó con otra versión de TensorRT o en otra GPU.** Los engines no son portables.
   `deserializeCudaEngine` devuelve nulo y, por el orden del código, se llama a `createExecutionContext`
   sobre ese nulo (fallo de segmentación). El logger silenciado oculta el motivo.
2. **Los nombres de los tensores no coinciden.** El código exige `input`, `output_boxes` y `output_scores`.
   Un ONNX exportado con otros nombres (por ejemplo `images` y `output0` de YOLOv5/8) da formas inválidas.
3. **El engine tiene formas dinámicas** (−1). El código nunca llama a `setInputShape`, así que los tamaños
   salen mal.
4. **Otra resolución de entrada.** `preprocess` siempre produce 416×416; si el modelo espera 640×640, la
   copia se sale del buffer o lee basura.
5. **Otro formato de salida.** Se esperan cajas `x1, y1, x2, y2` normalizadas y scores por clase ya
   combinados. Un YOLO con `cx, cy, w, h`, objectness aparte o salida en un solo tensor necesita otro
   decodificado. `snapshot.hpp` sí asume `cx, cy, w, h`.
6. **Otro número de clases que `obj.names`**: decodificado erróneo, sin aviso.

## H. Análisis y optimización avanzada

### 43. ¿Cuál es el archivo de código más importante de OD y por qué? — R
**Archivos:** `src/main.cpp`, `src/post/post.cpp`

**`src/main.cpp`**: es el único que conecta todo. Contiene el bucle, abre la captura, llama a preprocesado,
TensorRT y tracker, une la distancia, decide el objetivo (`target_hint`) y publica todos los tópicos. Si
falta, no hay OD.

El segundo es **`src/post/post.cpp`**: contiene la lógica más compleja y más fácil de romper (decodificado,
NMS, asociación y ciclo de vida de los tracks). Ahí se decide qué ve el robot.

### 44. ¿Cómo podemos reducir aún más la latencia de OD? — R
**Archivos:** `src/main.cpp`, `src/pre/pre.cpp`, `src/trt/trt.cpp`, `src/post/camera_motion.cpp`, `src/monitor/record.cpp`, `src/post/post.cpp`

Primero hay que medir (ver la pregunta 27). Candidatos, de mayor a menor impacto probable:
1. **Transporte**: bajar `rtpjitterbuffer latency=100` y decodificar H.264 por hardware (`nvh264dec` o
   NVDEC) en vez de `avdec_h264` (CPU).
2. **Pipeline en paralelo**: hoy captura, inferencia, postproceso, dibujo y grabación van en serie en un
   solo hilo. Se puede capturar en un hilo y grabar (`VideoWriter`, que codifica en CPU) y publicar en otro.
3. **Flujo óptico**: `goodFeaturesToTrack` (300 puntos) y LK en 640×480 en cada frame. Se puede hacer a
   media resolución o cada N frames.
4. **Preprocesado en GPU** (CUDA o NPP): `resize`, conversión a RGB y normalización. Evita CPU y copias.
5. **Engine en FP16 o INT8**, si aún es FP32, y CUDA Graphs.
6. **Menos copias**: `infer()` copia a la memoria pinned y luego a `std::vector`; se puede leer directo de
   la memoria pinned.
7. **Menos trabajo por frame**: construir JSON con `ostringstream` para cada track e `imshow` cuestan CPU.
   `waitKey(1)` ya es mínimo.

### 45. ¿Cuál es el cálculo de Kalman utilizado en OD y cómo se puede mejorar? — M
**Archivos:** `src/post/tracker_model.cpp`, `src/post/tracker_model.hpp`, `src/post/tracker_state.hpp`, `src/post/post.cpp`

No es un filtro de Kalman: es un filtro de ganancias fijas, de la familia alfa-beta, con EMA. El estado
(`TrackState`) es `cx, cy, vx, vy, w, h`, más la caja suavizada `bx1..by2`, en píxeles por frame. Los
parámetros están en `TrackerModelConfig`: `vel_gain = 0.55`, `vel_damp = 0.90`, `size_alpha = 0.70`,
`box_alpha = 0.85` y `score_inertia = 0.90`.

**Predicción** (`predict_track`):
```
c ← c + v
v ← 0.90·v
caja ← caja + v
```

**Corrección con una detección z** (`update_track_from_det`):
```
innovación  d = z − c_predicho
velocidad   v ← (1 − g)·v + g·d        g = 0,55 (alta) o 0,19 (baja)
alta:       c ← z;   w,h ← 0,7·w,h + 0,3·det;   caja ← 0,85·caja + 0,15·caja_det
baja:       c ← c + g·d;   suavizado aún mayor;   el score no sube
```

**Defecto**: la velocidad se actualiza con la *innovación* y no con el desplazamiento, y además se
amortigua. Para un objeto a velocidad constante u, el valor estable es
v ≈ 0,55u / 1,145 ≈ **0,48u**: estima la mitad de la velocidad real. Por eso las predicciones se quedan
cortas y las cajas arrastran (preguntas 26 y 36). En un alfa-beta correcto, `v ← v + β·d`.

Hay otras limitaciones:
- No hay covarianza: la ganancia no depende de la incertidumbre.
- Las compuertas son fijas en píxeles, no de Mahalanobis.
- Las unidades son por frame y no usan dt: si bajan los FPS, cambia el comportamiento.

Mejoras:
1. **Un Kalman real** con velocidad constante, como en SORT o ByteTrack: estado `[cx, cy, a, h, vx, vy, va, vh]`,
   matrices F, H, Q y R, predicción `P ← FPFᵀ + Q` y ganancia `K = PHᵀ(HPHᵀ + R)⁻¹`.
2. **Ruido de medida R escalado por la confianza**, como el NSA Kalman de StrongSORT, en vez de la rama
   "baja" fija.
3. **Compuerta de Mahalanobis** con la covarianza en lugar de los píxeles fijos.
4. **Usar dt real** entre frames.
5. Con el Kalman, quitar el EMA de la caja, o hacerlo adaptativo.
