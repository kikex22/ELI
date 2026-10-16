# OD de EMMA — 45 preguntas (Claude Code, prueba ciega)

Respuestas escritas leyendo solo el código de `/home/adjor/Emma` (sobre todo `src/od`, `src/bridge` y `scripts/`). Las rutas son relativas a `/home/adjor/Emma`. Lo que el código no permite confirmar va marcado como **hipótesis**.

Resumen de la arquitectura: el robot envía H.264/RTP por UDP:5600. `scripts/pc.sh` lo decodifica con `gst-launch` y lo deja en memoria compartida (`/tmp/udp_cam`). El ejecutable `trt_pipeline` (`src/od/src/main.cpp`) lee esa memoria compartida y ejecuta, en un solo hilo, este bucle: preproceso, TensorRT, postproceso y tracker, dibujo, publicación en ROS 2 y grabación. La distancia la calcula `src/bridge/core/r2.py` con la cámara de profundidad y se la devuelve a OD por `/object_distance`.

---

## od45-01. ¿Cómo está estructurado OD?
**Archivos:** src/od/CMakeLists.txt, src/od/src/main.cpp, src/od/src/post/post.cpp, src/od/src/trt/trt.cpp, src/od/src/pre/pre.cpp, src/od/arm_cam/arm_cam.cpp

**Respuesta:** OD es un paquete ROS 2 (ament_cmake + CUDA) con 3 ejecutables (`CMakeLists.txt:80-189`):
- `trt_pipeline`: todo `src/*.cpp` salvo benchmark. Es la cámara principal (Astra).
- `arm_cam_udp`: `arm_cam/*.cpp` más los módulos compartidos. Es la cámara del brazo, UDP:5601.
- `od_benchmark`: benchmark offline.

Módulos dentro de `src/od/src/`:
- `pre/`: preproceso con OpenCV a un blob NCHW.
- `trt/`: wrapper de TensorRT.
- `post/`: decodificación, NMS, tracker "ByteTrack-lite" con modelo "Kalman-lite", reviver de tracks, compensación de movimiento de cámara y depth_mask.
- `monitor/`: grabación, overlay del bin, telemetrías, métricas y monitor GPU (NVML).
- `config/`: `local.yaml`.
- `cam/`: clase Camera V4L2, que no se usa.

Además: `msg/` con `Detection2D` y `Distance`, `launch/od.launch.py` y `tools/` (conversión ONNX→TRT y modo AP).

## od45-02. ¿Cuántos archivos de código tiene OD?
**Archivos:** src/od/CMakeLists.txt, src/od/src/main.cpp, src/od/arm_cam/arm_cam.cpp, src/od/src/post/post.cpp

**Respuesta:**
- `src/od/src`: 42 archivos C++ (20 `.cpp` y 22 `.hpp`).
- `src/od/arm_cam`: 5 archivos C++ (`arm_cam.cpp`, `target_policy.cpp/.hpp`, `callback.hpp`, `contact_diagnostic.hpp`) y además `inference.cpp`, que CMake no compila. En total son 48 archivos C++.
- Otros archivos de código: `CMakeLists.txt`, `launch/od.launch.py`, 2 `.msg` y 4 en `tools/` (`onnx_to_trt_ui.sh`, `filter_import.py`, `C12_ap_mode.sh`, `C13_wifi_mode.sh`).

Sin contar `build/`, `install/` ni `log/`. El archivo más grande es `arm_cam.cpp` (1894 líneas), seguido de `main.cpp` (807).

## od45-03. ¿Cómo funciona TensorRT en OD?
**Archivos:** src/od/src/trt/trt.cpp, src/od/src/trt/trt.hpp, src/od/src/main.cpp, src/od/tools/onnx_to_trt_ui.sh, src/od/src/pre/pre.cpp

**Respuesta:**
- **Construcción del engine:** el `.engine` se genera fuera del programa con `trtexec`, desde un ONNX (FP16 o FP32), usando `tools/onnx_to_trt_ui.sh:23-30`.
- **Carga al iniciar:** `TRTInfer` lee el archivo y llama a `deserializeCudaEngine` y luego a `createExecutionContext` (`trt.cpp:51-67`). Obtiene las formas de los tensores `input`, `output_boxes` y `output_scores`. Reserva memoria *pinned* en el host (`cudaMallocHost`) y memoria en el device, y fija las direcciones con `setTensorAddress` (`trt.cpp:88-100`).
- **En cada frame:** `infer()` hace `memcpy` al buffer pinned, `cudaMemcpyAsync` H2D, `enqueueV3` en un stream CUDA (con eventos que miden `gpu_ms`), copias D2H de boxes y scores, y `cudaStreamSynchronize` (`trt.cpp:121-145`). Es una inferencia síncrona con batch 1.
- `main.cpp:250` crea el objeto con `local_config.engine_path` y `main.cpp:371` lo llama.

## od45-04. ¿Cómo está estructurada la implementación de TensorRT?
**Archivos:** src/od/src/trt/trt.hpp, src/od/src/trt/trt.cpp, src/od/CMakeLists.txt, src/od/tools/onnx_to_trt_ui.sh

**Respuesta:**
- **Clase `TRTInfer`** (`trt.hpp:13-61`): punteros a `IRuntime`, `ICudaEngine` e `IExecutionContext`; nombres de los tensores (los valores por defecto son `"input"`, `"output_boxes"` y `"output_scores"`); formas y tamaños de cada tensor; buffers `h_*` (pinned) y `d_*` (device); un `cudaStream_t` y dos `cudaEvent_t`.
- **Constructor**: carga el engine y reserva la memoria. **Destructor**: libera todo. **`infer()`**: devuelve `TRTOutputs{boxes, scores, gpu_ms}`.
- **Helpers** (`trt.cpp:9-31`): un `Logger` silenciado (el `cerr` está comentado en la línea 15) y `ck()`, que lanza una excepción ante cualquier error CUDA.
- **CMake** busca `NvInfer.h`, `nvinfer` y `cudart` (`CMakeLists.txt:24-55`).

## od45-05. ¿Cómo se graba el feed de video?
**Archivos:** src/od/src/monitor/record.cpp, src/od/src/monitor/record.hpp, src/od/src/monitor/recording_control.hpp, src/od/src/main.cpp

**Respuesta:** `VideoRecorder` usa `cv::VideoWriter` con FOURCC `mp4v` (`record.cpp:14`).
- `start()` crea `<video_directory>/EMMA/astra/<fecha>/` y abre un archivo temporal `.temp_recording_*_<stamp>.mp4` (`record.cpp:87-140`).
- `write()` mantiene un FPS constante: si van retrasados, duplica frames hasta igualar `elapsed*fps` (`record.cpp:63-82`).
- `save()` renombra el temporal a `astra_annotated_<stamp>.mp4` o `astra_raw_<stamp>.mp4`. `cancel()` borra el temporal.
- En `main.cpp:287-318` se configuran 2 grabadores de 640x480 a 10 FPS (`RECORD_FPS`): uno anotado y otro *raw*.
- Se controlan por teclado (r/s para el anotado, o/p para el raw; `main.cpp:744-758`) o por el topic `/od/recording_cmd` con `start_both`, `start_raw`, `start_annotated`, `save` o `cancel`. `RecordingController` exige al menos 2 GB libres y publica `/od/recording_status_json` cada 500 ms (`recording_control.hpp:106-201`).
- Al salir, `shutdown()` guarda lo que esté grabando.

## od45-06. ¿Cuántos feeds se pueden grabar simultáneamente?
**Archivos:** src/od/src/main.cpp, src/od/src/monitor/recording_control.hpp, src/od/arm_cam/arm_cam.cpp, src/od/src/monitor/record.cpp

**Respuesta:** Por proceso, 2 a la vez: el anotado (con bboxes y overlay) y el raw. El comando `start_both` arranca los dos con el mismo *stamp* (`recording_control.hpp:121-146`). `trt_pipeline` lo hace con la cámara Astra (`main.cpp:287-318`) y `arm_cam_udp` con la del brazo (anotado de 416x416 y raw de 640x480; `arm_cam.cpp:300-340`). Si ambos procesos corren, son 4 grabaciones simultáneas (2 por cámara).

## od45-07. ¿En qué formato y resolución se realizan las inferencias?
**Archivos:** src/od/src/pre/pre.cpp, src/od/src/pre/pre.hpp, src/od/src/main.cpp, src/od/src/trt/trt.cpp

**Respuesta:**
- **Entrada al pipeline:** el frame llega en BGR a 640x480; si no tiene ese tamaño se redimensiona (`main.cpp:244-246, 345-346`).
- **Preproceso** (`pre.cpp`): resize a 416x416 (valor por defecto en `pre.hpp:6-7`), conversión BGR→RGB y paso a float32 dividiendo entre 255. No usa letterbox, así que estira la imagen de 4:3 a 1:1. Luego `split` y copia a un blob plano NCHW (1x3x416x416).
- **Inferencia:** TensorRT en float32 en el host. El engine puede ser FP16 o FP32 internamente, según `onnx_to_trt_ui.sh`.
- **Salidas:** `output_boxes` con `[N,4]` (x1, y1, x2, y2 normalizados entre 0 y 1) y `output_scores` con `[N,num_clases]`. Se reescalan a 640x480 en `post.cpp:158-165`.

## od45-08. ¿Cómo se transmite el feed de video?
**Archivos:** scripts/pc.sh, src/od/src/main.cpp, scripts/gstream_raw.sh, src/od/arm_cam/arm_cam.cpp, src/bridge/core/rgbout.py

**Respuesta:**
1. **Del robot al PC:** H.264 sobre RTP/UDP al puerto 5600. El emisor no está en el repo (**hipótesis:** el robot JetAuto/ROS1 codifica la Astra y la envía con `udpsink`).
2. **En el PC** (`scripts/pc.sh:27-35`): `udpsrc → rtpjitterbuffer latency=100 drop-on-latency=true → rtph264depay → h264parse → avdec_h264 → videoconvert/videoscale → BGR 640x480 → shmsink /tmp/udp_cam`.
3. **En OD** (`main.cpp:262-274`): `shmsrc socket-path=/tmp/udp_cam ! video/x-raw,format=BGR,640x480 ! appsink drop=true sync=false`, abierto con `cv::VideoCapture(..., CAP_GSTREAMER)`.
4. **Cámara del brazo:** lee UDP:5601 directamente con appsink y una cola `leaky` (`arm_cam.cpp:274-285`).

La imagen anotada solo se muestra localmente con `cv::imshow`. Hacia fuera no se envía video: se envían detecciones por ROS 2 y, a ROS1, mediante rosbridge (`rgbout.py`).

## od45-09. ¿Cómo está estructurado el main?
**Archivos:** src/od/src/main.cpp, src/od/src/config/local_config.cpp, src/od/src/post/post.cpp, src/od/src/monitor/bin_overlay.cpp

**Respuesta:** `main.cpp` sigue este orden:
1. **Helpers** (líneas 55-161): `loadClassNames`, `lrZone`, `bboxShape2d` y el parseo de `--video`/`--config`.
2. **Callback de distancia** (190-199), que guarda los datos en `g_track_distance`.
3. **Inicialización en `main()`** (201-330):
   - nodo `od_node` y executor;
   - 5 publishers y 1 subscriber;
   - `VideoTelemetry`, `loadLocalConfig`, las constantes (`CONF_THRESH` 0.30, `IOU_THRESH` 0.40, 640x480, `RECORD_FPS` 10), los nombres de clase, `TRTInfer`, `InferenceTelemetry` y `PostProcessor`;
   - `VideoCapture` (archivo de video o shm);
   - 2 `VideoRecorder` y `RecordingController`;
   - `BinOverlay` y `LiveRunMetrics`.
4. **Bucle `while (rclcpp::ok())`** (331-801):
   - leer frame;
   - movimiento de cámara con LK;
   - `preprocess` → `trt.infer` → `post.process_outputs`;
   - `spin_some` y expiración de distancias;
   - por cada track: continuidad, *target score*, dibujo, JSON, `Detection2D`;
   - publicar `target_hint`, `detection_frame` y `bin_alignment`;
   - grabar, teclado, `imshow` y métricas.
5. **Cierre** (803-806).

## od45-10. ¿Qué tracker utiliza OD?
**Archivos:** src/od/src/post/post.cpp, src/od/src/post/tracker_model.cpp, src/od/src/post/track_reviver.cpp, src/od/src/post/post.hpp, src/od/src/post/tracker_state.hpp, src/od/src/post/camera_motion.cpp

**Respuesta:** Es un tracker propio de tipo "ByteTrack-lite" (`post.cpp:8-10`):
- **Dos etapas de asociación:** primero con detecciones de confianza alta (≥0.30) y después con detecciones bajas (0.08–0.30), que solo sirven para mantener vivos los tracks que ya existen (`post.cpp:355-434`).
- **Asociación greedy por detección** (no usa el algoritmo húngaro). Tiene gates adaptativos de distancia, IoU y ratio de área, y siempre exige la misma clase (`post.cpp:271-353`).
- **Puntuación del match:** `0.65*IoU + 0.35*cercanía`.
- **Modelo de movimiento** "Kalman-lite" con velocidad EMA (`tracker_model.cpp`).
- **`TrackReviver`**, que revive tracks muertos (`track_reviver.cpp`).
- **Compensación del movimiento global de cámara** con flujo óptico (`camera_motion.cpp`).
- NMS final entre tracks.

## od45-11. ¿De qué color son los bounding boxes?
**Archivos:** src/od/src/main.cpp, src/od/src/monitor/bin_overlay.cpp, src/od/arm_cam/arm_cam.cpp

**Respuesta:**
- **Bboxes y etiquetas:** verde, BGR `{0,255,0}`, con grosor 2 (`main.cpp:490`; el texto en `main.cpp:613-621`).
- **Punto central:** amarillo, `{0,255,255}` (`main.cpp:491`).
- **Overlay del bin** (cuando está activo): caja amarilla `{0,255,255}`, línea de soltado magenta `{255,0,255}`, rango del brazo azul `{255,0,0}` y tolerancia cian `{255,255,0}` (`bin_overlay.cpp:201-301`).
- **Textos de grabación:** "REC" en rojo y "RAW REC" en naranja.

## od45-12. ¿Cómo se activa el overlay?
**Archivos:** src/od/src/main.cpp, src/od/src/monitor/bin_overlay.cpp, src/od/src/monitor/bin_overlay.hpp

**Respuesta:**
- **Activación:** con la tecla `b` o `B` en la ventana de OpenCV, que llama a `bin_overlay.toggle()` (`main.cpp:753-755`). Arranca desactivado (`enabled_(false)`, `bin_overlay.cpp:40`).
- **Lo que se ejecuta siempre:** `draw()` calcula el estado de alineación aunque el overlay esté apagado y se publica en `/bin_alignment`. El campo `enabled` indica si el overlay está activo (`main.cpp:697-714`).
- **Lo que depende del overlay:** solo el dibujo (`bin_overlay.cpp:197-199`).

No hay ningún topic ni parámetro para activarlo; solo la tecla.

## od45-13. ¿De dónde obtiene el overlay la información que muestra?
**Archivos:** src/od/src/monitor/bin_overlay.cpp, src/od/src/main.cpp, src/od/src/post/post.cpp

**Respuesta:** El overlay toma los tracks de salida del tracker (`post.process_outputs`). En `main.cpp:426-433` se convierten en `BinOverlayDetection` (clase, caja y score) y se pasan a `bin_overlay.draw(frame, overlay_detections)` (`main.cpp:697-698`).

Dentro de `draw()` (`bin_overlay.cpp:93-195`):
- elige el mejor `"bin"` (confianza, más un bonus por IoU con la caja suavizada anterior);
- suaviza con EMA (alpha 0.85, o 0.95 si el área salta más de un 35 %);
- compara el centro con `drop_line_x_=320`, con `arm_range_px_=90` y `center_tolerance_px_=35`, para producir `BIN_BASE_LEFT/RIGHT`, `BIN_ARM_LEFT/RIGHT`, `BIN_CENTER` o `BIN_LOST`.

Los valores fijos (línea en x=320, rangos) están en el constructor.

## od45-14. ¿Cómo utiliza OD OpenCV para generar y procesar las imágenes?
**Archivos:** src/od/src/main.cpp, src/od/src/pre/pre.cpp, src/od/src/post/camera_motion.cpp, src/od/src/monitor/bin_overlay.cpp, src/od/src/monitor/record.cpp, src/od/src/post/depth_mask.cpp

**Respuesta:**
- **Captura:** `cv::VideoCapture` con backend GStreamer o con un archivo (`main.cpp:267-274`).
- **Preparación del frame:**
  - `cv::resize` a 640x480;
  - `cvtColor` a gris para el movimiento de cámara;
  - `frame.clone()` para el raw.
- **Preproceso:** `resize` a 416, `cvtColor` BGR2RGB, `convertTo` a float y `split` a CHW (`pre.cpp`).
- **Movimiento de cámara:** `goodFeaturesToTrack` (300 puntos) y `calcOpticalFlowPyrLK`, y luego la mediana de dx/dy (`camera_motion.cpp`).
- **Dibujo:** `rectangle`, `circle`, `putText`, `line` y `getTextSize` (`main.cpp`, `bin_overlay.cpp`).
- **Visualización:** `imshow`, más `waitKey` para el teclado.
- **Grabación:** `VideoWriter` mp4v (`record.cpp`).
- **`depth_mask.cpp`:** usa `morphologyEx` y `moments`, pero no se llama desde `main`.

## od45-15. ¿Por qué se mueven tanto los bounding boxes?
**Archivos:** src/od/src/post/tracker_model.cpp, src/od/src/post/post.cpp, src/od/src/post/camera_motion.cpp, src/od/src/main.cpp, src/od/src/pre/pre.cpp

**Respuesta (parte hipótesis):**
1. **YOLO tiembla de un frame a otro y no hay letterbox.** Al estirar 640x480 a 416x416 se amplifica el error vertical (`pre.cpp:7`). Además, `int` trunca las coordenadas (`post.cpp:162-165`).
2. **El tracker mueve la caja aunque no haya detección nueva:**
   - `predict_track` le suma la velocidad en cada frame (`tracker_model.cpp:20-23`);
   - la velocidad sale de una EMA con `vel_gain=0.55` que reacciona fuerte al ruido de medición (`tracker_model.cpp:46-47`);
   - con `vel_damp=0.9` la caja sigue "deslizándose" varios frames.
3. **La compensación de movimiento de cámara** desplaza todos los tracks según la mediana del flujo LK (`post.cpp:220-230`). Si el objeto o la escena dominan los puntos, o hay poca textura, ese desplazamiento es erróneo; el umbral es solo de 1.5 px y el clamp de ±80 px (`camera_motion.cpp:76-81`).
4. **Las detecciones de baja confianza** también corrigen la caja (`post.cpp:404-434`).
5. **Los tracks con `missed>0`** se siguen dibujando con la posición predicha (`post.cpp:461-491`).

## od45-16. ¿Cómo podemos evitar que se pierda el tracking cuando un objeto gira?
**Archivos:** src/od/src/post/post.cpp, src/od/src/post/track_reviver.cpp, src/od/src/post/tracker_model.cpp, src/od/src/post/post.hpp

**Respuesta (propuesta, hipótesis):** Al girar, cambian el aspecto, el área y a veces la clase predicha, y el match falla por estas razones:
- **Gate de ratio de área:** `AREA_RATIO_MAX` vale 1.70 y llega como máximo a 3.0 (`post.cpp:299-309, 340-345`).
- **Gate de IoU:** exige `iou_v ≥ gate` o estar cerca del centro (`post.cpp:337-338`).
- **Clase estricta:** `t.cls != d.cls` → `false` (`post.cpp:313`).

Opciones:
- relajar `AREA_RATIO_MAX` y el gate de aspecto (`post.hpp:58-64`);
- permitir un cambio de clase con penalización (o usar la clase por votación del track) en lugar de rechazarlo;
- subir `MAX_MISSED_FRAMES` y el TTL/`DIST_PX` del reviver (`track_reviver.hpp:39-42`);
- bajar `LOW_CONF_THRESH`, para que las detecciones débiles durante el giro mantengan el track;
- a futuro, añadir un descriptor de apariencia (re-ID, al estilo DeepSORT/BoT-SORT).

## od45-17. ¿Cómo podemos cambiar el color de los bounding boxes dependiendo de su clase?
**Archivos:** src/od/src/main.cpp, src/od/src/post/post.hpp

**Respuesta:** El color está fijo en `cv::rectangle(frame, r, {0,255,0}, 2)` (`main.cpp:490`) y en el `putText` de la etiqueta (`main.cpp:613-621`). En el bucle ya están disponibles `t.cls` y `class_name` (`main.cpp:410`).

Para cambiarlo:
1. Crear una paleta `static const std::vector<cv::Scalar> kPalette{...}` o un `std::unordered_map<std::string, cv::Scalar>` (por ejemplo `bin` → amarillo, `bottle` → azul).
2. Calcular `const cv::Scalar color = kPalette[t.cls % kPalette.size()];`.
3. Usar `color` en `rectangle` y en `putText`.

Si se quiere sin recompilar, se puede leer el color de un archivo (como `local.yaml`) en `local_config.cpp`. **Hipótesis:** hoy no existe esa clave.

## od45-18. ¿Qué ocurre desde que la cámara captura una imagen hasta que OD detecta un objeto?
**Archivos:** scripts/pc.sh, src/od/src/main.cpp, src/od/src/pre/pre.cpp, src/od/src/trt/trt.cpp, src/od/src/post/post.cpp, src/od/src/post/tracker_model.cpp

**Respuesta:**
1. El robot codifica la imagen en H.264/RTP y la envía a UDP:5600 (**hipótesis** sobre el emisor).
2. `gst-launch` aplica el jitterbuffer de 100 ms, decodifica con `avdec_h264`, convierte a BGR 640x480 y lo escribe en `shmsink /tmp/udp_cam` (`pc.sh:27-35`).
3. `cap.read` lee desde `shmsrc`/appsink (`main.cpp:336`) y redimensiona si hace falta.
4. Se estima el movimiento de cámara con LK (`main.cpp:357-365`).
5. `preprocess` genera el blob 416x416 RGB/255 en NCHW (`pre.cpp`).
6. `trt.infer`: copia H2D, `enqueueV3` y copia D2H (`trt.cpp:121-145`).
7. `process_outputs` (`post.cpp:114-495`):
   - toma el argmax de clase por caja y descarta lo que esté por debajo de 0.08;
   - escala a píxeles y filtra cajas menores de 10 px;
   - aplica NMS por clase con IoU 0.40;
   - separa alta y baja confianza;
   - compensa el movimiento de cámara y predice;
   - asocia, revive o crea tracks;
   - marca los no vistos (`missed`) y aplica NMS de tracks.
8. `main` dibuja y publica `/detections_2d` y los JSON.

## od45-19. ¿Qué componentes se inician cuando arrancamos OD?
**Archivos:** scripts/pc.sh, src/od/launch/od.launch.py, src/od/src/main.cpp, src/od/src/trt/trt.cpp, src/od/src/config/local_config.cpp, src/bridge/core/rgbout.py

**Respuesta:**
- **Con `scripts/pc.sh`** (tmux): `gst-launch` (UDP→shm), `sleep 3`, `ros2 run od trt_pipeline`, `ros2 run bridge rgbout` y `ros2 run bridge rangein`.
- **Con `scripts/orin.sh`:** lo mismo, pero con `bridge r2` (profundidad) en lugar de rgbout.
- **Con `od.launch.py`:** `trt_pipeline` y, cuando ya arrancó, `rgbout`.

Dentro de `trt_pipeline` (`main.cpp:203-330`):
- `rclcpp` y el nodo `od_node`;
- publishers de `/detections_2d`, `/object_distance_json`, `/od/detection_frame_json`, `/od/target_hint` y `/bin_alignment`;
- el subscriber de `/object_distance`;
- `VideoTelemetry`, `LocalConfig` y los nombres de clase;
- `TRTInfer` (stream CUDA, engine y buffers), `InferenceTelemetry` y `PostProcessor`;
- `VideoCapture`, 2 `VideoRecorder` y `RecordingController` (publisher y subscriber de grabación);
- `BinOverlay`, `LiveRunMetrics` (con `GPUMonitor`/NVML, **hipótesis** por el include de `monitor.hpp`) y la ventana `imshow`.

## od45-20. ¿Qué parámetros podemos cambiar sin recompilar el programa?
**Archivos:** src/od/src/config/local_config.cpp, src/od/src/config/local_config.hpp, src/od/config/local.example.yaml, src/od/src/main.cpp, src/od/arm_cam/arm_cam.cpp, scripts/pc.sh

**Respuesta:**
- **En `local.yaml`:**
  - las claves son `engine_path`, `names_path`, `arm_engine_path`, `arm_names_path` y `video_directory` (`local_config.cpp:54-70`);
  - la ruta se elige con `--config <ruta>`, la variable de entorno `OD_CONFIG` o la lista de rutas candidatas (`local_config.cpp:72-91`).
- **Por línea de comandos:** `--video <archivo>` (o una ruta `.mp4/.avi/.mkv/.mov`) para el modo offline (`main.cpp:115-146`).
- **Contenido de `obj.names`:** define las clases.
- **El engine mismo:** se puede cambiar de modelo regenerando el `.engine` con el mismo formato de entrada y salida.
- **Pipeline GStreamer en `pc.sh`:** puerto, latencia del jitterbuffer, resolución.
- **En `arm_cam_udp`:** los parámetros ROS `contact.*` (`arm_cam.cpp:208-219`).
- **En tiempo de ejecución:** las teclas b, r, s, o, p y Esc, y el topic `/od/recording_cmd`.

En cambio, `CONF_THRESH`, `IOU_THRESH`, la resolución, los FPS de grabación y todos los parámetros del tracker y del overlay son constantes en el código y **requieren recompilar**.

## od45-21. ¿Qué sucede si falta un archivo de configuración?
**Archivos:** src/od/src/config/local_config.cpp, src/od/src/config/local_config.hpp, src/od/src/main.cpp, src/od/src/trt/trt.cpp, src/od/src/post/post.cpp

**Respuesta:**
- **Si falta `local.yaml`:** `loadLocalConfig` prueba todas las rutas candidatas. Si no encuentra ninguna, imprime `[OD_CONFIG] No se encontro config/local.yaml; usando rutas por defecto.` (`local_config.cpp:137`) y usa los valores fijos de `local_config.hpp:7-16` (`/home/adjor/Computer_Vision/darknet_1/yolov4_4C/4C_14.engine`, `obj.names`, `/home/adjor/Videos`).
  - Las líneas sin `:` y las claves desconocidas solo generan un aviso.
  - Si se pasa `--config` con una ruta inexistente, también se usan los valores por defecto.
- **Si falta el `.engine`:** `TRTInfer` lanza `runtime_error("No se pudo abrir engine")` (`trt.cpp:52-53`). Nadie la captura, así que el proceso aborta.
- **Si falta `obj.names`:** `classNames` queda vacío y `process_outputs` devuelve `{}` (`post.cpp:134-135`). OD corre sin detectar nada y sin avisar.

## od45-22. ¿Cómo se comunican entre sí las diferentes partes de OD?
**Archivos:** src/od/src/main.cpp, scripts/pc.sh, src/bridge/core/r2.py, src/bridge/core/rgbout.py, src/od/src/monitor/recording_control.hpp, src/od/msg/Detection2D.msg

**Respuesta:**
1. **Dentro de `trt_pipeline`:** llamadas directas a funciones en un solo hilo (`preprocess` → `TRTInfer::infer` → `PostProcessor::process_outputs` → `BinOverlay::draw` → `VideoRecorder::write`). El movimiento de cámara pasa por `post.setCameraMotion` y la distancia por el mapa global `g_track_distance`.
2. **Entre GStreamer y OD:** memoria compartida `/tmp/udp_cam` (`shmsink`/`shmsrc`).
3. **Con otros nodos:** topics de ROS 2 (`Detection2D`, `Distance` y `String` con JSON). Los callbacks se procesan con `exec.spin_some()` dentro del bucle (`main.cpp:390`).
4. **Con ROS1 (robot):** rosbridge por websocket en `192.168.68.71:9091`, con `roslibpy` (`rgbout.py`, `rangein.py`).

## od45-23. ¿Por qué OD tarda tanto en mostrar la imagen al iniciarse?
**Archivos:** src/od/src/trt/trt.cpp, scripts/pc.sh, src/od/src/main.cpp, src/od/src/monitor/live_run_metrics.cpp

**Respuesta (hipótesis razonada):**
- `pc.sh` hace `sleep 3` antes de lanzar `trt_pipeline` (`pc.sh:40`).
- Al arrancar, `TRTInfer` deserializa el engine, crea el contexto CUDA y reserva memoria (`trt.cpp:46-100`). Eso suele tardar unos segundos, sobre todo la inicialización de CUDA en el primer uso.
- La primera inferencia es lenta (*warmup*); `LiveRunMetrics` descarta los primeros 10 frames (`main.cpp:324-329`).
- El video necesita un keyframe H.264 para poder decodificarse, y el jitterbuffer añade 100 ms. Si `gst` aún no escribió en `/tmp/udp_cam`, `cap.read` falla y el bucle hace `continue` sin llamar a `imshow` (`main.cpp:336-343`).
- La ventana solo aparece en el primer `cv::imshow` (`main.cpp:784`).

## od45-24. ¿Qué sucede si la cámara se desconecta mientras OD está funcionando?
**Archivos:** src/od/src/main.cpp, src/od/src/monitor/video_telemetry.hpp, src/od/src/monitor/live_run_metrics.cpp, scripts/pc.sh

**Respuesta:** En modo en vivo, `cap.read` falla o devuelve un frame vacío. Entonces:
- se registran `video_telemetry.readFailure()` y `run_metrics.readFailure()`, se hace `spin_some()` y `continue` (`main.cpp:336-343`);
- no hay reconexión, ni timeout, ni espera: el bucle gira intentando leer;
- la ventana se queda congelada en el último frame porque `imshow` y `waitKey` no se llaman, así que el teclado (Esc) tampoco responde;
- no se publican detecciones;
- `/od/video_status_json` muestra `alive:false` y un `frame_read_failure_percent` alto (`video_telemetry.hpp:85-116`).

En modo offline (archivo), el fallo de lectura hace `break` y el programa termina.

**Hipótesis:** como `shmsrc` es `is-live`, si `gst-launch` sigue vivo pero ya no llegan paquetes UDP, `read` podría bloquearse en lugar de fallar.

## od45-25. ¿Qué pasa si la cámara entrega imágenes más rápido de lo que OD puede procesarlas?
**Archivos:** src/od/src/main.cpp, scripts/pc.sh, src/od/arm_cam/arm_cam.cpp, src/od/src/monitor/video_telemetry.hpp

**Respuesta:** Se descartan frames en lugar de acumularlos:
- el appsink de OD usa `drop=true sync=false` (`main.cpp:265`);
- `shmsink` usa `sync=false` y `wait-for-connection=false`;
- el `rtpjitterbuffer` usa `drop-on-latency=true` (`pc.sh:30-35`);
- en el brazo, `queue max-size-buffers=1 leaky=downstream` y `appsink max-buffers=1 drop=true` (`arm_cam.cpp:281-282`).

El bucle es secuencial, así que el FPS efectivo queda limitado por `pre + infer + post + dibujo` y se procesa el frame más reciente disponible. **Hipótesis:** el appsink sin `max-buffers=1` podría encolar algunos frames y añadir latencia.

Las grabaciones duplican frames para mantener los 10 FPS nominales (`record.cpp:63-82`). `/od/video_status_json` reporta el `annotated_fps` real frente al `configured_fps` de 30.

## od45-26. ¿Por qué el video se ve fluido, pero los bounding boxes aparecen retrasados?
**Archivos:** src/od/src/post/tracker_model.cpp, src/od/src/post/tracker_state.hpp, src/od/src/post/post.cpp, src/od/src/main.cpp, src/bridge/core/rgbout.py

**Respuesta:** En la ventana de OD, la caja se dibuja sobre el mismo frame que se infirió, así que el retraso no viene de la sincronización. Viene del **suavizado del bbox**:
- `box_alpha = 0.85` (`tracker_model.hpp:16`): cada frame solo aporta un 15 % de la medición nueva (`tracker_model.cpp:66-70`), y la caja tarda unos 6 frames en alcanzar el 63 % de un movimiento;
- el tamaño también se suaviza (`size_alpha` 0.70);
- en baja confianza el suavizado es aún mayor (`a ≈ 0.86`);
- la caja que se dibuja es `bx1..by2` (la suavizada), no la detección cruda (`post.cpp:465-469`).

Además, aunque el centro `t.cx` es la medición, la caja va detrás. **Hipótesis:** en los consumidores remotos (ROS1/rviz) se suma la latencia de rosbridge (`rgbout.py`) y de la red.

## od45-27. ¿Cómo podemos identificar dónde se está produciendo el retraso del video?
**Archivos:** src/od/src/monitor/live_run_metrics.cpp, src/od/src/monitor/video_telemetry.hpp, src/od/src/monitor/inference_telemetry.hpp, src/od/src/main.cpp, src/od/src/benchmark.cpp, scripts/pc.sh

**Respuesta:** OD ya instrumenta cada etapa:
- **Por frame:** `preprocess_ms`, `out.gpu_ms` (eventos CUDA solo alrededor de `enqueueV3`), `postprocess_ms` y `pipeline_ms`, que se pasan a `run_metrics.frame` (`main.cpp:355-388, 786-798`). `LiveRunMetrics` escribe al final un JSON `od_live_<ts>.json` en `<video_directory>/EMMA/metrics` con media, p95 y máximo (`live_run_metrics.cpp:214, 369`).
- **`/od/video_status_json`:** FPS anotado, jitter entre frames, latencia de procesamiento media y máxima, y porcentaje de fallos de lectura.
- **`/od/inference_status_json`:** media, p95 y máximo de la GPU, y FPS de inferencia.
- **`od_benchmark`:** mide el engine aislado, offline.

Método: comparar el FPS de la cámara frente al FPS anotado, `gpu_ms` frente a `pipeline_ms`, y la diferencia entre `processing_ms` y la suma de etapas (captura, dibujo y `imshow`).

Lo que no se mide es la latencia de red, la del decodificador y la del jitterbuffer (100 ms en `pc.sh:30`). Para verlas se puede usar `gstream_raw.sh` (video sin OD) o añadir timestamps de captura (**hipótesis**).

## od45-28. ¿De dónde obtiene OD los nombres de las clases que detecta?
**Archivos:** src/od/src/main.cpp, src/od/src/config/local_config.cpp, src/od/src/config/local_config.hpp, src/od/src/post/post.cpp

**Respuesta:** Del archivo `obj.names` (formato Darknet, una clase por línea) indicado en `names_path` de `local.yaml`. El valor por defecto es `/home/adjor/Computer_Vision/darknet_1/yolov4_4C/obj.names` (`local_config.hpp:9-10`).
- `loadClassNames` lo lee e ignora las líneas vacías (`main.cpp:55-64, 249`).
- El índice de clase es el argmax de `output_scores` (`post.cpp:150-155`) y se traduce con `classNames[t.cls]` (`main.cpp:410`).
- El número de clases (`class_names.size()`) se usa también para interpretar el tensor de scores (`post.cpp:134-143`), así que debe coincidir con el modelo.
- `arm_cam` usa su propio `arm_names_path`.

## od45-29. ¿Por qué OD detecta objetos que realmente no existen?
**Archivos:** src/od/src/post/post.cpp, src/od/src/post/post.hpp, src/od/src/post/tracker_model.cpp, src/od/src/main.cpp, src/od/src/post/track_reviver.cpp

**Respuesta (parte hipótesis):**
- `CONF_THRESH=0.30` es relativamente bajo (`main.cpp:241`).
- `MIN_SEEN_FRAMES=1` (`post.hpp:63`): una sola detección en un único frame ya crea un track y se dibuja.
- **Tracks fantasma:** cuando el objeto desaparece, el track sigue saliendo hasta 10 frames (`MAX_MISSED_FRAMES`) con la posición predicha. La salida no filtra `missed>0` (`post.cpp:461-491`), así que la caja se sigue dibujando y publicando en `/detections_2d`.
- Las detecciones débiles (≥0.08) mantienen vivos los tracks (`post.cpp:404-434`).
- El score tiene inercia (`max(score*0.9, det)`): se queda alto aunque la confianza real baje.
- El reviver puede "revivir" un ID sobre otra detección cercana de la misma clase.
- **Hipótesis:** el dataset y las 4 clases del modelo (`yolov4_4C`) producen falsos positivos en fondos similares. El estiramiento sin letterbox y los cambios de iluminación agravan el problema.

## od45-30. ¿Por qué le cuesta detectar objetos pequeños o que están lejos?
**Archivos:** src/od/src/pre/pre.cpp, src/od/src/post/post.cpp, src/od/src/main.cpp, src/bridge/core/r2.py

**Respuesta:**
- **Resolución:** el frame de 640x480 se reduce a 416x416 (`pre.cpp:7`). Un objeto lejano ocupa muy pocos píxeles de entrada y se deforma por el cambio de aspecto.
- **Filtros duros de tamaño:**
  - se descartan las cajas de menos de 10 px de ancho o alto (`post.cpp:167`);
  - en la salida, las de menos de 5 px (`post.cpp:482`).
- **Confianza:** los objetos pequeños suelen tener baja confianza. Entre 0.08 y 0.30 nunca crean track, solo mantienen los existentes (`post.cpp:211-215, 393-396`).
- **Video:** la compresión H.264 degrada los detalles finos.
- **Distancia:** `r2.py` limita la profundidad a 0.15–3.0 m (`r2.py:36-37`).
- **Hipótesis:** YOLOv4 a 416 y un dataset con pocos ejemplos lejanos.

## od45-31. ¿Qué sucede cuando cambia bruscamente la iluminación?
**Archivos:** src/od/src/post/camera_motion.cpp, src/od/src/pre/pre.cpp, src/od/src/post/post.cpp, src/od/src/post/tracker_model.cpp

**Respuesta (mayormente hipótesis):**
- **Sin normalización fotométrica:** OD no la aplica (solo divide entre 255, `pre.cpp:9`). La confianza del detector cae o cambia, y las detecciones pasan a la rama de baja confianza o desaparecen. Entonces los tracks sobreviven hasta 10 frames por predicción y después mueren o se reviven.
- **Error en la estimación de movimiento:** el flujo óptico LK (`camera_motion.cpp:35-49`) asume brillo constante. Un cambio brusco puede generar un desplazamiento global falso y mover todos los tracks (hasta ±80 px) en `post.cpp:220-230`, lo que provoca saltos o pérdidas de asociación.
- **Hipótesis:** la exposición automática de la cámara tarda unos frames en estabilizarse.

## od45-32. ¿Cómo decide OD cuáles detecciones descartar por tener poca confianza?
**Archivos:** src/od/src/post/post.cpp, src/od/src/main.cpp, src/od/src/post/tracker_model.cpp

**Respuesta:** El score de cada caja es el máximo de su fila de scores de clase (`post.cpp:146-155`). Con ese valor:
- **< 0.08** (`LOW_CONF_THRESH`, `post.cpp:10, 156`): se descarta.
- **Entre 0.08 y 0.30** (`CONF_THRESH`, `main.cpp:241`): se marca `is_low`. Solo puede asociarse a un track ya existente, con gates más estrictos y una ganancia reducida (×0.35). No crea tracks ni incrementa `seen` (`post.cpp:400-434`; `tracker_model.cpp:79-110`).
- **≥ 0.30:** es una detección alta. Asocia, revive o crea tracks.

Además:
- se aplica NMS por clase con IoU 0.40 (`post.cpp:182-206`) y NMS de tracks con IoU 0.60 (`post.cpp:494`);
- el target hint exige `t.score >= CONF_THRESH` (`main.cpp:466-472`);
- el score mostrado se redondea a pasos de 0.05 y vale 1.0 si es ≥0.95 (`post.cpp:18-21`).

## od45-33. ¿Qué sucede cuando dos objetos se cruzan frente a la cámara?
**Archivos:** src/od/src/post/post.cpp, src/od/src/post/tracker_model.cpp, src/od/src/post/track_reviver.cpp

**Respuesta:**
- **Clases distintas:** no se confunden, porque el match exige la misma clase (`post.cpp:313`).
- **Misma clase:**
  - la asociación es greedy: cada detección alta toma el mejor track libre por `0.65*IoU + 0.35*cercanía` (`post.cpp:358-372`). No hay optimización global (húngaro), así que se pueden **intercambiar los IDs**;
  - durante la oclusión, NMS (IoU 0.40) puede fusionar las dos detecciones en una; el track ocluido acumula `missed` y sigue con su velocidad predicha;
  - en la salida, `suppress_overlapping_tracks` con IoU 0.60 oculta uno de los dos tracks superpuestos (`post.cpp:59-91, 494`);
  - al separarse, el track perdido puede reasociarse (gates que crecen con `missed`), revivirse o recibir un ID nuevo.
- **En `main`:** `continuity` y la distancia se reinician para los tracks no observados (`main.cpp:421-424`).

## od45-34. ¿Por qué a veces un mismo objeto recibe diferentes identificadores?
**Archivos:** src/od/src/post/post.cpp, src/od/src/post/track_reviver.cpp, src/od/src/post/track_reviver.hpp, src/od/src/post/post.hpp

**Respuesta:** Se asigna un ID nuevo (`next_id_++`, `post.cpp:244`) cuando una detección alta no encuentra track y el reviver tampoco la recupera. Ocurre si:
1. **Pasó demasiado tiempo:** el track lleva más de 10 frames sin verse y, además, pasaron más de 15 frames en el buffer de muertos (`TTL_FRAMES`, `track_reviver.hpp:39`).
2. **Se movió demasiado:** más de 180 px (`DIST_PX`), o más que el gate adaptativo.
3. **Cambió de tamaño o forma:** la caja varió más que `AREA_RATIO`, por zoom o por un giro.
4. **Cambió la clase predicha:** el match exige la misma clase (`post.cpp:313`; `track_reviver.cpp:70`).
5. **Se superó el límite de tracks:** con más de 30, `prune_tracks_global` elimina tracks de baja prioridad sin pasarlos al reviver (`post.cpp:96-109`).
6. **Hubo un cruce** con otro objeto de la misma clase (asociación greedy).
7. **Hubo un salto de la cámara** que la compensación por flujo óptico no corrigió bien.

## od45-35. ¿Qué ocurre cuando un objeto sale de la imagen y después regresa?
**Archivos:** src/od/src/post/track_reviver.cpp, src/od/src/post/post.cpp, src/od/src/post/track_reviver.hpp, src/od/src/main.cpp

**Respuesta:**
1. **Mientras falta:** el track suma `missed` en cada frame. Sigue saliendo con la posición predicha (con la caja recortada al borde de la imagen) hasta `MAX_MISSED_FRAMES=10` (`post.cpp:439-450`).
2. **Al morir:** pasa a `reviver_.push_dead()`, donde vive `TTL_FRAMES=15` frames (`track_reviver.cpp:24-34`). Ahí también recibe la compensación de movimiento de cámara.
3. **Si regresa dentro de esos ~25 frames** (unos 0.8 s a 30 FPS), con la misma clase, a menos de 180 px y con un área compatible (×3): se **revive con el mismo ID**. La caja se reinicia con la detección y la velocidad se re-estima (`track_reviver.cpp:53-143`).
4. **Si no:** recibe un ID nuevo.

En `main`, cuando el ID desaparece de la salida se borran su `track_continuity` y la distancia (`main.cpp:636-643`), así que el contador `stable_frames` vuelve a empezar.

## od45-36. ¿Cómo podemos suavizar los bounding boxes sin introducir demasiado retraso?
**Archivos:** src/od/src/post/tracker_model.cpp, src/od/src/post/tracker_model.hpp, src/od/src/post/post.cpp, src/od/src/monitor/bin_overlay.cpp

**Respuesta (propuesta):** Hoy se usa una EMA fija con `box_alpha=0.85` y `size_alpha=0.70` (`tracker_model.hpp:11-17`), que suaviza mucho pero retrasa la caja. Alternativas:
- **Bajar los alphas** (por ejemplo `box_alpha` a 0.5–0.65) y ajustar según la telemetría.
- **Alpha adaptativo** según la innovación o la velocidad: poco suavizado cuando el objeto se mueve y mucho cuando está quieto. Es la misma idea que `jump_bbox_alpha_` en `bin_overlay.cpp:140-142`, pero al revés. El **One Euro Filter** es la versión clásica.
- **Suavizar el tamaño y no el centro:** usar el centro filtrado más la velocidad (predicción), y aplicar la EMA solo a w/h.
- **Kalman real** (por ejemplo `cv::KalmanFilter` con estado `[cx, cy, w, h, v...]`), con ruido de medición escalado por la confianza: ajusta la ganancia automáticamente.
- **Separar dos salidas:** la caja cruda para el control (`/detections_2d`, profundidad) y la suavizada solo para la visualización.

## od45-37. ¿Cómo determina OD cuándo debe dejar de seguir un objeto?
**Archivos:** src/od/src/post/post.cpp, src/od/src/post/post.hpp, src/od/src/post/track_reviver.cpp, src/od/src/post/tracker_model.cpp

**Respuesta:**
- **Contador `missed`:** sube en cada frame sin asociación, alta o baja (`post.cpp:442`). Cualquier match, incluso uno de baja confianza, lo pone a 0 (`tracker_model.cpp:113`).
- **Fin del track:** cuando `missed > MAX_MISSED_FRAMES` (10, u 8 en modo brazo; `post.hpp:48-54, 62`), el track sale de `tracks_` y pasa al reviver.
- **Fin definitivo:** en el reviver se elimina tras `TTL_FRAMES=15` (`track_reviver.cpp:24-34`).
- **Otras salidas:**
  - con más de `MAX_TRACKS_TOTAL=30` tracks se podan los de menor prioridad, calculada como `score*(1+seen/5) - missed*0.2` (`post.cpp:96-109`);
  - en la salida se ocultan los tracks con caja menor de 5 px o no visibles por el NMS de tracks.

## od45-38. ¿Cómo calcula OD la distancia entre la cámara y un objeto?
**Archivos:** src/bridge/core/r2.py, src/od/src/main.cpp, src/od/msg/Distance.msg, src/od/msg/Detection2D.msg, scripts/orin.sh

**Respuesta:** OD no calcula la distancia. Lo hace el nodo `depth_from_detections_multi` (`src/bridge/core/r2.py`, que se lanza desde `orin.sh`) así:
1. Se suscribe a `/detections_2d`, `/camera/depth/image_raw` y `/camera/depth/camera_info` (`r2.py:53-72`).
2. Convierte la profundidad de mm a m (×0.001) y usa las detecciones con menos de 0.25 s de antigüedad.
3. Para cada bbox:
   - recorta un 25 % por cada lado (*center crop*);
   - toma los píxeles válidos entre 0.15 y 3.0 m;
   - calcula **z = percentil 15**;
   - calcula u, v como la media de los píxeles válidos y `x = (u - cx)*z/fx - 0.03` (offset del gripper), el ángulo y el lado (`r2.py:150-245`).
4. Publica solo el objeto **más cercano** en `/object_distance` (`Distance`) y en `/object_distance_json`.

OD lo recibe en `distanceCallback` (`main.cpp:193-199`) y lo muestra como `" X.XXm"` en la etiqueta.

`src/od/src/post/depth_mask.cpp` (máscara por profundidad y centroide) existe, pero no se usa.

## od45-39. ¿Qué sucede cuando hay detecciones, pero no llegan datos de profundidad?
**Archivos:** src/od/src/main.cpp, src/bridge/core/r2.py

**Respuesta:**
- Si no llegan mensajes `/object_distance`, `g_track_distance` está vacío o caduca a los 350 ms (`DISTANCE_TIMEOUT`, `main.cpp:190, 392-398`). Entonces `distance=-1`:
  - la etiqueta no muestra metros;
  - el JSON por detección lleva `distance_available:false` y `distance_m:null` (`main.cpp:568-579`);
  - OD no publica su `/object_distance_json` (`main.cpp:586-611`).
- `/detections_2d`, `/od/detection_frame_json` y `/od/target_hint` se siguen publicando con normalidad (el target hint no depende de la profundidad).
- En `r2.py`, si no hay intrinsics (`fx`) o detecciones, `depth_cb` retorna sin publicar (`r2.py:120-124`).
- Como `r2.py` solo publica el objeto más cercano, los demás tracks nunca tienen distancia.

## od45-40. ¿Cómo relaciona OD la información de profundidad con cada bounding box?
**Archivos:** src/od/src/main.cpp, src/bridge/core/r2.py, src/od/msg/Distance.msg, src/od/msg/Detection2D.msg

**Respuesta:** La relación se hace por **`track_id`**:
1. OD publica `Detection2D` con `track_id` y la caja en píxeles de 640x480 (`main.cpp:623-633`).
2. `r2.py` usa esas coordenadas directamente sobre la imagen de profundidad, asumiendo la misma resolución y alineación (**hipótesis:** no hay registro ni reescalado entre RGB y profundidad, solo un clamp). Devuelve `Distance{track_id, class_name, distance}`.
3. `distanceCallback` guarda `{metros, clase, hora}` en `g_track_distance[track_id]` (`main.cpp:193-199`).
4. En el bucle, la distancia solo se usa si se cumplen tres condiciones (`main.cpp:534-547`): el track se observó en este frame (`missed==0`), la clase coincide y el dato tiene menos de 350 ms.
5. Si el track deja de observarse o desaparece, la distancia se borra (`main.cpp:421-424, 636-643`).

## od45-41. ¿Qué información envía OD a los demás nodos de ROS 2?
**Archivos:** src/od/src/main.cpp, src/od/msg/Detection2D.msg, src/od/src/monitor/recording_control.hpp, src/od/src/monitor/video_telemetry.hpp, src/od/src/monitor/inference_telemetry.hpp, src/od/arm_cam/arm_cam.cpp

**Respuesta:** Desde `trt_pipeline` (nodo `od_node`):
- **`/detections_2d`** (`od/Detection2D`, uno por track y por frame): `track_id`, `class_name`, `confidence`, `xmin/ymin/xmax/ymax`, `cx/cy`.
- **`/object_distance_json`** (String JSON, solo si hay distancia): distancia, `x_offset` normalizado, id, clase, `stable_frames`, edad del dato, `cx/cy`, `bbox_w/h/area`, `aspect_ratio` y forma 2D (`main.cpp:586-611`). **Ojo:** `r2.py` publica en el mismo topic con otro esquema.
- **`/od/detection_frame_json`:** el frame completo, con `frame_id`, timestamp, tamaño, `target_hint` y la lista de detecciones con scores visuales y de distancia.
- **`/od/target_hint`** (solo si es válido): el mejor objetivo por `visual_score = 0.6*bottom + 0.2*center + 0.1*conf + 0.1*stability`, excluyendo `bin` (`main.cpp:461-488, 645-673`).
- **`/bin_alignment`:** `enabled`, `status`, `offset`, `ready`, `found`, `cx`, `cy`, `area`.
- **Telemetría:** `/od/video_status_json`, `/od/inference_status_json` y `/od/recording_status_json`.

Se suscribe a `/object_distance` y a `/od/recording_cmd`.

`arm_cam_udp` publica `/alignment_command`, `/arm_cam_bbox`, `/arm_cam_stop`, `/arm_cam_grasp_verify` y `/arm_cam_contact_diagnostic`.

## od45-42. ¿Por qué TensorRT podría dejar de funcionar al cargar un modelo nuevo?
**Archivos:** src/od/src/trt/trt.cpp, src/od/src/trt/trt.hpp, src/od/src/pre/pre.hpp, src/od/src/post/post.cpp, src/od/tools/onnx_to_trt_ui.sh, src/od/src/main.cpp

**Respuesta:**
1. **El engine no es portable.** Debe generarse con `trtexec` en la misma GPU y la misma versión de TensorRT y CUDA (`onnx_to_trt_ui.sh`). Si no, `deserializeCudaEngine` devuelve `nullptr`.
   - Ojo: `trt.cpp:64` llama a `engine_->createExecutionContext()` **antes** de comprobar si es nulo (línea 66), así que el proceso cae con un segfault y no con un error legible.
   - El `Logger` está silenciado (`trt.cpp:15`), así que no se ve la causa.
2. **Los nombres de los tensores están fijos:** `input`, `output_boxes` y `output_scores` (`trt.hpp:16-18`). Un ONNX con otros nombres devuelve formas inválidas.
3. **El tamaño de entrada está fijo en 416x416** en `preprocess` (`pre.hpp:6-7`). Si el modelo nuevo usa 608 o 640, `memcpy` de `input_bytes_` lee fuera del blob (`trt.cpp:122`).
4. **No se soportan formas dinámicas:** nunca se llama a `setInputShape`. Con dimensiones `-1` el volumen sale negativo.
5. **Se asume un formato de salida concreto:** `[N,4]` (x1, y1, x2, y2 normalizados) y `[N,C]`, que es el estilo YOLOv4 → ONNX (`post.cpp:140-165`). YOLOv5/v8 usan otro formato (`[1,84,8400]`, `cxcywh`).
6. **Las clases deben coincidir:** el número de líneas de `obj.names` debe ser igual a C.
7. **Pueden faltar recursos:** memoria de GPU o workspace insuficientes.

## od45-43. ¿Cuál es el archivo de código más importante de OD y por qué?
**Archivos:** src/od/src/main.cpp, src/od/src/post/post.cpp, src/od/src/trt/trt.cpp

**Respuesta:** `src/od/src/main.cpp`. Es el punto de entrada de `trt_pipeline` y orquesta todo:
- configuración, captura GStreamer y TensorRT;
- tracker;
- fusión con la distancia;
- lógica de selección de objetivo (`target_hint`);
- dibujo y publicación de los 5 topics principales;
- grabación y métricas.

Si se mide por la lógica algorítmica, el segundo es `src/od/src/post/post.cpp`: decodificación, NMS y todo el tracker ByteTrack-lite, que determina qué cajas e IDs ve el resto del sistema.

## od45-44. ¿Cómo podemos reducir aún más la latencia de OD?
**Archivos:** scripts/pc.sh, src/od/src/main.cpp, src/od/src/pre/pre.cpp, src/od/src/trt/trt.cpp, src/od/src/post/camera_motion.cpp, src/od/tools/onnx_to_trt_ui.sh

**Respuesta (propuestas, a validar con las métricas existentes):**
- **Transporte:**
  - bajar `rtpjitterbuffer latency=100` a unos 20–50 ms (`pc.sh:30`);
  - evitar el salto por shm: leer UDP directamente con appsink, como hace `arm_cam.cpp:274-285`, o añadir `max-buffers=1` al appsink;
  - usar un decodificador por hardware (`nvh264dec`/`nvv4l2decoder`) en lugar de `avdec_h264`.
- **Inferencia:** engine FP16 o INT8 (`onnx_to_trt_ui.sh`).
- **Preproceso en GPU** (resize, normalización y HWC→CHW con CUDA/NPP o `cv::cuda`), o al menos `cv::dnn::blobFromImage`. Escribir directamente en el buffer pinned para evitar los `memcpy` y `assign` de `trt.cpp:122, 143-144`.
- **Movimiento de cámara:** `goodFeaturesToTrack` con 300 puntos más LK en cada frame es costoso. Se puede reducir la resolución, la frecuencia (cada N frames) o el número de puntos (`camera_motion.cpp:20-26`).
- **Pipeline paralelo:** un hilo de captura, uno de inferencia y otro de dibujo/publicación (hoy todo es secuencial), y un executor de ROS en un hilo aparte.
- **Menos trabajo por frame:** reducir `clone()`, la construcción de JSON con `ostringstream` y el `imshow` cuando no hace falta.
- **Latencia percibida de las cajas:** bajar `box_alpha` (ver od45-26/36).

## od45-45. ¿Cuál es el cálculo de Kalman utilizado en OD y cómo se puede mejorar?
**Archivos:** src/od/src/post/tracker_model.cpp, src/od/src/post/tracker_model.hpp, src/od/src/post/tracker_state.hpp, src/od/src/post/post.cpp, src/od/src/post/track_reviver.cpp

**Respuesta:** No es un filtro de Kalman real, sino un **"Kalman-lite"**: un filtro α-β/EMA con ganancias fijas y sin covarianzas (`tracker_model.hpp:7-17`). El estado es `cx, cy, vx, vy, w, h` más la caja suavizada `bx1..by2` (`tracker_state.hpp`).

- **Predicción** (`tracker_model.cpp:7-24`):
  - `cx += vx; cy += vy`;
  - `v *= vel_damp (0.90)`;
  - la caja se desplaza con `v`.
- **Corrección con detección alta** (`tracker_model.cpp:39-77`):
  - innovación `d = z - x_pred`;
  - `v = (1-g)·v + g·d` con `g = vel_gain = 0.55`;
  - la posición se reemplaza por la medición (`cx = z`);
  - `w, h = 0.7·prev + 0.3·med`;
  - caja = `0.85·prev + 0.15·med`;
  - `score = max(0.9·score, det)`.
- **Corrección con detección baja** (`tracker_model.cpp:79-110`):
  - `g = 0.55·0.35`;
  - `cx += g·d` (corrección parcial);
  - más suavizado del tamaño y de la caja;
  - el score no se infla.

**Cómo mejorarlo:**
- Pasar a un KF de velocidad constante al estilo SORT/ByteTrack: estado `[cx, cy, a, h, vx, vy, va, vh]`, con matrices F, H, Q y R (se puede usar `cv::KalmanFilter`).
- Ganancia de Kalman calculada a partir de P, con R escalado por `1/confianza`, en lugar de usar `g` fijo, `vel_damp` y `box_alpha`.
- Usar la posición filtrada en lugar de la medición cruda.
- Gating por distancia de Mahalanobis en lugar de los umbrales en píxeles.
- Asociación global con el algoritmo húngaro sobre `1-IoU` o Mahalanobis.
- Mantener la predicción sin corrección durante la oclusión, con P creciente, en lugar del `+18 px` por frame perdido.
- Opcional: OC-SORT, para el cambio de dirección, o añadir apariencia (BoT-SORT/DeepSORT) para los giros y cruces.
