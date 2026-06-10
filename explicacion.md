
## 8. Análisis técnico del flujo de datos entre módulos

### 8.1 Protocolo de comunicación TCP: módulo de bajo nivel → módulo intermedio

La comunicación entre el controlador C de Webots y el cliente Python se basa en un protocolo TCP *pull-based* asimétrico implementado sobre un socket bloqueante en el lado emisor y no bloqueante en el lado receptor.

El controlador en C actúa como **servidor TCP** escuchando en el puerto 9002. Para evitar el problema de `TIME_WAIT` en reconexiones rápidas, el socket de escucha se configura con `SO_REUSEADDR`. El socket de cliente no usa `O_NONBLOCK` (decisión de diseño explícita en el código) porque el envío de frames BGRA puede superar los 100 KB y `send()` necesita ser bloqueante para completar la escritura sin devolver `EAGAIN`. La latencia de envío se reduce mediante la desactivación del algoritmo de Nagle (`TCP_NODELAY`).

El protocolo define un ciclo request/response iniciado siempre por el cliente Python: éste envía la cadena de texto `"FRAME\n"` y el controlador C responde con un paquete binario secuencial en formato little-endian con la siguiente estructura:

```
[8 bytes]        cabecera cámara frontal: ancho × alto (int32 × 2)
[W × H × 4 B]   píxeles cámara frontal en formato BGRA (uint8 por canal)
[12 bytes]       posición GPS X, Y, Z (float32 × 3, en metros)
[4 bytes]        distancia sensor IR frontal (float32, metros)
[8 bytes]        cabecera cámara inferior: ancho × alto (int32 × 2)
[dW × dH × 4 B] píxeles cámara inferior en formato BGRA
```

Los comandos de movimiento enviados por el cliente llegan al controlador C como texto plano con el formato `"vx vy vz yaw\n"` y son parseados con `sscanf`. El receptor C ejecuta `recv()` con `MSG_DONTWAIT` en cada iteración del simulador, procesando todas las líneas del buffer con `strtok`. Esta arquitectura garantiza que el controlador nunca bloquea el hilo de simulación esperando datos de red.

En el lado Python, `webots_comm.recv_exact()` implementa una lectura garantizada acumulativa: llama a `socket.recv()` en bucle hasta recibir exactamente `n` bytes, manejando correctamente los casos de fragmentación de paquetes TCP.

### 8.2 Deserialización y transformación de imagen

Una vez recibidos los bytes brutos de la cámara, `webots_comm.decode_frame_rgb()` aplica la conversión de espacio de color. Webots envía los píxeles en orden BGRA (Blue-Green-Red-Alpha), formato nativo de OpenCV. La conversión a RGB se realiza mediante indexación NumPy: `frame[:, :, 2::-1]` invierte los tres primeros canales en una sola operación vectorial sobre el array, descartando implícitamente el canal alpha al no incluirlo en el resultado.

El resultado es un array NumPy de forma `(H, W, 3)` con dtype `uint8` que sirve como representación interna unificada para todos los módulos posteriores.

### 8.3 Anotación del frame (HUD)

Antes de ser enviado al VLM, el frame pasa por `image_annotator.annotate_frame()`, que utiliza OpenCV para superponer información de estado sobre la imagen. Este paso es arquitectónicamente relevante: convierte datos numéricos (posición GPS, score de obstáculos, fase de vuelo) en información visual que el propio modelo puede interpretar sin necesidad de razonar sobre valores crudos en el texto del prompt. Las anotaciones incluyen barra de progreso X, indicador de deriva Y con etiqueta direccional `DRIFT LEFT/RIGHT`, altitud, distancia IR frontal, y un overlay rojo semitransparente cuando el detector CV confirma obstáculo en el tercio central del frame.

### 8.4 Pipeline de codificación para el VLM

`vision_pipeline.encode_both_profiles()` produce dos variantes del mismo frame anotado para gestionar el presupuesto de contexto del modelo:

- **Perfil `default`**: redimensión proporcional a 384 px en el eje mayor, compresión JPEG a calidad 78, submuestreo de crominancia 4:2:0 (`subsampling=2`). El escalado usa `Image.Resampling.LANCZOS` (Pillow), que preserva mejor los bordes frente a algoritmos bilineales.
- **Perfil `tiny`**: 256 px máximo, calidad 65. Se usa como fallback si el modelo devuelve un error de ventana de contexto (`is_context_window_error`).

La compresión JPEG se realiza sobre un buffer en memoria (`io.BytesIO`) sin escritura a disco. El resultado se codifica en Base64 con `base64.b64encode()` y se formatea como data URL: `data:image/jpeg;base64,<b64>`. Este formato es el estándar para incrustar imágenes en la API de OpenAI, y LM Studio lo soporta a través de su compatibilidad con dicha especificación.

### 8.5 Construcción del mensaje multimodal y llamada al VLM

`llm_invoker.invoke_direct_llm()` construye la estructura de mensaje LangChain. Se utilizan dos tipos de objeto de `langchain_core.messages`:

- `SystemMessage`: contiene el prompt de misión completo cargado desde el fichero `.md`. Define el comportamiento del agente, el referencial espacial, las fases de vuelo y el formato de salida obligatorio.
- `HumanMessage`: contiene una lista de bloques de contenido heterogéneos según el protocolo multimodal de OpenAI. Cuando la cámara inferior está activa, la lista incluye cuatro elementos: texto de etiqueta de cámara frontal, bloque `image_url` con la imagen frontal, texto de etiqueta de cámara inferior, bloque `image_url` con la imagen inferior, y finalmente el texto de estado de `build_vlm_user_text()`.

La llamada HTTP se realiza a través de `LMStudioCompatChatOpenAI`, una subclase de `ChatOpenAI` de LangChain que apunta a `http://127.0.0.1:1235/v1`. Los parámetros del modelo son: temperatura 0.3, máximo 512 tokens, timeout 90 s. Se inyecta `extra_body: {"enable_thinking": False}` para deshabilitar el modo de razonamiento extendido de Qwen3, que consumiría todo el presupuesto de tokens antes de producir contenido accionable.

### 8.6 Parseo de la respuesta del VLM

La respuesta textual del modelo se procesa mediante `navigator.parse_movement()` con dos expresiones regulares independientes:

```python
m = re.search(r"movement\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", texto)
r = re.search(r"rotation\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", texto)
```

Los valores extraídos se acotan: `movement ∈ [0, 1]`, `rotation ∈ [-1, 1]`. Este diseño hace que el formato de salida sea tolerante a variaciones de verbalización del modelo siempre que incluya los literales `movement` y `rotation` seguidos de su valor numérico.

### 8.7 Guardarraíles de control y controlador GPS de respaldo

`navigator.apply_control_guardrails()` implementa un blending entre la salida del VLM y un controlador determinista de referencia (`compute_gps_guided_control`). El controlador GPS calcula el heading actual promediando hasta 3 vectores de desplazamiento consecutivos del historial de posición, estima el error angular hacia el objetivo con `atan2`, aplica ganancia proporcional `Kp = 0.70` y suaviza la salida con un filtro exponencial de primer orden `α = 0.40`. El blending aplica las siguientes reglas de precedencia:

- Si la latencia del VLM supera el presupuesto (60 s), se sustituye la decisión completa por la del GPS.
- En fases de aterrizaje o llegada, el GPS toma el mando completo.
- En crucero, si el VLM y el GPS discrepan en signo de rotación, o si el VLM produce una corrección significativamente menor (`< 60%` del valor GPS), prevalece el GPS.
- El movimiento del VLM se acota por el máximo del GPS para evitar avance excesivo con error de rumbo grande.

### 8.8 Detección de obstáculos por visión computacional

`navigator.estimate_obstacle_avoidance()` opera sobre el frame RGB mediante NumPy puro, sin dependencia de OpenCV, para mantener el coste computacional mínimo dentro del ciclo de control. La capa de detección fusiona dos fuentes:

- **Sensor IR** como disparador primario con umbrales duros (0.70 m / 1.40 m).
- **Análisis de imagen** sobre la ROI central del frame (45%–92% vertical, 12%–88% horizontal): se calcula `np.gradient()` para obtener magnitud de borde y un mapa de oscuridad; el score final es `0.65 × edge_map + 0.35 × dark_map`.

`get_committed_obstacle_avoidance()` añade histéresis de dirección: una vez elegida la dirección de esquive, se mantiene durante 6 frames consecutivos para eliminar oscilaciones izquierda/derecha cuando los scores laterales son simétricos.

### 8.9 Worker asíncrono y gestión de concurrencia

En modo asíncrono (`VLM_ASYNC_MODE=1`), `LLMWorker` ejecuta la inferencia en un hilo daemon separado. La cola de comunicación tiene capacidad máxima 1: cuando llega un nuevo frame mientras el modelo sigue procesando el anterior, el frame pendiente se descarta explícitamente con `queue.get_nowait()` antes de encolar el nuevo. Esto garantiza que el modelo siempre procesa el estado más reciente y no acumula latencia por backpressure. El resultado se protege con un `threading.Lock` y se expone como un `dataclass` `LLMResult`. El hilo principal consulta el resultado en cada ciclo y activa el fallback GPS si el resultado supera la edad máxima `DECISION_MAX_LATENCY_S × CONTROL_RATE_HZ + 24` frames.

### 8.10 Escalado final y envío del comando

Los valores normalizados `[movement, rotation, vz]` se convierten a velocidades físicas mediante:

```python
vx  = movement × 0.10 / SPEED_DIVISOR    # [m/s], MAX_FORWARD = 0.10
yaw = rotation × 0.18 / SPEED_DIVISOR    # [rad/s], MAX_YAW = 0.18
vz  = vz        / SPEED_DIVISOR
```

El `SPEED_DIVISOR` (por defecto 2.0) actúa como freno de seguridad global configurable por variable de entorno sin necesidad de recompilar. El comando se serializa como cadena de texto `"vx vy vz yaw\n"` y se envía por el mismo socket TCP, cerrando el bucle de control.

En conjunto, este flujo implementa un ciclo cerrado a 2 Hz en el que cada iteración atraviesa cuatro capas de transformación de datos: deserialización binaria TCP → procesamiento NumPy/OpenCV → codificación JPEG+Base64 → protocolo HTTP multimodal OpenAI. El camino de retorno recorre: texto generado por el VLM → extracción regex → blending determinista → comando TCP en coma flotante.
