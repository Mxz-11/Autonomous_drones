# Documentación del Proyecto

## Arquitectura general

- **Webots** simula el dron (Crazyflie) y expone un servidor TCP en el puerto `9002`.
- **vlm_client.py** conecta a Webots, recibe frames de cámara y GPS, y decide comandos de velocidad (`vx vy vz yaw`).
- **LM Studio** sirve el modelo multimodal localmente mediante una API compatible con OpenAI (`http://localhost:1235/v1`).

---

## Módulos de VLM_Client

| Módulo | Rol |
|---|---|
| `vlm_client.py` | Punto de entrada. Loop principal: recibe frame -> invoca LLM -> envía comando. |
| `mission_config.py` | Todos los parámetros configurables (velocidades, puerto, objetivos GPS...). |
| `llm_config.py` | Configuración de modelos LLM. Permite elegir modelo de decisión y de resumen. |
| `mission_state.py` | Log estructurado de eventos de la misión. |
| `hybrid_memory.py` | Memoria híbrida: resumen estratégico persistente en disco + buffer de eventos recientes. |
| `agent_tools.py` | Herramientas LangChain invocables por el agente (registrar evento, decidir movimiento...). |
| `vision_pipeline.py` | Procesado de imagen: codificación base64 y perfiles de compresión. |
| `image_annotator.py` | Añade HUD sobre los frames (progreso X, deriva Y, score de obstáculos). |
| `navigator.py` | Lógica de navegación: parseo de las decisiones, velocidad, evasión de obstáculos. |
| `webots_comm.py` | Gestión de la conexión TCP con Webots (frames y GPS). |
| `llm_invoker.py` | Invocación al LLM, tanto en modo directo como en modo agente ReAct. |
| `prompt_loader.py` | Carga el prompt de la misión desde el fichero `.md` correspondiente. |

---

## Controlador Webots (`crazyflie.c`)

Controlador en C que corre dentro de Webots. Se encarga de:

- Aceptar conexiones TCP en el puerto `9002`.
- Leer comandos de velocidad (`vx, vy, vz, yaw_rate`) y aplicarlos al dron.
- Estabilizar el vuelo mediante control PID (altitud, cabeceo, guiñada).
- Capturar y enviar frames de cámara al cliente Python cuando recibe la orden `FRAME`.
- Enviar paquete de posición GPS (x, y, z) en cada paso de simulación.

---

## Ejecución desde cero

### 1. Prerequisitos

- Python **3.12** (recomendado; 3.14 puede dar warnings de Pydantic).
- [LM Studio](https://lmstudio.ai) instalado.
- Webots instalado con el mundo del proyecto cargado.

### 2. Entorno virtual e instalación

```bash
# Desde la raíz del proyecto
python3.12 -m venv venv312
source venv312/bin/activate
pip install -r VLM_Client/requirements.txt
```

### 3. Configurar LM Studio

1. Descarga el modelo `Qwen3-VL-8B` desde la pestaña *Search*.
2. Ve a *Local Server*, selecciona el modelo y pulsa *Start Server*.
3. Verifica que el servidor escucha en `http://localhost:1235/v1`.  
   Si el puerto es distinto, edita `LMSTUDIO_OPENAI_BASE` en `VLM_Client/llm_config.py`.

Puedes verificar la conexión con:
```bash
cd VLM_Client && python llm_config.py
```

### 4. Abrir Webots

Carga el mundo `.wbt` correspondiente a la misión. El controlador `crazyflie.c` se lanzará automáticamente y quedará esperando conexión en el puerto `9002`.

### 5. Lanzar el cliente

```bash
cd VLM_Client
source ../venv312/bin/activate

# Misión por defecto (prompt: prompts/mision1.md)
python vlm_client.py

# Misión con objetivo GPS explícito y prompt personalizado
DRONE_TARGET_X=27 DRONE_TARGET_Y=0 python vlm_client.py --prompt prompts/mision1.md

# Misión de búsqueda de objeto (ej: barril rojo en test2.wbt)
DRONE_TARGET_X=4 DRONE_TARGET_Y=2 VLM_ARRIVAL_HOVER=1 python vlm_client.py --prompt prompts/seek_red.md
```

---

## Tests unitarios

La suite cubre 9 módulos (228 tests) y **no requiere Webots ni LM Studio** — todo corre en local sin dependencias externas.

```bash
cd VLM_Client
source ../venv312/bin/activate
python -m pytest tests/ -v
```

Para instalar pytest si no está disponible:

```bash
pip install pytest
```

| Archivo de test | Módulo cubierto | Tests |
|---|---|---|
| `test_mission_config.py` | Variables de entorno y constantes | 17 |
| `test_mission_state.py` | `MissionState` (eventos, posición, thread-safety) | 40 |
| `test_navigator.py` | Parseo LLM, guardrails GPS, obstacle avoidance | 38 |
| `test_vision_pipeline.py` | Encoding de frames, detección de errores de contexto | 30 |
| `test_metricas.py` | Parseo de logs JSONL, métricas de sesión | 44 |
| `test_prompt_loader.py` | Carga de prompts, parseo de args CLI | 12 |
| `test_hybrid_memory.py` | Persistencia en disco, contexto, resumen estratégico | 27 |
| `test_webots_comm.py` | Protocolo TCP socket, decodificación BGRA→RGB | 11 |
| `test_agent_tools.py` | Tools LangChain, estado global del agente | 16 |

---

### Variables de entorno útiles

| Variable | Por defecto | Efecto |
|---|---|---|
| `DRONE_TARGET_X / Y` | `None` | Activa navegación guiada por GPS hacia esa coordenada. |
| `VLM_USE_AGENT` | `0` | `1` activa modo agente ReAct. |
| `VLM_ARRIVAL_HOVER` | `0` | `1` detiene el dron al llegar al objetivo en lugar de aterrizar. |
| `VLM_OBSTACLE_AVOID` | `1` | `0` desactiva la evasión automática de obstáculos. |
| `VLM_CRUISE_Z` | `0.60` | Altitud de crucero en metros. |
| `VLM_SPEED_DIVISOR` | `2.0` | Divide todas las velocidades de salida. |
