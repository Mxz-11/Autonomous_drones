"""
 ██╗   ██╗██╗     ███╗   ███╗     ██████╗██╗     ██╗███████╗███╗   ██╗████████╗
 ██║   ██║██║     ████╗ ████║    ██╔════╝██║     ██║██╔════╝████╗  ██║╚══██╔══╝
 ██║   ██║██║     ██╔████╔██║    ██║     ██║     ██║█████╗  ██╔██╗ ██║   ██║   
 ╚██╗ ██╔╝██║     ██║╚██╔╝██║    ██║     ██║     ██║██╔══╝  ██║╚██╗██║   ██║   
  ╚████╔╝ ███████╗██║ ╚═╝ ██║    ╚██████╗███████╗██║███████╗██║ ╚████║   ██║   
   ╚═══╝  ╚══════╝╚═╝     ╚═╝     ╚═════╝╚══════╝╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝  
  ───────────────────────────────────────────────────────────────────────────────
   vlm_client.py  —  LangChain Agent · Drone VLM Controller
  ───────────────────────────────────────────────────────────────────────────────
   Done by: Mxz-11 (Maximo Valenciano Alvarez)
  ───────────────────────────────────────────────────────────────────────────────

   ┌─────────┐  socket   ┌──────────────┐  LangChain  ┌──────────┐
   │ Webots  │◄──────────│  vlm_client  │────────────►│ LM Studio│
   │  (dron) │ cmd/frames│   (agente)   │  decisions  │  (LLM)   │
   └─────────┘           └──────┬───────┘             └──────────┘
                      MissionState · HybridMem · AgentTools
  ───────────────────────────────────────────────────────────────────────────────
   TCP Socket :9002  │  ReAct Agent  │  Pull-based frames 
"""

import os
import socket
import struct
import time
import re
import json
import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

from llm_config import (
    LMSTUDIO_OPENAI_BASE,
    get_decision_llm,
    get_summary_llm,
    is_probably_network_llm_error,
    probe_openai_compatible_server,
)
from mission_state import MissionState
from hybrid_memory import HybridMemory
from agent_tools import ALL_TOOLS, init_tools
from vision_pipeline import encode_both_profiles, jpeg_b64_to_data_url, is_context_window_error
from telemetry import init_telemetry, get_tracer, get_meter
from advanced_logger import MissionLogger, get_logger, extract_token_usage

# ================= PROMPT DEBUG LOG =================

_PROMPT_LOG_DIR = os.path.join(os.path.dirname(__file__), "logs", "prompts")
os.makedirs(_PROMPT_LOG_DIR, exist_ok=True)


def _log_llm_request(
    frame_id: int,
    mode: str,
    system_prompt: str,
    user_text: str,
    img_b64: str,
    response_text: str,
    latency_s: float,
) -> None:
    """
    Guarda en disco el prompt EXACTO enviado al LLM y la imagen,
    para poder depurar qué está viendo y recibiendo el modelo.
    """
    logger = get_logger("prompt_debug")
    ts = time.strftime("%H%M%S")

    # Guardar imagen JPEG decodificada
    import base64 as _b64
    img_path = os.path.join(_PROMPT_LOG_DIR, f"frame_{frame_id:05d}.jpg")
    try:
        with open(img_path, "wb") as f:
            f.write(_b64.b64decode(img_b64))
    except Exception as e:
        logger.warning("No se pudo guardar imagen frame %d: %s", frame_id, e)
        img_path = "<error>"

    # Guardar prompt completo como JSON
    prompt_path = os.path.join(_PROMPT_LOG_DIR, f"frame_{frame_id:05d}_prompt.json")
    prompt_data = {
        "frame_id": frame_id,
        "timestamp": ts,
        "mode": mode,
        "system_prompt": system_prompt,
        "user_text": user_text,
        "image_file": os.path.basename(img_path),
        "image_b64_length": len(img_b64),
        "response": response_text,
        "latency_s": round(latency_s, 3),
    }
    try:
        with open(prompt_path, "w") as f:
            json.dump(prompt_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning("No se pudo guardar prompt frame %d: %s", frame_id, e)

    # Log conciso en consola
    logger.info(
        "[PROMPT] frame=%d mode=%s | sys=%d chars | user=%s | img=%d bytes b64 | resp=%s | %.1fs",
        frame_id, mode,
        len(system_prompt),
        user_text[:80],
        len(img_b64),
        response_text[:100].replace("\n", " "),
        latency_s,
    )

# ================= CONFIGURACIÓN =================

WEBOTS_IP = "127.0.0.1"
WEBOTS_PORT = 9002
SOCKET_TIMEOUT_S = 10.0

MAX_FORWARD = 0.10
MAX_YAW = 0.18
MAX_DESCEND = -0.18
MAX_ASCEND = 0.22

# Factor divisor de velocidad: ralentiza el dron para compensar latencia LLM.
# 1.0 = velocidad normal, 2.0 = mitad de velocidad, 5.0 = un quinto.
# Configurable con: export VLM_SPEED_DIVISOR=2
try:
    SPEED_DIVISOR = max(1.0, float(os.environ.get("VLM_SPEED_DIVISOR", "2.0")))
except ValueError:
    SPEED_DIVISOR = 2.0
# Rate limiter: en modo síncrono solo limita si el LLM responde muy rápido.
# En modo async mantiene el ritmo de control desacoplado.
CONTROL_RATE_HZ = 2.0
CONTROL_PERIOD_S = 1.0 / CONTROL_RATE_HZ

SUMMARY_UPDATE_INTERVAL = 20
MAX_RECENT_EVENTS = 6
RESET_STRATEGIC_SUMMARY_ON_START = True
ENABLE_SUMMARY_UPDATES = False

TARGET_X = 27.0
TARGET_Y = 0.0
TARGET_X_SLOWDOWN_RADIUS = 3.0
TARGET_X_REACHED_TOL = 0.35
ROT_CORRECTION_THRESHOLD_Y = 0.8
HIGH_Y_ERROR_THRESHOLD = 2.0
DECISION_MAX_LATENCY_S = 60.0
USE_LLM_GUIDANCE = True
HEADING_MIN_STEP = 0.05       # más sensible: detecta movimiento desde 5cm
HEADING_KP = 0.70             # bajado de 1.15 → evita oscilaciones
HEADING_SMOOTH_ALPHA = 0.40   # suavizado exponencial en rotation (0=sin suavizar, 1=congelar)
ARRIVAL_RADIUS = 0.6
OBSTACLE_AVOIDANCE_ENABLED = True
OBSTACLE_CENTER_THRESHOLD = 0.14
OBSTACLE_EDGE_THRESHOLD = 0.14
OBSTACLE_DARK_THRESHOLD = 0.26
OBSTACLE_STRONG_SCORE = 0.28
OBSTACLE_ALT_CRUISE_Z = 0.60
OBSTACLE_ALT_MAX_Z = 1.80

# Activar recepción de posición GPS desde el controlador.
# Requiere que crazyflie.c esté recompilado con el envío de GPS.
# Si usas el controlador antiguo, pon esto a False.
POSITION_ENABLED = True
# Número de floats GPS enviados por el controlador:
# 2 -> (x,y), 3 -> (x,y,z). Si llega 2D, z se asumirá 0.0.
POSITION_PACKET_FLOATS = 3

# ---- Modo de inferencia LLM ----
# SYNC (por defecto): Espera al LLM antes de pedir otro frame.
#   → Elimina requests zombie al 100%. Cada decisión corresponde al frame actual.
#   → El ritmo lo marca el LLM (~2-30s por frame).
# ASYNC (VLM_ASYNC_MODE=1): Comportamiento anterior con LLMWorker en hilo.
#   → Más rápido pero con resultados stale y posibles zombies.
SYNC_LLM_MODE = os.environ.get("VLM_ASYNC_MODE", "0").strip().lower() not in (
    "1",
    "true",
    "yes",
    "on",
)

# Modo agente ReAct: varias llamadas HTTP al LLM por frame (pensamiento + tools + …).
# Modo directo (False): una sola llamada por frame — recomendado para tiempo real / LM Studio.
# Activa con: export VLM_USE_AGENT=1
USE_AGENT_MODE = os.environ.get("VLM_USE_AGENT", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
# Tope de pasos del grafo ReAct (evita bucles largos; no fuerza 1 sola llamada).
def _agent_recursion_limit() -> int:
    try:
        return max(4, int(os.environ.get("VLM_AGENT_RECURSION_LIMIT", "12")))
    except ValueError:
        return 12


AGENT_RECURSION_LIMIT = _agent_recursion_limit()

# ================================================


# ================= SOCKET a Webots =================

def connect_to_webots() -> socket.socket:
    """Conecta al servidor de Webots con reintentos automáticos."""
    logger = get_logger("socket")
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(SOCKET_TIMEOUT_S)
            sock.connect((WEBOTS_IP, WEBOTS_PORT))
            logger.info("Conectado a Webots")
            return sock
        except ConnectionRefusedError:
            logger.warning("Conexión rechazada, reintentando en 2s...")
            time.sleep(2)


def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Recibe exactamente N bytes del socket."""
    data = b""
    while len(data) < n:
        try:
            packet = sock.recv(n - len(data))
        except socket.timeout as e:
            raise TimeoutError(f"Timeout recibiendo {n} bytes (recibidos {len(data)})") from e
        if not packet:
            raise ConnectionError("Socket cerrado")
        data += packet
    return data


# ================= PROMPT DEL SISTEMA =================

VLM_DIRECT_SYSTEM = (
    "You are a drone's visual navigation system. You see the drone's front camera.\n"
    "\n"
    "MISSION: Fly forward to X=27, land on the green H helipad (at Y≈0).\n"
    "\n"
    "THINK STEP BY STEP (briefly):\n"
    "1. What do you see ahead? (obstacles, open space, helipad, buildings)\n"
    "2. Is the path clear or blocked?\n"
    "3. Are we drifting left/right from the goal line (Y≈0)?\n"
    "\n"
    "Then output EXACTLY this as the LAST line of your response:\n"
    "movement=<0.0 to 1.0>, rotation=<-1.0 to 1.0>\n"
    "\n"
    "Where:\n"
    "- movement: forward speed (0=stop, 0.3=slow, 0.6=moderate, 1=full)\n"
    "- rotation: turn direction (negative=left, positive=right, 0=straight)\n"
    "  Use small values like ±0.1–0.3 for gentle corrections, ±0.5–1.0 only for sharp turns."
)

SYSTEM_PROMPT = (
    "Drone visual navigator to X=27, land on green H (Y≈0). Tools: log, status, memory. "
    "Think step by step, then last line MUST be: movement=<0..1>, rotation=<-1..1>"
)

VLM_USER_CTX_CHARS = 160


def build_vlm_user_text(
    frame_id: int,
    pos: dict,
    hybrid_memory: HybridMemory,
) -> str:
    line = f"#{frame_id} X={pos['x']:.2f} Y={pos['y']:.2f} Z={pos['z']:.2f}"
    if VLM_USER_CTX_CHARS <= 0:
        return line
    ctx = hybrid_memory.get_context_text().replace("\n", " ").strip()
    if len(ctx) > VLM_USER_CTX_CHARS:
        ctx = ctx[-VLM_USER_CTX_CHARS :]
    return f"{line} | {ctx}" if ctx else line


# ================= LLM WORKER (hilo desacoplado) =================

@dataclass
class LLMResult:
    """Resultado de una inferencia LLM, compartido entre hilos."""
    movement: float = 0.8
    rotation: float = 0.0
    answer: str = "movement=0.8, rotation=0.0"
    latency_s: float = 0.0
    frame_id: int = 0
    phase: str = "llm_pending"


class LLMWorker:
    """
    Hilo dedicado al LLM desacoplado del bucle de control.

    El bucle principal deposita frames en una cola de tamaño 1
    (el más reciente siempre gana). El hilo LLM procesa a su
    propio ritmo y escribe el resultado en `last_result`.

    Principios de diseño:
      - Queue maxsize=1: si llega un frame nuevo mientras el LLM
        está ocupado, el frame viejo se descarta. Así el modelo
        siempre ve el estado más actual cuando termina.
      - El bucle de control nunca espera: lee `last_result` y sigue.
      - `is_busy` permite al bucle saber si hay inferencia en curso
        (útil para decidir si usar GPS fallback o resultado cacheado).
    """

    def __init__(self):
        import queue as _queue_mod
        self._queue: "_queue_mod.Queue[Optional[dict]]" = _queue_mod.Queue(maxsize=1)
        self.last_result: LLMResult = LLMResult()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.is_busy = False
        self._total_inferences = 0
        self._total_dropped = 0

        # Referencias a los objetos del sistema (se inyectan en start())
        self._agent = None
        self._decision_llm = None
        self._mission_state = None
        self._hybrid_memory = None
        self._use_agent_mode = False

    def start(self, agent, decision_llm, mission_state, hybrid_memory,
              use_agent_mode: bool = False) -> None:
        """
        Arranca el hilo worker. Debe llamarse una vez, después de
        inicializar todos los componentes del sistema.
        """
        self._agent = agent
        self._decision_llm = decision_llm
        self._mission_state = mission_state
        self._hybrid_memory = hybrid_memory
        self._use_agent_mode = use_agent_mode
        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._run,
            name="llm-worker",
            daemon=True,
        )
        self._thread.start()
        get_logger("llm_worker").info("LLM Worker arrancado (hilo desacoplado).")

    def submit(
        self,
        img_b64: str,
        frame_id: int,
        img_b64_fb: Optional[str] = None,
    ) -> bool:
        """
        Envía un frame al worker para su procesamiento.

        Si la cola ya tiene un frame pendiente (LLM ocupado),
        descarta el frame antiguo y encola el nuevo.

        Returns:
            True si se encoló correctamente, False si el worker está parado.
        """
        if self._stop_event.is_set():
            return False

        payload = {
            "img_b64": img_b64,
            "frame_id": frame_id,
            "img_b64_fb": img_b64_fb,
        }

        # Descartar frame viejo si el LLM no da abasto
        if self._queue.full():
            try:
                self._queue.get_nowait()
                self._total_dropped += 1
            except Exception:
                pass

        try:
            self._queue.put_nowait(payload)
            return True
        except Exception:
            return False

    def get_result(self) -> LLMResult:
        """Devuelve el último resultado disponible (thread-safe)."""
        with self._lock:
            return LLMResult(
                movement=self.last_result.movement,
                rotation=self.last_result.rotation,
                answer=self.last_result.answer,
                latency_s=self.last_result.latency_s,
                frame_id=self.last_result.frame_id,
                phase=self.last_result.phase,
            )

    def stop(self, timeout: float = 5.0) -> None:
        """Para el worker de forma limpia."""
        self._stop_event.set()
        try:
            self._queue.put_nowait(None)   # señal de parada
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        get_logger("llm_worker").info(
            f"LLM Worker detenido. "
            f"Inferencias: {self._total_inferences}, "
            f"Frames descartados: {self._total_dropped}"
        )

    def _run(self) -> None:
        """Bucle interno del hilo worker."""
        logger = get_logger("llm_worker")
        while not self._stop_event.is_set():
            try:
                payload = self._queue.get(timeout=1.0)
            except Exception:
                continue   # timeout → comprobar stop_event y volver

            if payload is None:
                break      # señal de parada

            img_b64 = payload["img_b64"]
            frame_id = payload["frame_id"]
            img_b64_fb = payload.get("img_b64_fb")

            self.is_busy = True
            try:
                t0 = time.time()
                if self._use_agent_mode:
                    if self._agent is None:
                        raise RuntimeError("Modo agente sin grafo ReAct inicializado.")
                    movement, rotation, answer = invoke_agent(
                        self._agent, self._mission_state,
                        self._hybrid_memory, img_b64, frame_id,
                        img_b64_fb=img_b64_fb,
                    )
                else:
                    movement, rotation, answer = invoke_direct_llm(
                        self._decision_llm, self._mission_state,
                        self._hybrid_memory, img_b64, frame_id,
                        img_b64_fb=img_b64_fb,
                    )
                latency = time.time() - t0

                with self._lock:
                    self.last_result = LLMResult(
                        movement=movement,
                        rotation=rotation,
                        answer=answer,
                        latency_s=latency,
                        frame_id=frame_id,
                        phase="llm_async",
                    )
                self._total_inferences += 1

            except Exception as e:
                logger.error(f"Error en LLM Worker (frame {frame_id}): {e}")
                # Mantener el último resultado válido, no sobreescribir con zeros
            finally:
                self.is_busy = False


# Instancia global del worker
_llm_worker = LLMWorker()

_LLM_NET_ERR_LOG_INTERVAL_S = 20.0
_llm_network_error_last_full_log_ts: float = 0.0


def _log_throttled_llm_network_error(logger_name: str, exc: BaseException) -> None:
    """Evita inundar logs si el servidor LM Studio no está accesible."""
    global _llm_network_error_last_full_log_ts
    log = get_logger(logger_name)
    now = time.time()
    if now - _llm_network_error_last_full_log_ts >= _LLM_NET_ERR_LOG_INTERVAL_S:
        _llm_network_error_last_full_log_ts = now
        log.error(
            "Sin conexión al LLM (%s). OPENAI base=%s — si LM Studio no muestra peticiones, "
            "revisa que el servidor local esté iniciado y que LMSTUDIO_OPENAI_BASE coincida.",
            exc,
            LMSTUDIO_OPENAI_BASE,
        )
    else:
        log.debug("LLM (red, suprimido): %s", exc)


# ================= INICIALIZACIÓN DEL AGENTE =================

def create_agent_system():
    """
    Crea y configura todo el sistema del agente:
      1. LLMs locales (decisión + resumen).
      2. MissionState (log de eventos).
      3. HybridMemory (resumen persistente + buffer).
      4. Agent Tools (herramientas invocables).
      5. Agente ReAct de LangGraph.

    Returns:
        Tupla (agent, mission_state, hybrid_memory, decision_llm)
    """
    ml = MissionLogger()
    logger = get_logger("init")
    logger.info("=" * 60)
    logger.info("  INICIALIZANDO SISTEMA DE AGENTE LANGCHAIN")
    logger.info("=" * 60)

    logger.info("[1/5] Configurando modelos LLM...")
    decision_llm = get_decision_llm()
    summary_llm = get_summary_llm()
    _dm = getattr(decision_llm, "model_name", None) or getattr(decision_llm, "model", "?")
    _sm = getattr(summary_llm, "model_name", None) or getattr(summary_llm, "model", "?")
    logger.info(f"      Modelo decisión: {_dm}")
    logger.info(f"      Modelo resumen:  {_sm}")
    logger.info(f"      API OpenAI-compatible: {LMSTUDIO_OPENAI_BASE}")
    probe_ok, probe_detail = probe_openai_compatible_server()
    if probe_ok:
        logger.info(f"      Probe GET /models OK → {probe_detail}")
    else:
        logger.error(f"      Probe fallido: {probe_detail}")

    logger.info("[2/5] Creando estado de misión...")
    mission_state = MissionState("drone_landing_mission_x27")
    mission_state.set_metadata("mission_type", "x27_landing")
    mission_state.set_metadata("webots_port", WEBOTS_PORT)
    mission_state.set_metadata("position_packet_floats", POSITION_PACKET_FLOATS)
    logger.info(f"      Misión: {mission_state.mission_name}")

    logger.info("[3/5] Configurando memoria híbrida...")
    hybrid_memory = HybridMemory(
        mission_state=mission_state,
        recent_events_count=MAX_RECENT_EVENTS,
    )
    # Evita contaminar la misión actual con contexto persistido de ejecuciones previas.
    if RESET_STRATEGIC_SUMMARY_ON_START:
        hybrid_memory.update_summary_manual("")
        logger.info("      Resumen estratégico previo reiniciado al arrancar.")

    logger.info(f"      Buffer eventos: {MAX_RECENT_EVENTS}")
    logger.info(f"      Resumen previo: {'Sí' if hybrid_memory.strategic_summary else 'No'}")

    logger.info("[4/5] Registrando herramientas del agente...")
    init_tools(mission_state, hybrid_memory, decision_llm, summary_llm)
    for t in ALL_TOOLS:
        logger.info(f"      → {t.name}")

    logger.info("[5/5] Creando agente LangGraph (ReAct)...")

    if USE_AGENT_MODE:
        agent = create_react_agent(
            model=decision_llm,
            tools=ALL_TOOLS,
            prompt=SYSTEM_PROMPT,
        )
        # Verificar si bind_tools realmente funcionó
        if not getattr(decision_llm, "_bind_tools_supported", True):
            logger.warning(
                "      ⚠ bind_tools NO soportado por el modelo. "
                "ReAct se creó pero las herramientas NO están registradas. "
                "El agente funcionará como LLM directo con overhead extra."
            )
        logger.info(
            "      ReAct activo: hasta varias llamadas LLM por frame (recursion_limit=%s).",
            AGENT_RECURSION_LIMIT,
        )
    else:
        agent = None
        logger.info(
            "      ReAct omitido (VLM_USE_AGENT=0): se usará LLM directo, 1 petición por frame."
        )

    logger.info("Sistema de agente inicializado correctamente.")
    logger.info("=" * 60)

    return agent, mission_state, hybrid_memory, decision_llm


# ================= PROCESAMIENTO DE FRAME =================

def decode_frame_rgb(frame_bytes: bytes, width: int, height: int) -> np.ndarray:
    """
    Decodifica bytes RGBA a array RGB (H, W, 3).
    """
    frame = np.frombuffer(frame_bytes, np.uint8).reshape((height, width, 4))
    return frame[:, :, :3]


def estimate_obstacle_avoidance(frame_rgb: np.ndarray) -> tuple[bool, float, float, float]:
    """
    Detecta bloqueo frontal en el frame y propone corrección:
      - blocked: si hay obstáculo frontal probable.
      - avoid_rotation: giro recomendado (-1 izquierda, +1 derecha).
      - speed_scale: factor multiplicador para movement.
      - center_score: severidad del bloqueo frontal.
    """
    h, w, _ = frame_rgb.shape
    gray = frame_rgb.astype(np.float32).mean(axis=2) / 255.0

    # ROI frontal (zona media-baja de la imagen).
    y0, y1 = int(h * 0.45), int(h * 0.92)
    x0, x1 = int(w * 0.12), int(w * 0.88)
    roi = gray[y0:y1, x0:x1]

    if roi.size == 0:
        return False, 0.0, 1.0, 0.0

    gy, gx = np.gradient(roi)
    edge_mag = np.hypot(gx, gy)
    edge_map = (edge_mag > OBSTACLE_EDGE_THRESHOLD).astype(np.float32)
    dark_map = (roi < OBSTACLE_DARK_THRESHOLD).astype(np.float32)
    obs_map = 0.65 * edge_map + 0.35 * dark_map

    rw = roi.shape[1]
    c0, c1 = int(rw * 0.35), int(rw * 0.65)
    l0, l1 = 0, int(rw * 0.35)
    r0, r1 = int(rw * 0.65), rw

    center_score = float(obs_map[:, c0:c1].mean()) if c1 > c0 else 0.0
    left_score = float(obs_map[:, l0:l1].mean()) if l1 > l0 else center_score
    right_score = float(obs_map[:, r0:r1].mean()) if r1 > r0 else center_score

    if center_score < OBSTACLE_CENTER_THRESHOLD:
        return False, 0.0, 1.0, center_score

    severity = min(1.0, max(0.0, (center_score - OBSTACLE_CENTER_THRESHOLD) / 0.35))
    # Girar hacia el lado con menor ocupación visual.
    turn_sign = -1.0 if left_score < right_score else 1.0
    avoid_rotation = turn_sign * min(0.95, 0.30 + 0.55 * severity)
    speed_scale = max(0.20, 1.0 - 0.70 * severity)

    return True, avoid_rotation, speed_scale, center_score


def parse_movement(answer: str) -> tuple[float, float]:
    """
    Extrae movement y rotation de la respuesta del agente.

    Args:
        answer: Texto de respuesta del agente.

    Returns:
        Tupla (movement, rotation) con valores numéricos.
    """
    answer_lower = str(answer).lower().strip()

    movement = 0.0
    rotation = 0.0

    float_pattern = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    m = re.search(rf"movement\s*[:=]\s*{float_pattern}", answer_lower)
    if m:
        movement = float(m.group(1))

    r = re.search(rf"rotation\s*[:=]\s*{float_pattern}", answer_lower)
    if r:
        rotation = float(r.group(1))

    movement = max(0.0, min(1.0, movement))
    rotation = max(-1.0, min(1.0, rotation))

    return movement, rotation


def compute_rule_based_control(pos_x: float, pos_y: float) -> tuple[float, float, float, str]:
    """
    Controlador determinista para garantizar avance de misión incluso si el LLM
    falla, responde lento o no respeta formato.
    """
    dx = TARGET_X - pos_x

    if dx <= TARGET_X_REACHED_TOL:
        # En zona objetivo: detener avance horizontal y descender.
        landing_rotation = max(-0.3, min(0.3, -0.15 * pos_y))
        return 0.0, landing_rotation, MAX_DESCEND, "landing"

    if dx <= TARGET_X_SLOWDOWN_RADIUS:
        ratio = dx / TARGET_X_SLOWDOWN_RADIUS
        movement = max(0.2, min(0.55, 0.55 * ratio))
    else:
        movement = 0.8

    y_error = pos_y - TARGET_Y
    if abs(y_error) <= 0.35:
        rotation = 0.0
    else:
        rotation = max(-0.65, min(0.65, -0.12 * y_error))

    return movement, rotation, 0.0, "cruise"


def apply_control_guardrails(
    llm_movement: float,
    llm_rotation: float,
    pos_x: float,
    pos_y: float,
    llm_latency: float,
) -> tuple[float, float, float, str]:
    """
    Aplica reglas de seguridad para que el dron no se quede en avance ciego.
    """
    rb_movement, rb_rotation, rb_vz, phase = compute_rule_based_control(pos_x, pos_y)

    # Si la inferencia tarda demasiado, priorizar totalmente control determinista.
    if llm_latency > DECISION_MAX_LATENCY_S:
        return rb_movement, rb_rotation, rb_vz, f"{phase}_latency_fallback"

    # Siempre asegurar consistencia de misión.
    if phase == "landing":
        return rb_movement, rb_rotation, rb_vz, "landing_guardrail"

    movement = llm_movement
    rotation = llm_rotation

    # Guardrail 1: lejos del objetivo, no aceptar velocidades demasiado bajas.
    if TARGET_X - pos_x > TARGET_X_REACHED_TOL:
        movement = max(movement, rb_movement)

    # Guardrail 2: desacelerar cuando está cerca del objetivo.
    if TARGET_X - pos_x <= TARGET_X_SLOWDOWN_RADIUS:
        movement = min(movement, rb_movement)

    # Guardrail 3: asegurar dirección y magnitud de corrección lateral.
    y_error = pos_y - TARGET_Y
    if abs(y_error) > ROT_CORRECTION_THRESHOLD_Y:
        wrong_direction = (rotation * rb_rotation) < 0
        insufficient = abs(rotation) < abs(rb_rotation) * 0.75
        if wrong_direction or insufficient:
            rotation = rb_rotation

    # Guardrail 4: si el error lateral es muy alto, reducir avance para priorizar centrado.
    if abs(y_error) > HIGH_Y_ERROR_THRESHOLD:
        movement = min(movement, 0.45)

    movement = max(0.0, min(1.0, movement))
    rotation = max(-1.0, min(1.0, rotation))
    return movement, rotation, 0.0, "llm_guarded"


def _normalize_angle(rad: float) -> float:
    """Normaliza ángulos al rango [-pi, pi]."""
    return math.atan2(math.sin(rad), math.cos(rad))


_last_rotation: float = 0.0   # estado interno para suavizado exponencial


def compute_gps_guided_control(mission_state: MissionState) -> tuple[float, float, float, str]:
    """
    Controlador GPS mejorado:
    - Promedia los últimos 3 vectores de desplazamiento para estimar heading estable.
    - Bootstrap proporcional al error angular directo (no a Y directamente).
    - Suavizado exponencial en rotation para evitar oscilaciones.
    - Mantiene altitud mínima de crucero si llega coordenada Z.
    """
    global _last_rotation
    pos = mission_state.position
    x, y, z = pos["x"], pos["y"], pos.get("z", 0.0)
    dx = TARGET_X - x
    dy = TARGET_Y - y
    dist = math.hypot(dx, dy)

    if dist <= ARRIVAL_RADIUS:
        _last_rotation = 0.0
        return 0.0, 0.0, MAX_DESCEND, "gps_landing"

    # ---- Estimar heading actual promediando últimos 3 desplazamientos ----
    history = mission_state.get_recent_positions(4)
    heading_known = False
    heading_now = 0.0
    if len(history) >= 2:
        vectors = []
        for i in range(len(history) - 1, 0, -1):
            hx = history[i]["x"] - history[i - 1]["x"]
            hy = history[i]["y"] - history[i - 1]["y"]
            step = math.hypot(hx, hy)
            if step >= HEADING_MIN_STEP:
                vectors.append((hx / step, hy / step))
            if len(vectors) == 3:
                break
        if vectors:
            avg_x = sum(v[0] for v in vectors) / len(vectors)
            avg_y = sum(v[1] for v in vectors) / len(vectors)
            heading_now = math.atan2(avg_y, avg_x)
            heading_known = True

    desired_heading = math.atan2(dy, dx)

    if heading_known:
        heading_error = _normalize_angle(desired_heading - heading_now)
        raw_rotation = max(-0.9, min(0.9, HEADING_KP * heading_error))
        phase = "gps_track_heading"
    else:
        # Bootstrap: corrección proporcional al error lateral, con saturación suave.
        # Evita el -0.22*y que se saturaba a ±0.8 cuando |Y| era grande.
        y_sign = math.copysign(1.0, y) if abs(y) > 0.1 else 0.0
        lateral_factor = min(1.0, abs(y) / 4.0)  # escala lineal hasta Y=4m, luego cte
        raw_rotation = -y_sign * 0.55 * lateral_factor
        phase = "gps_bootstrap_heading"

    # Suavizado exponencial: evita cambios bruscos de yaw frame a frame
    rotation = HEADING_SMOOTH_ALPHA * _last_rotation + (1.0 - HEADING_SMOOTH_ALPHA) * raw_rotation
    _last_rotation = rotation

    # ---- Velocidad de avance según distancia y error angular ----
    if dist > 12.0:
        movement = 0.90
    elif dist > 6.0:
        movement = 0.70
    elif dist > 2.5:
        movement = 0.50
    else:
        movement = 0.28

    if heading_known:
        abs_err = abs(_normalize_angle(desired_heading - heading_now))
        if abs_err > 1.2:
            movement = min(movement, 0.18)   # error grande → casi parado, gira primero
        elif abs_err > 0.6:
            movement = min(movement, 0.32)

    # ---- Altitud de crucero (solo si viene Z real) ----
    vz = 0.0
    if POSITION_PACKET_FLOATS == 3:
        cruise_z = getattr(compute_gps_guided_control, "_cruise_z", 0.55)
        if z < cruise_z - 0.15 and phase != "gps_landing":
            vz = min(MAX_ASCEND, 0.08 + (cruise_z - z) * 0.20)
        elif z > cruise_z + 0.25 and phase != "gps_landing":
            vz = max(-0.06, -(z - cruise_z) * 0.10)

    return movement, rotation, vz, phase


# ================= INVOCACIÓN DEL AGENTE =================

def invoke_agent(
    agent,
    mission_state: MissionState,
    hybrid_memory: HybridMemory,
    img_b64: str,
    frame_id: int,
    img_b64_fb: Optional[str] = None,
) -> tuple[float, float, str]:
    """
    Invoca al agente LangGraph con el frame actual y el contexto de la misión.

    Args:
        agent: Agente ReAct de LangGraph.
        mission_state: Estado global de la misión.
        hybrid_memory: Memoria híbrida del agente.
        img_b64: JPEG base64 (sin prefijo data:).
        frame_id: ID secuencial del frame.
        img_b64_fb: JPEG base64 más pequeño para reintento si el contexto excede.

    Returns:
        Tupla (movement, rotation) decidida por el agente.
    """
    mission_state.log_event("vlm", "frame_received", {
        "frame_id": frame_id,
        "timestamp": time.time(),
    })
    pos = mission_state.position
    user_text = build_vlm_user_text(frame_id, pos, hybrid_memory)

    def _run_agent(b64: str):
        user_message = [
            {
                "type": "image_url",
                "image_url": {"url": jpeg_b64_to_data_url(b64)},
            },
            {"type": "text", "text": user_text},
        ]
        return agent.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config={"recursion_limit": AGENT_RECURSION_LIMIT},
        )

    try:
        t_start = time.time()
        try:
            result = _run_agent(img_b64)
        except Exception as e1:
            if img_b64_fb and is_context_window_error(e1):
                get_logger("invoke_agent").warning(
                    "Context overflow; reintentando frame reducido."
                )
                result = _run_agent(img_b64_fb)
            else:
                raise e1
        latency = time.time() - t_start

        messages = result.get("messages", [])
        ai_msg = messages[-1] if messages else None
        answer = (
            ai_msg.content
            if (ai_msg and hasattr(ai_msg, "content"))
            else "movement=0.5, rotation=0.0"
        )
        movement, rotation = parse_movement(answer)

        ml = MissionLogger()
        if ai_msg:
            tokens = extract_token_usage(ai_msg)
            model_name = getattr(ai_msg, "response_metadata", {}).get("model_name", "agent_llm")
            ml.log_llm_call(
                model=model_name,
                prompt_tokens=tokens.get("prompt_tokens", 0),
                completion_tokens=tokens.get("completion_tokens", 0),
                total_tokens=tokens.get("total_tokens", 0),
                latency_s=latency,
                response_preview=answer,
            )

        # Guardar prompt exacto + imagen para depuración
        _log_llm_request(
            frame_id=frame_id,
            mode="agent",
            system_prompt=SYSTEM_PROMPT,
            user_text=user_text,
            img_b64=img_b64,
            response_text=answer,
            latency_s=latency,
        )

        mission_state.log_event("agent", "decision_output", {
            "frame_id": frame_id,
            "movement": movement,
            "rotation": rotation,
            "raw_answer": answer[:200],
        })

        return movement, rotation, answer

    except Exception as e:
        if is_probably_network_llm_error(e):
            _log_throttled_llm_network_error("invoke_agent", e)
        else:
            get_logger("invoke_agent").error("Error del agente: %s", e, exc_info=True)
        mission_state.log_event("agent", "error", {
            "frame_id": frame_id,
            "error": str(e)[:200],
        })
        return 0.0, 0.0, ""


# ================= INVOCACIÓN DIRECTA (sin agente) =================

def invoke_direct_llm(
    decision_llm,
    mission_state: MissionState,
    hybrid_memory: HybridMemory,
    img_b64: str,
    frame_id: int,
    img_b64_fb: Optional[str] = None,
) -> tuple[float, float, str]:
    """
    Modo alternativo: invoca al LLM directamente SIN pasar por el agente.
    Útil como fallback o para comparar rendimiento.

    Envía la imagen + contexto directamente al modelo multimodal.
    """
    pos = mission_state.position
    user_text = build_vlm_user_text(frame_id, pos, hybrid_memory)

    def _messages(b64: str):
        return [
            SystemMessage(content=VLM_DIRECT_SYSTEM),
            HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": jpeg_b64_to_data_url(b64)},
                    },
                    {"type": "text", "text": user_text},
                ]
            ),
        ]

    try:
        t_start = time.time()
        try:
            response = decision_llm.invoke(_messages(img_b64))
        except Exception as e1:
            if img_b64_fb and is_context_window_error(e1):
                get_logger("invoke_direct").warning(
                    "Context overflow; reintentando frame reducido."
                )
                response = decision_llm.invoke(_messages(img_b64_fb))
            else:
                raise e1
        latency = time.time() - t_start
        answer = (response.content or "").strip()
        movement, rotation = parse_movement(answer)

        tokens = extract_token_usage(response)
        ml = MissionLogger()
        model_id = getattr(
            decision_llm, "model_name", None
        ) or getattr(decision_llm, "model", "direct_llm")
        ml.log_llm_call(
            model=str(model_id),
            prompt_tokens=tokens.get("prompt_tokens", 0),
            completion_tokens=tokens.get("completion_tokens", 0),
            total_tokens=tokens.get("total_tokens", 0),
            latency_s=latency,
            response_preview=answer,
        )

        # Guardar prompt exacto + imagen para depuración
        _log_llm_request(
            frame_id=frame_id,
            mode="direct",
            system_prompt=VLM_DIRECT_SYSTEM,
            user_text=user_text,
            img_b64=img_b64,
            response_text=answer,
            latency_s=latency,
        )

        mission_state.log_event("vlm_direct", "decision_made", {
            "frame_id": frame_id,
            "movement": movement,
            "rotation": rotation,
        })

        return movement, rotation, answer

    except Exception as e:
        if is_probably_network_llm_error(e):
            _log_throttled_llm_network_error("invoke_direct", e)
        else:
            get_logger("invoke_direct").error("LLM directo falló: %s", e, exc_info=True)
        return 0.0, 0.0, ""


# ================= LOOP PRINCIPAL =================

def main():
    """
    Loop principal del cliente VLM.

    1. Inicializa telemetría OpenTelemetry.
    2. Inicializa el sistema de agente.
    3. Conecta a Webots por socket.
    4. Para cada frame recibido:
       a. Convierte a base64.
       b. Recibe posición X/Y/Z del dron.
       c. Invoca al agente (o LLM directo como fallback).
       d. Parsea movement/rotation.
       e. Envía comando al dron.
       f. Actualiza la memoria periódicamente.
    """
    # ---- Telemetría ----
    init_telemetry("vlm_client")
    tracer = get_tracer("main_loop")
    meter = get_meter("drone_metrics")

    frame_counter = meter.create_counter(
        name="drone.frames_processed",
        description="Total frames processed",
        unit="1",
    )
    decision_histogram = meter.create_histogram(
        name="drone.decision_latency",
        description="Agent decision latency",
        unit="s",
    )
    pos_x_gauge = meter.create_gauge(
        name="drone.position_x",
        description="Drone GPS X position",
        unit="m",
    )
    pos_y_gauge = meter.create_gauge(
        name="drone.position_y",
        description="Drone GPS Y position",
        unit="m",
    )
    pos_z_gauge = meter.create_gauge(
        name="drone.position_z",
        description="Drone GPS Z position",
        unit="m",
    )

    agent, mission_state, hybrid_memory, decision_llm = create_agent_system()

    # Determinar modo de operación
    if USE_AGENT_MODE:
        mission_mode = "agent"
    elif USE_LLM_GUIDANCE and SYNC_LLM_MODE:
        mission_mode = "direct_llm_sync"
    elif USE_LLM_GUIDANCE:
        mission_mode = "direct_llm_async"
    else:
        mission_mode = "gps_autopilot"

    mission_state.log_event("system", "mission_started", {
        "mode": mission_mode,
        "sync_llm": SYNC_LLM_MODE,
        "webots_ip": WEBOTS_IP,
        "webots_port": WEBOTS_PORT,
    })

    # ---- Arrancar el worker LLM solo en modo ASYNC ----
    use_async_worker = (USE_LLM_GUIDANCE or USE_AGENT_MODE) and not SYNC_LLM_MODE
    if use_async_worker:
        _llm_worker.start(
            agent=agent,
            decision_llm=decision_llm,
            mission_state=mission_state,
            hybrid_memory=hybrid_memory,
            use_agent_mode=USE_AGENT_MODE,
        )

    sock = connect_to_webots()
    mission_state.log_event("system", "webots_connected", {
        "ip": WEBOTS_IP, "port": WEBOTS_PORT
    })

    frame_id = 0
    next_control_tick = time.perf_counter()
    _consecutive_llm_failures = 0  # track para GPS fallback automático

    print("\n[LOOP] Iniciando bucle principal...")
    if USE_AGENT_MODE:
        mode_label = "AGENTE LangChain (async worker)"
    elif USE_LLM_GUIDANCE and SYNC_LLM_MODE:
        mode_label = "LLM Directo SÍNCRONO (espera por inferencia, sin zombies)"
    elif USE_LLM_GUIDANCE:
        mode_label = "LLM Directo + Guardrails (async worker)"
    else:
        mode_label = "GPS Autopilot (determinista)"
    print(f"[MODE] {mode_label}")
    if SPEED_DIVISOR > 1.0:
        print(f"[SLOW-MO] Velocidad dividida x{SPEED_DIVISOR:.1f} (VLM_SPEED_DIVISOR)")
    print(f"[PROMPT-LOG] Prompts e imágenes guardados en: {_PROMPT_LOG_DIR}")
    print()

    ml = MissionLogger()
    ml.log_system("Bucle principal iniciado", data={"mode": mission_mode, "sync": SYNC_LLM_MODE})

    while True:
        try:
            # Limita la frecuencia de decisiones/comandos para evitar control a trompicones.
            now = time.perf_counter()
            if now < next_control_tick:
                time.sleep(next_control_tick - now)
            next_control_tick = max(next_control_tick + CONTROL_PERIOD_S, time.perf_counter())

            # ---- Solicitar frame al controlador (pull-based) ----
            try:
                sock.send(b"FRAME\n")
            except BrokenPipeError:
                ml.log_connection_event("connection_lost", {"reason": "BrokenPipe on FRAME request", "action": "reconnecting"})
                mission_state.log_event("system", "connection_lost", {
                    "reason": "BrokenPipe on FRAME request"
                })
                sock.close()
                sock = connect_to_webots()
                continue

            # ---- Leer header + frame + posición (con tracing) ----
            with tracer.start_as_current_span("frame_receive") as span:
                header = recv_exact(sock, 8)
                w, h = struct.unpack("ii", header)

                if w <= 0 or h <= 0 or w > 2000 or h > 2000:
                    ml.log_connection_event("invalid_header", {"width": w, "height": h, "action": "reconnecting"})
                    mission_state.log_event("system", "invalid_header", {
                        "width": w, "height": h
                    })
                    sock.close()
                    sock = connect_to_webots()
                    continue

                img_bytes = recv_exact(sock, w * h * 4)
                frame_rgb = decode_frame_rgb(img_bytes, w, h)

                # ---- Recibir posición GPS (2 o 3 floats) ----
                pos_x, pos_y, pos_z = 0.0, 0.0, 0.0
                if POSITION_ENABLED:
                    if POSITION_PACKET_FLOATS not in (2, 3):
                        raise ValueError("POSITION_PACKET_FLOATS debe ser 2 o 3")
                    pos_bytes = recv_exact(sock, 4 * POSITION_PACKET_FLOATS)
                    if POSITION_PACKET_FLOATS == 2:
                        pos_x, pos_y = struct.unpack("ff", pos_bytes)
                    else:
                        pos_x, pos_y, pos_z = struct.unpack("fff", pos_bytes)

                    # Validar GPS: NaN/Inf rompen el control del dron
                    if math.isfinite(pos_x) and math.isfinite(pos_y) and math.isfinite(pos_z):
                        mission_state.update_position(pos_x, pos_y, pos_z)
                    else:
                        get_logger("gps").warning(
                            "GPS inválido (NaN/Inf): x=%s y=%s z=%s — usando última posición válida",
                            pos_x, pos_y, pos_z,
                        )
                        prev = mission_state.position
                        pos_x, pos_y, pos_z = prev["x"], prev["y"], prev["z"]

                frame_id += 1
                frame_counter.add(1)
                pos_x_gauge.set(pos_x)
                pos_y_gauge.set(pos_y)
                pos_z_gauge.set(pos_z)

                span.set_attribute("frame.id", frame_id)
                span.set_attribute("frame.width", w)
                span.set_attribute("frame.height", h)
                span.set_attribute("drone.pos_x", pos_x)
                span.set_attribute("drone.pos_y", pos_y)
                span.set_attribute("drone.pos_z", pos_z)
                
                ml.log_frame_received(frame_id, w, h, pos_x, pos_y, pos_z)

            # ---- Decidir movimiento (con tracing) ----
            with tracer.start_as_current_span("agent_invoke") as span:
                t0 = time.time()

                if (USE_LLM_GUIDANCE or USE_AGENT_MODE) and SYNC_LLM_MODE:
                    # ============================================================
                    # MODO SÍNCRONO: Llama al LLM y ESPERA el resultado.
                    # Garantiza 0 requests zombie — cada decisión corresponde
                    # exactamente al frame que acaba de recibir.
                    # ============================================================
                    img_b64, img_b64_fb = encode_both_profiles(frame_rgb)

                    try:
                        if USE_AGENT_MODE and agent is not None:
                            movement, rotation, answer = invoke_agent(
                                agent, mission_state, hybrid_memory,
                                img_b64, frame_id, img_b64_fb=img_b64_fb,
                            )
                        else:
                            movement, rotation, answer = invoke_direct_llm(
                                decision_llm, mission_state, hybrid_memory,
                                img_b64, frame_id, img_b64_fb=img_b64_fb,
                            )
                        latency = time.time() - t0

                        # Verificar si la respuesta fue vacía (error interno)
                        if not answer:
                            raise RuntimeError("LLM returned empty response")

                        _consecutive_llm_failures = 0
                        guarded_movement, guarded_rotation, guarded_vz, guard_reason = apply_control_guardrails(
                            llm_movement=movement,
                            llm_rotation=rotation,
                            pos_x=pos_x,
                            pos_y=pos_y,
                            llm_latency=latency,
                        )
                        phase = "llm_sync" if not USE_AGENT_MODE else "agent_sync"

                    except Exception as sync_exc:
                        _consecutive_llm_failures += 1
                        latency = time.time() - t0
                        get_logger("sync_llm").warning(
                            "LLM sync falló (intento %d, %.1fs): %s",
                            _consecutive_llm_failures, latency, sync_exc,
                        )
                        # GPS fallback para este frame — el siguiente se reintenta con LLM
                        movement, rotation, guarded_vz, phase = compute_gps_guided_control(mission_state)
                        answer = f"sync_gps_fallback (fail #{_consecutive_llm_failures})"
                        guarded_movement, guarded_rotation, guard_reason = movement, rotation, phase

                elif (USE_LLM_GUIDANCE or USE_AGENT_MODE) and not SYNC_LLM_MODE:
                    # ============================================================
                    # MODO ASÍNCRONO (legacy): Worker en hilo separado.
                    # ============================================================
                    img_b64, img_b64_fb = encode_both_profiles(frame_rgb)
                    _llm_worker.submit(img_b64, frame_id, img_b64_fb)

                    llm_result = _llm_worker.get_result()

                    result_age_frames = frame_id - llm_result.frame_id
                    llm_has_result = llm_result.frame_id > 0

                    max_stale_frames = int(DECISION_MAX_LATENCY_S * CONTROL_RATE_HZ) + 24
                    llm_stale = result_age_frames > max_stale_frames

                    if not llm_has_result or llm_stale:
                        movement, rotation, guarded_vz, phase = compute_gps_guided_control(mission_state)
                        reason = "cold_start" if not llm_has_result else f"stale_{result_age_frames}f"
                        answer = f"gps_fallback ({reason})"
                        latency = time.time() - t0
                        guarded_movement, guarded_rotation, guard_reason = movement, rotation, phase
                    else:
                        movement = llm_result.movement
                        rotation = llm_result.rotation
                        answer = llm_result.answer
                        latency = llm_result.latency_s
                        guarded_movement, guarded_rotation, guarded_vz, guard_reason = apply_control_guardrails(
                            llm_movement=movement,
                            llm_rotation=rotation,
                            pos_x=pos_x,
                            pos_y=pos_y,
                            llm_latency=latency,
                        )
                        phase = "llm_async" if not USE_AGENT_MODE else "agent_async"

                else:
                    # Modo GPS puro (USE_LLM_GUIDANCE=False)
                    movement, rotation, guarded_vz, phase = compute_gps_guided_control(mission_state)
                    answer = f"gps_controller phase={phase}"
                    latency = time.time() - t0
                    guarded_movement, guarded_rotation, guard_reason = movement, rotation, phase

                decision_histogram.record(latency, attributes={"decision.phase": phase})
                span.set_attribute("decision.phase", phase)
                span.set_attribute("decision.movement_raw", movement)
                span.set_attribute("decision.rotation_raw", rotation)
                span.set_attribute("decision.movement_guarded", guarded_movement)
                span.set_attribute("decision.rotation_guarded", guarded_rotation)
                span.set_attribute("decision.guard_reason", guard_reason)
                span.set_attribute("decision.latency_s", round(latency, 3))

                mission_state.log_event("controller", "guardrail_applied", {
                    "frame_id": frame_id,
                    "raw_movement": round(movement, 4),
                    "raw_rotation": round(rotation, 4),
                    "guarded_movement": round(guarded_movement, 4),
                    "guarded_rotation": round(guarded_rotation, 4),
                    "guarded_vz": round(guarded_vz, 4),
                    "reason": guard_reason,
                    "latency_s": round(latency, 4),
                })

                movement = guarded_movement
                rotation = guarded_rotation
                vz = guarded_vz

                if OBSTACLE_AVOIDANCE_ENABLED:
                    blocked, avoid_rotation, speed_scale, obstacle_score = estimate_obstacle_avoidance(frame_rgb)
                    if blocked and phase != "gps_landing":
                        original_rotation = rotation
                        original_movement = movement
                        original_vz = vz
                        rotation = avoid_rotation
                        movement = max(0.0, min(1.0, movement * speed_scale))
                        # Evasión vertical: ante bloqueo frontal, subir para sortear.
                        # Si no tenemos Z real (controlador 2D), igualmente ordenamos ascenso.
                        if POSITION_PACKET_FLOATS == 2:
                            vz = max(vz, MAX_ASCEND * 0.7)
                        else:
                            if pos_z < OBSTACLE_ALT_MAX_Z:
                                climb_strength = min(
                                    1.0,
                                    max(
                                        0.0,
                                        (obstacle_score - OBSTACLE_CENTER_THRESHOLD)
                                        / max(0.05, (OBSTACLE_STRONG_SCORE - OBSTACLE_CENTER_THRESHOLD)),
                                    ),
                                )
                                target_vz = 0.08 + climb_strength * (MAX_ASCEND - 0.08)
                                vz = max(vz, target_vz)
                        guard_reason = f"{guard_reason}+obstacle"
                        if frame_id % 10 == 0:
                            mission_state.log_event("controller", "obstacle_avoidance_applied", {
                                "frame_id": frame_id,
                                "obstacle_score": round(obstacle_score, 4),
                                "movement_before": round(original_movement, 4),
                                "movement_after": round(movement, 4),
                                "rotation_before": round(original_rotation, 4),
                                "rotation_after": round(rotation, 4),
                                "vz_before": round(original_vz, 4),
                                "vz_after": round(vz, 4),
                            })
                        span.set_attribute("obstacle.blocked", True)
                        span.set_attribute("obstacle.score", round(obstacle_score, 4))
                    else:
                        # Recuperación de altitud cuando ya no hay obstáculo frontal.
                        if phase != "gps_landing" and POSITION_PACKET_FLOATS == 3 and pos_z > OBSTACLE_ALT_CRUISE_Z:
                            vz = min(vz, -0.06)
                        span.set_attribute("obstacle.blocked", False)

                ml.log_agent_decision(frame_id, movement, rotation, latency, f"{answer} | guard={guard_reason}")

            # ---- Calcular y enviar comando (con tracing) ----
            # Aplicar divisor de velocidad ("cámara lenta") para compensar latencia LLM
            vx = (movement * MAX_FORWARD) / SPEED_DIVISOR
            vy = 0.0
            vz = vz / SPEED_DIVISOR
            yaw = (rotation * MAX_YAW) / SPEED_DIVISOR

            cmd = f"{vx} {vy} {vz} {yaw}\n"

            with tracer.start_as_current_span("command_send") as span:
                try:
                    sock.send(cmd.encode())
                    ml.log_command_sent(frame_id, vx, vy, vz, yaw)
                    mission_state.log_event("drone", "command_sent", {
                        "frame_id": frame_id,
                        "vx": vx, "vy": vy, "vz": vz, "yaw": yaw,
                        "pos_x": round(pos_x, 4),
                        "pos_y": round(pos_y, 4),
                        "pos_z": round(pos_z, 4),
                    })
                    span.set_attribute("cmd.vx", vx)
                    span.set_attribute("cmd.yaw", yaw)
                except BrokenPipeError:
                    ml.log_connection_event("connection_lost", {"reason": "BrokenPipe during command send"})
                    mission_state.log_event("system", "connection_lost", {
                        "reason": "BrokenPipe"
                    })
                    sock.close()
                    sock = connect_to_webots()

            # ---- Actualizar resumen periódicamente ----
            if ENABLE_SUMMARY_UPDATES and hybrid_memory.should_update_summary(SUMMARY_UPDATE_INTERVAL):
                ml.log_system("Iniciando actualización de resumen estratégico...")
                try:
                    prev_events_summarized = hybrid_memory.events_summarized_count
                    summary_llm = get_summary_llm()
                    t_mem_start = time.time()
                    hybrid_memory.update_summary(summary_llm)
                    mem_latency = time.time() - t_mem_start
                    if hybrid_memory.events_summarized_count > prev_events_summarized:
                        ml.log_memory_update(len(hybrid_memory.strategic_summary), hybrid_memory.events_summarized_count, mem_latency)
                    else:
                        ml.log_system(
                            "Actualización de resumen omitida por error/contexto del modelo.",
                            level=logging.WARNING,
                            data={"latency_s": round(mem_latency, 4)},
                        )
                except Exception as e:
                    ml.log_error("hybrid_memory", f"Error actualizando resumen: {e}")

            # ---- Limitar log de eventos (evitar memory leak) ----
            if mission_state.total_events > 200:
                removed = mission_state.clear_old_events(keep_last_n=100)
                ml.log_system(f"Eventos antiguos limpiados: {removed}", level=logging.DEBUG)

        except ConnectionError as e:
            ml.log_connection_event("connection_error", {"error": str(e), "action": "reconnecting"})
            mission_state.log_event("system", "connection_error", {"error": str(e)})
            try:
                sock.close()
            except Exception:
                pass
            sock = connect_to_webots()

        except TimeoutError as e:
            ml.log_connection_event("socket_timeout", {"error": str(e), "action": "reconnecting"})
            mission_state.log_event("system", "socket_timeout", {"error": str(e)})
            try:
                sock.close()
            except Exception:
                pass
            sock = connect_to_webots()

        except KeyboardInterrupt:
            ml.log_system("Interrumpido por usuario", level=logging.WARNING)
            mission_state.log_event("system", "mission_stopped", {"reason": "user"})
            if use_async_worker:
                _llm_worker.stop()
            hybrid_memory.save_summary()
            ml.log_system("Resumen guardado en disco.")
            ml.log_session_summary()
            break

        except Exception as e:
            import traceback
            tb_str = traceback.format_exc()
            ml.log_error("main_loop", e, tb_str)
            mission_state.log_event("system", "unexpected_error", {"error": str(e)})
            time.sleep(1)


# ================= ENTRY POINT =================
if __name__ == "__main__":
    main()