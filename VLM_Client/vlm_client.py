"""
vlm_client.py — Cliente VLM principal con agente LangChain.

Punto de entrada del sistema. Conecta a Webots por socket, recibe frames
de la cámara del dron, y usa un agente LangChain con herramientas para
decidir los movimientos basándose en el estado completo de la misión.

Arquitectura:
  ┌─────────┐    socket     ┌──────────────┐    LangChain    ┌──────────┐
  │ Webots  │◄──────────────│ vlm_client   │────────────────►│ LM Studio│
  │ (dron)  │   cmd/frames  │ (agente)     │   decisiones    │ (LLM)    │
  └─────────┘               └──────┬───────┘                 └──────────┘
                                   │
                        ┌──────────┼──────────┐
                        ▼          ▼          ▼
                 MissionState  HybridMem  AgentTools
"""

import socket
import struct
import time
import base64
import io
import re
import json

import numpy as np
from PIL import Image

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

from llm_config import get_decision_llm, get_summary_llm
from mission_state import MissionState
from hybrid_memory import HybridMemory
from agent_tools import ALL_TOOLS, init_tools
from telemetry import init_telemetry, get_tracer, get_meter


# ================= CONFIGURACIÓN =================

WEBOTS_IP = "127.0.0.1"
WEBOTS_PORT = 9002

MAX_FORWARD = 0.5
MAX_YAW = 0.8

SUMMARY_UPDATE_INTERVAL = 20
MAX_RECENT_EVENTS = 30

# Activar recepción de posición X/Y desde el controlador.
# Requiere que crazyflie.c esté recompilado con el envío de GPS.
# Si usas el controlador antiguo, pon esto a False.
POSITION_ENABLED = True

# ================================================


# ================= SOCKET a Webots =================

def connect_to_webots() -> socket.socket:
    """Conecta al servidor de Webots con reintentos automáticos."""
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((WEBOTS_IP, WEBOTS_PORT))
            print("[OK] Conectado a Webots")
            return sock
        except ConnectionRefusedError:
            print("[WARN] Conexión rechazada, reintentando en 2s...")
            time.sleep(2)


def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Recibe exactamente N bytes del socket."""
    data = b""
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            raise ConnectionError("Socket cerrado")
        data += packet
    return data


# ================= PROMPT DEL SISTEMA =================

SYSTEM_PROMPT = (
    "You are the intelligent control system of an autonomous drone.\n\n"
    "Your mission is to pilot the drone following a figure-eight trajectory. "
    "You receive frames from the drone's camera and your GPS position (X, Y), "
    "and you must decide the movements.\n\n"
    "You have access to tools to log events, check the mission status, "
    "update memory, and generate decisions.\n\n"
    "RULES:\n"
    "1. Always maintain forward movement.\n"
    "2. Alternate rotation to form two opposite loops.\n"
    "3. Use your X/Y position to estimate your trajectory.\n"
    "4. ALWAYS respond with movement=X, rotation=Y at the end.\n\n"
    "Final response format:\n"
    "movement=<value between 0 and 1>, rotation=<value between -1 and 1>"
)


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
    print("\n" + "=" * 60)
    print("  INICIALIZANDO SISTEMA DE AGENTE LANGCHAIN")
    print("=" * 60)

    print("\n[1/5] Configurando modelos LLM...")
    decision_llm = get_decision_llm()
    summary_llm = get_summary_llm()
    print(f"      Modelo decisión: {decision_llm.model_name}")
    print(f"      Modelo resumen:  {summary_llm.model_name}")

    print("[2/5] Creando estado de misión...")
    mission_state = MissionState("drone_figure8_mission")
    mission_state.set_metadata("mission_type", "figure_eight")
    mission_state.set_metadata("webots_port", WEBOTS_PORT)
    print(f"      Misión: {mission_state.mission_name}")

    print("[3/5] Configurando memoria híbrida...")
    hybrid_memory = HybridMemory(
        mission_state=mission_state,
        recent_events_count=MAX_RECENT_EVENTS,
    )
    print(f"      Buffer eventos: {MAX_RECENT_EVENTS}")
    print(f"      Resumen previo: {'Sí' if hybrid_memory.strategic_summary else 'No'}")

    print("[4/5] Registrando herramientas del agente...")
    init_tools(mission_state, hybrid_memory, decision_llm, summary_llm)
    for t in ALL_TOOLS:
        print(f"      → {t.name}")

    print("[5/5] Creando agente LangGraph (ReAct)...")

    agent = create_react_agent(
        model=decision_llm,
        tools=ALL_TOOLS,
        prompt=SYSTEM_PROMPT,
    )

    print("\n[OK] Sistema de agente inicializado correctamente.")
    print("=" * 60 + "\n")

    return agent, mission_state, hybrid_memory, decision_llm


# ================= PROCESAMIENTO DE FRAME =================

def frame_to_base64(frame_bytes: bytes, width: int, height: int) -> str:
    """
    Convierte los bytes crudos de un frame RGBA a base64 PNG.

    Args:
        frame_bytes: Bytes RGBA del frame.
        width: Ancho del frame.
        height: Alto del frame.

    Returns:
        String base64 de la imagen PNG.
    """
    frame = np.frombuffer(frame_bytes, np.uint8).reshape((height, width, 4))
    frame = frame[:, :, :3]  
    image = Image.fromarray(frame)

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def parse_movement(answer: str) -> tuple[float, float]:
    """
    Extrae movement y rotation de la respuesta del agente.

    Args:
        answer: Texto de respuesta del agente.

    Returns:
        Tupla (movement, rotation) con valores numéricos.
    """
    answer_lower = answer.lower().strip()

    movement = 0.0
    rotation = 0.0

    m = re.search(r"movement\s*=\s*([0-9.]+)", answer_lower)
    if m:
        movement = float(m.group(1))

    r = re.search(r"rotation\s*=\s*(-?[0-9.]+)", answer_lower)
    if r:
        rotation = float(r.group(1))

    movement = max(0.0, min(1.0, movement))
    rotation = max(-1.0, min(1.0, rotation))

    return movement, rotation


# ================= INVOCACIÓN DEL AGENTE =================

def invoke_agent(
    agent,
    mission_state: MissionState,
    hybrid_memory: HybridMemory,
    img_b64: str,
    frame_id: int,
) -> tuple[float, float]:
    """
    Invoca al agente LangGraph con el frame actual y el contexto de la misión.

    Args:
        agent: Agente ReAct de LangGraph.
        mission_state: Estado global de la misión.
        hybrid_memory: Memoria híbrida del agente.
        img_b64: Imagen del frame en base64.
        frame_id: ID secuencial del frame.

    Returns:
        Tupla (movement, rotation) decidida por el agente.
    """
    mission_state.log_event("vlm", "frame_received", {
        "frame_id": frame_id,
        "timestamp": time.time(),
    })
    context = hybrid_memory.get_context_text()

    user_message = (
        f"Frame #{frame_id} recibido del dron.\n\n"
        f"CONTEXTO DE MISIÓN:\n{context}\n\n"
        f"Basándote en el contexto y el estado de la misión, "
        f"decide los valores de movement y rotation para el dron.\n"
        f"Recuerda: movement entre 0 y 1, rotation entre -1 y 1."
    )

    try:
        result = agent.invoke({
            "messages": [HumanMessage(content=user_message)],
        })

        # Extraer la última respuesta del agente
        messages = result.get("messages", [])
        answer = messages[-1].content if messages else "movement=0.5, rotation=0.0"
        movement, rotation = parse_movement(answer)

        mission_state.log_event("agent", "decision_output", {
            "frame_id": frame_id,
            "movement": movement,
            "rotation": rotation,
            "raw_answer": answer[:200],
        })

        return movement, rotation

    except Exception as e:
        print(f"[ERROR] Error del agente: {e}") 
        mission_state.log_event("agent", "error", {
            "frame_id": frame_id,
            "error": str(e)[:200],
        })
        return 0.5, 0.0


# ================= INVOCACIÓN DIRECTA (sin agente) =================

def invoke_direct_llm(
    decision_llm,
    mission_state: MissionState,
    hybrid_memory: HybridMemory,
    img_b64: str,
    frame_id: int,
) -> tuple[float, float]:
    """
    Modo alternativo: invoca al LLM directamente SIN pasar por el agente.
    Útil como fallback o para comparar rendimiento.

    Envía la imagen + contexto directamente al modelo multimodal.
    """
    context = hybrid_memory.get_context_text()
    pos = mission_state.position

    messages = [
        SystemMessage(content=(
            "You are the control system of a drone. "
            "Decide how it should move using two values:\n"
            "- movement: number between 0 and 1 (forward velocity)\n"
            "- rotation: number between -1 and 1 (rotation, neg=left)\n\n"
            "Mission: fly following a figure-eight trajectory.\n"
            "Respond ONLY: movement=<value>, rotation=<value>"
        )),
        HumanMessage(content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
            },
            {
                "type": "text",
                "text": (
                    f"Frame #{frame_id}\n"
                    f"GPS Position: X={pos['x']}, Y={pos['y']}\n"
                    f"Context:\n{context}\n"
                    f"Decide movement and rotation."
                ),
            },
        ]),
    ]

    try:
        response = decision_llm.invoke(messages)
        answer = response.content.strip()
        movement, rotation = parse_movement(answer)

        mission_state.log_event("vlm_direct", "decision_made", {
            "frame_id": frame_id,
            "movement": movement,
            "rotation": rotation,
        })

        return movement, rotation

    except Exception as e:
        print(f"[ERROR] LLM directo falló: {e}")
        return 0.5, 0.0


# ================= LOOP PRINCIPAL =================

def main():
    """
    Loop principal del cliente VLM.

    1. Inicializa telemetría OpenTelemetry.
    2. Inicializa el sistema de agente.
    3. Conecta a Webots por socket.
    4. Para cada frame recibido:
       a. Convierte a base64.
       b. Recibe posición X/Y del dron.
       c. Invoca al agente (o LLM directo como fallback).
       d. Parsea movement/rotation.
       e. Envía comando al dron.
       f. Actualiza la memoria periódicamente.
    """
    USE_AGENT_MODE = True

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

    agent, mission_state, hybrid_memory, decision_llm = create_agent_system()

    mission_state.log_event("system", "mission_started", {
        "mode": "agent" if USE_AGENT_MODE else "direct",
        "webots_ip": WEBOTS_IP,
        "webots_port": WEBOTS_PORT,
    })

    sock = connect_to_webots()
    mission_state.log_event("system", "webots_connected", {
        "ip": WEBOTS_IP, "port": WEBOTS_PORT
    })

    frame_id = 0

    print("\n[LOOP] Iniciando bucle principal...")
    print(f"[MODE] {'AGENTE LangChain' if USE_AGENT_MODE else 'LLM Directo'}\n")

    while True:
        try:
            # ---- Solicitar frame al controlador (pull-based) ----
            try:
                sock.send(b"FRAME\n")
            except BrokenPipeError:
                print("[WARN] BrokenPipe al pedir frame, reconectando...")
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
                    print(f"[ERROR] Header inválido: {w}x{h}, reconectando...")
                    mission_state.log_event("system", "invalid_header", {
                        "width": w, "height": h
                    })
                    sock.close()
                    sock = connect_to_webots()
                    continue

                img_bytes = recv_exact(sock, w * h * 4)
                img_b64 = frame_to_base64(img_bytes, w, h)

                # ---- Recibir posición X/Y (8 bytes: 2 floats) ----
                pos_x, pos_y = 0.0, 0.0
                if POSITION_ENABLED:
                    pos_bytes = recv_exact(sock, 8)
                    pos_x, pos_y = struct.unpack("ff", pos_bytes)
                    mission_state.update_position(pos_x, pos_y)

                frame_id += 1
                frame_counter.add(1)
                pos_x_gauge.set(pos_x)
                pos_y_gauge.set(pos_y)

                span.set_attribute("frame.id", frame_id)
                span.set_attribute("frame.width", w)
                span.set_attribute("frame.height", h)
                span.set_attribute("drone.pos_x", pos_x)
                span.set_attribute("drone.pos_y", pos_y)

            # ---- Decidir movimiento (con tracing) ----
            with tracer.start_as_current_span("agent_invoke") as span:
                t0 = time.time()

                if USE_AGENT_MODE:
                    movement, rotation = invoke_agent(
                        agent, mission_state, hybrid_memory,
                        img_b64, frame_id,
                    )
                else:
                    movement, rotation = invoke_direct_llm(
                        decision_llm, mission_state, hybrid_memory,
                        img_b64, frame_id,
                    )

                latency = time.time() - t0
                decision_histogram.record(latency)
                span.set_attribute("decision.movement", movement)
                span.set_attribute("decision.rotation", rotation)
                span.set_attribute("decision.latency_s", round(latency, 3))

            # ---- Calcular y enviar comando (con tracing) ----
            vx = movement * MAX_FORWARD
            vy = 0.0
            vz = 0.0
            yaw = rotation * MAX_YAW

            cmd = f"{vx} {vy} {vz} {yaw}\n"

            with tracer.start_as_current_span("command_send") as span:
                try:
                    sock.send(cmd.encode())
                    print(
                        f"[F{frame_id:04d}] pos=({pos_x:.2f},{pos_y:.2f}) "
                        f"mov={movement:.2f} rot={rotation:.2f} → {cmd.strip()}"
                    )
                    mission_state.log_event("drone", "command_sent", {
                        "frame_id": frame_id,
                        "vx": vx, "vy": vy, "vz": vz, "yaw": yaw,
                        "pos_x": round(pos_x, 4),
                        "pos_y": round(pos_y, 4),
                    })
                    span.set_attribute("cmd.vx", vx)
                    span.set_attribute("cmd.yaw", yaw)
                except BrokenPipeError:
                    print("[WARN] BrokenPipe, reconectando...")
                    mission_state.log_event("system", "connection_lost", {
                        "reason": "BrokenPipe"
                    })
                    sock.close()
                    sock = connect_to_webots()

            # ---- Actualizar resumen periódicamente ----
            if hybrid_memory.should_update_summary(SUMMARY_UPDATE_INTERVAL):
                print("[MEM] Actualizando resumen estratégico...")
                try:
                    summary_llm = get_summary_llm()
                    hybrid_memory.update_summary(summary_llm)
                    print(f"[MEM] Resumen actualizado ({len(hybrid_memory.strategic_summary)} chars)")
                except Exception as e:
                    print(f"[MEM] Error actualizando resumen: {e}")

            # ---- Limitar log de eventos (evitar memory leak) ----
            if mission_state.total_events > 200:
                removed = mission_state.clear_old_events(keep_last_n=100)
                print(f"[MEM] Eventos antiguos limpiados: {removed}")

        except ConnectionError as e:
            print(f"[ERROR] Conexión perdida: {e}, reconectando...")
            mission_state.log_event("system", "connection_error", {"error": str(e)})
            try:
                sock.close()
            except Exception:
                pass
            sock = connect_to_webots()

        except KeyboardInterrupt:
            print("\n[EXIT] Interrumpido por usuario")
            mission_state.log_event("system", "mission_stopped", {"reason": "user"})
            hybrid_memory.save_summary()
            print("[EXIT] Resumen guardado en disco.")
            break

        except Exception as e:
            print(f"[ERROR] Error inesperado: {e}")
            import traceback
            traceback.print_exc()
            mission_state.log_event("system", "unexpected_error", {"error": str(e)})
            time.sleep(1)


# ================= ENTRY POINT =================
if __name__ == "__main__":
    main()
