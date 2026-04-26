"""
agent_tools.py — Herramientas (Tools) que el agente LangChain puede invocar.

Define funciones decoradas con @tool que permiten al agente:
  - Registrar eventos en MissionState.
  - Actualizar la memoria / resumen estratégico.
  - Generar decisiones de vuelo.
  - Enviar el payload completo a un modelo.
  - Consultar el estado de la misión.

Estas herramientas se inyectan al agente en vlm_client.py.
"""

import json
import time
from langchain_core.tools import tool

from advanced_logger import MissionLogger, get_logger

# --- Referencias globales ---
_mission_state = None
_hybrid_memory = None
_decision_llm = None
_summary_llm = None


def init_tools(mission_state, hybrid_memory, decision_llm, summary_llm):
    """
    Inicializa las referencias globales para que las tools puedan acceder
    al estado de la misión, la memoria y los LLMs.

    Debe llamarse una vez al arrancar el agente.
    """
    global _mission_state, _hybrid_memory, _decision_llm, _summary_llm
    _mission_state = mission_state
    _hybrid_memory = hybrid_memory
    _decision_llm = decision_llm
    _summary_llm = summary_llm


# =====================================================================
# HERRAMIENTA 1: Registrar evento
# =====================================================================
@tool
def register_event(actor: str, action: str, data: str = "") -> str:
    """
    Registra un evento en el log de la misión.

    Args:
        actor:  Quién genera el evento (ej: 'drone', 'vlm', 'agent', 'user').
        action: Tipo de acción (ej: 'frame_received', 'decision_made').
        data:   Datos adicionales como string JSON (ej: '{"altitude": 1.0}').

    Returns:
        Confirmación con el ID del evento registrado.
    """
    if _mission_state is None:
        return "ERROR: MissionState no inicializado."

    t0 = time.time()
    parsed_data = data
    if data:
        try:
            parsed_data = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            parsed_data = data  

    event = _mission_state.log_event(actor, action, parsed_data)
    result = f"Evento #{event['id']} registrado: [{actor}] {action}"
    
    MissionLogger().log_tool_call("register_event", {"actor": actor, "action": action}, result, time.time() - t0)
    return result


# =====================================================================
# HERRAMIENTA 2: Actualizar memoria / resumen estratégico
# =====================================================================
@tool
def update_memory() -> str:
    """
    Actualiza el resumen estratégico de la misión usando un LLM.
    El LLM condensará los eventos recientes en un resumen de alto nivel.
    El resumen se persiste automáticamente en disco.

    Returns:
        El nuevo resumen estratégico generado.
    """
    if _hybrid_memory is None or _summary_llm is None:
        return "ERROR: HybridMemory o SummaryLLM no inicializados."

    t0 = time.time()
    new_summary = _hybrid_memory.update_summary(_summary_llm)
    result = f"Resumen actualizado:\n{new_summary}"
    
    MissionLogger().log_tool_call("update_memory", None, f"Length: {len(new_summary)}", time.time() - t0)
    return result


# =====================================================================
# HERRAMIENTA 3: Generar decisión de vuelo
# =====================================================================
@tool
def generate_decision(situation: str = "") -> str:
    """
    Consulta al LLM principal con el contexto completo de la misión
    para decidir la siguiente acción del dron.

    Args:
        situation: Descripción adicional de la situación actual
                   (ej: 'El dron está cerca de un obstáculo').

    Returns:
        La decisión del LLM con movement y rotation.
    """
    if _decision_llm is None or _hybrid_memory is None:
        return "ERROR: DecisionLLM o HybridMemory no inicializados."

    from langchain_core.messages import SystemMessage, HumanMessage

    t0 = time.time()
    context = _hybrid_memory.get_context_text()

    messages = [
        SystemMessage(content=(
            "You are the control system of a drone. Based on the mission context "
            "and the current situation, decide how the drone should move.\n\n"
            "Respond ONLY in this format:\n"
            "movement=<value between 0 and 1>, rotation=<value between -1 and 1>\n\n"
            "Where:\n"
            "- movement: forward speed (0=stopped, 1=maximum)\n"
            "- rotation: rotation (negative=left, positive=right)\n"
        )),
        HumanMessage(content=(
            f"MISSION CONTEXT:\n{context}\n\n"
            f"CURRENT SITUATION:\n{situation or 'No additional information'}\n\n"
            f"What is the next action?"
        )),
    ]

    try:
        response = _decision_llm.invoke(messages)
        answer = response.content.strip()

        if _mission_state:
            _mission_state.log_event("agent", "decision_made", {"raw": answer})

        MissionLogger().log_tool_call("generate_decision", {"situation": situation}, answer, time.time() - t0)
        return answer
    except Exception as e:
        error_msg = f"ERROR al generar decisión: {e}"
        MissionLogger().log_tool_call("generate_decision", {"situation": situation}, error_msg, time.time() - t0)
        return error_msg


# =====================================================================
# HERRAMIENTA 4: Enviar payload completo al modelo
# =====================================================================
@tool
def send_full_payload(max_events: int = 50) -> str:
    """
    Envía el payload JSON completo del estado de la misión a un modelo LLM
    y devuelve su respuesta. Útil para análisis profundo del estado.

    Args:
        max_events: Número máximo de eventos a incluir en el payload.

    Returns:
        La respuesta del modelo al analizar el payload completo.
    """
    if _decision_llm is None or _mission_state is None:
        return "ERROR: DecisionLLM o MissionState no inicializados."

    from langchain_core.messages import SystemMessage, HumanMessage

    t0 = time.time()
    payload_json = _mission_state.to_json(max_events=max_events)

    messages = [
        SystemMessage(content=(
            "You are a drone mission analyst. You will receive the full "
            "mission payload and you must analyze it. Identify:\n"
            "1. Overall mission status.\n"
            "2. Movement patterns.\n"
            "3. Possible issues or anomalies.\n"
            "4. Recommendation for the next action.\n"
            "Be concise but informative."
        )),
        HumanMessage(content=f"MISSION PAYLOAD:\n{payload_json}"),
    ]

    try:
        response = _decision_llm.invoke(messages)
        answer = response.content.strip()
        MissionLogger().log_tool_call("send_full_payload", {"max_events": max_events}, answer, time.time() - t0)
        return answer
    except Exception as e:
        error_msg = f"ERROR al enviar payload: {e}"
        MissionLogger().log_tool_call("send_full_payload", {"max_events": max_events}, error_msg, time.time() - t0)
        return error_msg


# =====================================================================
# HERRAMIENTA 5: Consultar estado de la misión
# =====================================================================
@tool
def get_mission_status() -> str:
    """
    Devuelve un resumen del estado actual de la misión, incluyendo
    el número de eventos, el resumen estratégico y los últimos eventos.
    Útil para que el agente tenga visibilidad del estado antes de decidir.

    Returns:
        Texto con el estado completo de la misión.
    """
    if _mission_state is None or _hybrid_memory is None:
        return "ERROR: MissionState o HybridMemory no inicializados."

    t0 = time.time()
    status = _hybrid_memory.get_context_text()
    MissionLogger().log_tool_call("get_mission_status", None, f"Length: {len(status)}", time.time() - t0)
    return status


# =====================================================================
# Lista de todas las herramientas (se importa desde vlm_client.py)
# =====================================================================
ALL_TOOLS = [
    register_event,
    update_memory,
    get_mission_status,
    # generate_decision y send_full_payload se excluyen del agente:
    # el propio agente ReAct ya es el decisor, llamarlas generaría
    # una segunda llamada al LLM innecesaria (~30s extra por frame).
]


# ================= TEST RÁPIDO =================
if __name__ == "__main__":
    print("=" * 50)
    print("Test de Agent Tools (sin LLM)")
    print("=" * 50)

    from mission_state import MissionState
    from hybrid_memory import HybridMemory

    state = MissionState("test_tools")
    memory = HybridMemory(state, summary_path="/tmp/test_tools_summary.json")

    init_tools(state, memory, None, None)
    result = register_event.invoke({
        "actor": "test",
        "action": "test_event",
        "data": '{"key": "value"}'
    })
    print(f"\n[1] register_event: {result}")

    result = get_mission_status.invoke({})
    print(f"\n[2] get_mission_status:\n{result}")
    print(f"\n[3] Herramientas disponibles ({len(ALL_TOOLS)}):")
    for t in ALL_TOOLS:
        print(f"    - {t.name}: {t.description[:60]}...")

    import os
    os.remove("/tmp/test_tools_summary.json")
    print("\n[OK] Todos los tests pasaron correctamente.")