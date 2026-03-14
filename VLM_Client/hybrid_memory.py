"""
hybrid_memory.py — Memoria híbrida para el agente de misión.

Combina:
  1. Resumen estratégico persistente en disco (JSON).
  2. Buffer de eventos recientes (referencia al MissionState).
  3. Actualización automática del resumen usando un LLM.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from mission_state import MissionState
from telemetry import get_tracer


DEFAULT_SUMMARY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "mission_summary.json"
)


class HybridMemory:
    """
    Memoria híbrida que combina un resumen estratégico persistente
    en disco con un buffer de eventos recientes del MissionState.

    El resumen puede actualizarse automáticamente invocando un LLM
    que condensa los eventos recientes en una narrativa de alto nivel.
    """

    def __init__(
        self,
        mission_state: MissionState,
        summary_path: str = DEFAULT_SUMMARY_PATH,
        recent_events_count: int = 20,
    ):
        """
        Args:
            mission_state:       Referencia al estado global de la misión.
            summary_path:        Ruta del fichero JSON para persistir el resumen.
            recent_events_count: Número de eventos recientes a mantener en el buffer.
        """
        self.mission_state = mission_state
        self.summary_path = summary_path
        self.recent_events_count = recent_events_count

        self.strategic_summary: str = ""
        self.last_summary_update: str | None = None
        self.events_summarized_count: int = 0

        self.load_summary()

    # ---------- Persistencia en disco ----------
    def load_summary(self) -> bool:
        """
        Carga el resumen estratégico desde disco.

        Returns:
            True si se cargó correctamente, False si no existía.
        """
        if not os.path.exists(self.summary_path):
            return False

        try:
            with open(self.summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.strategic_summary = data.get("summary", "")
            self.last_summary_update = data.get("last_update", None)
            self.events_summarized_count = data.get("events_summarized", 0)
            return True
        except (json.JSONDecodeError, IOError) as e:
            print(f"[WARN] No se pudo cargar el resumen: {e}")
            return False

    def save_summary(self) -> None:
        """Guarda el resumen estratégico en disco como JSON."""
        data = {
            "mission_name": self.mission_state.mission_name,
            "summary": self.strategic_summary,
            "last_update": self.last_summary_update,
            "events_summarized": self.events_summarized_count,
        }
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ---------- Actualizar resumen con LLM ----------
    def update_summary(self, llm: ChatOpenAI) -> str:
        """
        Usa un LLM para condensar los eventos recientes en un resumen
        estratégico actualizado. El resultado se persiste en disco.

        Args:
            llm: Modelo de lenguaje para generar el resumen.

        Returns:
            El nuevo resumen estratégico generado.
        """
        recent = self.mission_state.get_recent_events(self.recent_events_count)

        if not recent:
            return self.strategic_summary  

        events_text = json.dumps(recent, indent=2, ensure_ascii=False)

        messages = [
            SystemMessage(content=(
                "You are a drone mission assistant. Your task is to generate a concise "
                "strategic summary of the mission based on the previous summary and "
                "recent events. The summary must:\n"
                "- Capture the current status of the mission.\n"
                "- Identify patterns or trends.\n"
                "- Point out issues or anomalies.\n"
                "- Be useful for making the next decision.\n"
                "Respond ONLY with the summary, without additional explanations."
            )),
            HumanMessage(content=(
                f"PREVIOUS SUMMARY:\n{self.strategic_summary or '(No previous summary)'}\n\n"
                f"RECENT EVENTS ({len(recent)} events):\n{events_text}\n\n"
                f"Generate the updated strategic summary:"
            )),
        ]

        try:
            with get_tracer("hybrid_memory").start_as_current_span("update_summary") as span:
                response = llm.invoke(messages)
                self.strategic_summary = response.content.strip()
                self.last_summary_update = datetime.now(timezone.utc).isoformat()
                self.events_summarized_count = self.mission_state.total_events

                span.set_attribute("summary.length", len(self.strategic_summary))
                span.set_attribute("events.summarized", self.events_summarized_count)

                self.save_summary()
                return self.strategic_summary

        except Exception as e:
            print(f"[ERROR] No se pudo actualizar el resumen: {e}")
            return self.strategic_summary

    def update_summary_manual(self, summary_text: str) -> None:
        """
        Actualiza el resumen estratégico manualmente (sin usar LLM).

        Args:
            summary_text: Texto del nuevo resumen.
        """
        self.strategic_summary = summary_text
        self.last_summary_update = datetime.now(timezone.utc).isoformat()
        self.events_summarized_count = self.mission_state.total_events
        self.save_summary()

    # ---------- Contexto combinado ----------
    def get_context(self) -> dict[str, Any]:
        """
        Genera el contexto completo combinando resumen estratégico
        y eventos recientes. Este dict se puede pasar al agente
        para que tenga toda la información disponible.

        Returns:
            Dict con:
              - strategic_summary: Resumen de alto nivel.
              - recent_events: Últimos N eventos.
              - mission_info: Metadata de la misión.
        """
        return {
            "strategic_summary": self.strategic_summary or "(Sin resumen disponible)",
            "last_summary_update": self.last_summary_update,
            "current_position": self.mission_state.position,
            "recent_events": self.mission_state.get_recent_events(
                self.recent_events_count
            ),
            "mission_info": {
                "name": self.mission_state.mission_name,
                "total_events": self.mission_state.total_events,
                "start_time": self.mission_state.start_time,
                "metadata": self.mission_state.metadata,
            },
        }

    def get_context_text(self) -> str:
        """
        Genera el contexto como texto plano, ideal para inyectar
        en el prompt del agente.
        """
        ctx = self.get_context()
        recent_text = "\n".join(
            f"  [{e['timestamp']}] {e['actor']}: {e['action']} → {e.get('data', '')}"
            for e in ctx["recent_events"]
        )
        return (
            f"=== RESUMEN ESTRATÉGICO ===\n"
            f"{ctx['strategic_summary']}\n\n"
            f"=== POSICIÓN ACTUAL ===\n"
            f"X={ctx['current_position']['x']}, Y={ctx['current_position']['y']}\n\n"
            f"=== EVENTOS RECIENTES ({len(ctx['recent_events'])}) ===\n"
            f"{recent_text}\n\n"
            f"=== INFO MISIÓN ===\n"
            f"Nombre: {ctx['mission_info']['name']}\n"
            f"Total eventos: {ctx['mission_info']['total_events']}\n"
            f"Inicio: {ctx['mission_info']['start_time']}\n"
        )

    def should_update_summary(self, every_n_events: int = 20) -> bool:
        """
        Indica si es momento de actualizar el resumen estratégico.
        Devuelve True si han pasado al menos N eventos desde
        la última actualización.

        Args:
            every_n_events: Cada cuántos eventos nuevos actualizar.
        """
        new_events = self.mission_state.total_events - self.events_summarized_count
        return new_events >= every_n_events

    def __repr__(self) -> str:
        return (
            f"HybridMemory(summary_len={len(self.strategic_summary)}, "
            f"events_summarized={self.events_summarized_count})"
        )


# ================= TEST RÁPIDO =================
if __name__ == "__main__":
    from telemetry import init_telemetry
    init_telemetry("test_hybrid_memory", enable_console=False)

    print("=" * 50)
    print("Test de HybridMemory (sin LLM)")
    print("=" * 50)

    state = MissionState("test_hybrid_memory")
    state.log_event("drone", "takeoff", {"altitude": 1.0})
    state.log_event("vlm", "frame_received", {"frame_id": 1})
    state.log_event("agent", "decision_made", {"movement": 0.5, "rotation": 0.1})
    state.log_event("drone", "command_sent", {"vx": 0.25, "yaw": 0.08})
    state.log_event("vlm", "frame_received", {"frame_id": 2})

    state.update_position(1.5, -2.3)

    test_path = "/tmp/test_mission_summary.json"
    memory = HybridMemory(state, summary_path=test_path, recent_events_count=10)
    print(f"\n[1] Memoria creada: {memory}")
    memory.update_summary_manual(
        "El dron ha despegado a 1m de altitud y ha comenzado a recibir "
        "frames del simulador. Se han tomado 2 decisiones de navegación."
    )
    print(f"\n[2] Resumen actualizado: {memory.strategic_summary[:80]}...")

    ctx = memory.get_context()
    print(f"\n[3] Contexto combinado:")
    print(f"    Resumen: {ctx['strategic_summary'][:60]}...")
    print(f"    Posición: {ctx['current_position']}")
    print(f"    Eventos recientes: {len(ctx['recent_events'])}")
    print(f"    Misión: {ctx['mission_info']['name']}")

    print(f"\n[4] Contexto como texto:")
    print(memory.get_context_text())

    memory2 = HybridMemory(state, summary_path=test_path)
    assert memory2.strategic_summary == memory.strategic_summary
    print(f"[5] Persistencia verificada: resumen cargado desde disco OK")

    print(f"\n[6] ¿Actualizar resumen? (every_n=3): {memory.should_update_summary(3)}")
    print(f"    ¿Actualizar resumen? (every_n=50): {memory.should_update_summary(50)}")

    os.remove(test_path)
    print("\n[OK] Todos los tests pasaron correctamente.")
