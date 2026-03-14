"""
mission_state.py — Estado global de la misión.

Mantiene un log estructurado de todos los eventos de la misión y permite
serializar el estado completo como payload JSON para enviarlo al LLM.
Incluye tracking de la posición X/Y del dron y métricas OpenTelemetry.
"""

import json
from datetime import datetime, timezone
from typing import Any

from telemetry import get_meter


class MissionState:
    """
    Estado global de la misión del dron.

    Almacena un log cronológico de eventos con estructura:
        {actor, action, data, timestamp}

    Permite serializar el estado completo como JSON para que un LLM
    pueda tomar decisiones basadas en toda la historia de la misión.
    """

    def __init__(self, mission_name: str = "drone_mission"):
        """
        Args:
            mission_name: Nombre identificativo de la misión.
        """
        self.mission_name: str = mission_name
        self.start_time: str = datetime.now(timezone.utc).isoformat()
        self.event_log: list[dict[str, Any]] = []
        self.metadata: dict[str, Any] = {}

        # ---------- Posición X/Y ----------
        self.position: dict[str, float] = {"x": 0.0, "y": 0.0}
        self.position_history: list[dict[str, Any]] = []
        self._max_position_history: int = 100

        # ---------- Métricas OTel ----------
        meter = get_meter("mission_state")
        self._events_counter = meter.create_counter(
            name="mission.events_logged",
            description="Total mission events logged",
            unit="1",
        )

    # ---------- Registro de eventos ----------

    def log_event(self, actor: str, action: str, data: Any = None) -> dict:
        """
        Registra un evento en el log de la misión.

        Args:
            actor:  Quién genera el evento ("drone", "vlm", "agent", "user", etc.)
            action: Tipo de acción ("frame_received", "decision_made", "command_sent", etc.)
            data:   Datos asociados al evento (dict, str, número, etc.)

        Returns:
            El evento registrado.
        """
        event = {
            "id": len(self.event_log),
            "actor": actor,
            "action": action,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.event_log.append(event)
        self._events_counter.add(1, {"actor": actor, "action": action})
        return event

    # ---------- Consultas ----------

    def get_recent_events(self, n: int = 10) -> list[dict]:
        """Devuelve los últimos N eventos del log."""
        return self.event_log[-n:]

    def get_events_by_actor(self, actor: str) -> list[dict]:
        """Filtra eventos por actor."""
        return [e for e in self.event_log if e["actor"] == actor]

    def get_events_by_action(self, action: str) -> list[dict]:
        """Filtra eventos por tipo de acción."""
        return [e for e in self.event_log if e["action"] == action]

    @property
    def total_events(self) -> int:
        """Número total de eventos registrados."""
        return len(self.event_log)

    # ---------- Posición X/Y ----------

    def update_position(self, x: float, y: float) -> None:
        """
        Actualiza la posición actual del dron y la agrega al historial.

        Args:
            x: Coordenada X (GPS).
            y: Coordenada Y (GPS).
        """
        self.position = {"x": round(x, 4), "y": round(y, 4)}
        self.position_history.append({
            "x": self.position["x"],
            "y": self.position["y"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        # Limitar historial
        if len(self.position_history) > self._max_position_history:
            self.position_history = self.position_history[-self._max_position_history:]

    def get_recent_positions(self, n: int = 10) -> list[dict]:
        """Devuelve las últimas N posiciones."""
        return self.position_history[-n:]

    # ---------- Serialización ----------

    def to_payload(self, max_events: int | None = None) -> dict:
        """
        Serializa el estado completo de la misión como un dict (JSON-ready).
        Este payload se puede enviar directamente a un LLM.

        Args:
            max_events: Si se especifica, solo incluye los últimos N eventos.

        Returns:
            Dict con toda la información de la misión.
        """
        events = self.event_log
        if max_events is not None:
            events = self.event_log[-max_events:]

        return {
            "mission_name": self.mission_name,
            "start_time": self.start_time,
            "total_events": self.total_events,
            "events_included": len(events),
            "current_position": self.position,
            "metadata": self.metadata,
            "events": events,
        }

    def to_json(self, max_events: int | None = None, indent: int = 2) -> str:
        """Serializa el estado como string JSON."""
        return json.dumps(self.to_payload(max_events), indent=indent, ensure_ascii=False)

    # ---------- Gestión ----------

    def clear_old_events(self, keep_last_n: int = 50) -> int:
        """
        Elimina eventos antiguos, manteniendo solo los últimos N.
        Util para limitar uso de memoria.

        Args:
            keep_last_n: Número de eventos recientes a conservar.

        Returns:
            Número de eventos eliminados.
        """
        if len(self.event_log) <= keep_last_n:
            return 0
        removed = len(self.event_log) - keep_last_n
        self.event_log = self.event_log[-keep_last_n:]
        return removed

    def set_metadata(self, key: str, value: Any) -> None:
        """Establece un valor de metadata de la misión."""
        self.metadata[key] = value

    def __repr__(self) -> str:
        return (
            f"MissionState(name='{self.mission_name}', "
            f"events={self.total_events}, "
            f"started='{self.start_time}')"
        )


# ================= TEST RÁPIDO =================
if __name__ == "__main__":
    from telemetry import init_telemetry
    init_telemetry("test_mission_state", enable_console=False)

    print("=" * 50)
    print("Test de MissionState")
    print("=" * 50)

    state = MissionState("test_mission")
    print(f"\n[1] Estado creado: {state}")

    state.log_event("drone", "takeoff", {"altitude": 1.0})
    state.log_event("vlm", "frame_received", {"frame_id": 1, "size": "640x480"})
    state.log_event("agent", "decision_made", {"movement": 0.5, "rotation": 0.2})
    state.log_event("drone", "command_sent", {"vx": 0.25, "yaw": 0.16})
    state.log_event("vlm", "frame_received", {"frame_id": 2, "size": "640x480"})
    state.log_event("agent", "decision_made", {"movement": 0.8, "rotation": -0.3})
    print(f"\n[2] Eventos registrados: {state.total_events}")

    # --- Test posición X/Y ---
    state.update_position(1.23, 4.56)
    state.update_position(1.30, 4.60)
    print(f"\n[3] Posición actual: {state.position}")
    print(f"    Historial: {len(state.position_history)} registros")

    recent = state.get_recent_events(3)
    print(f"\n[4] Últimos 3 eventos:")
    for e in recent:
        print(f"    {e['actor']:8s} | {e['action']:20s} | {e['data']}")

    payload = state.to_payload(max_events=4)
    print(f"\n[5] Payload (últimos 4 eventos):")
    print(f"    mission:  {payload['mission_name']}")
    print(f"    total:    {payload['total_events']}")
    print(f"    included: {payload['events_included']}")
    print(f"    position: {payload['current_position']}")

    print(f"\n[6] JSON completo:")
    print(state.to_json(max_events=2))

    removed = state.clear_old_events(keep_last_n=3)
    print(f"\n[7] Eventos eliminados: {removed}, quedan: {state.total_events}")

    print("\n[OK] Todos los tests pasaron correctamente.")
