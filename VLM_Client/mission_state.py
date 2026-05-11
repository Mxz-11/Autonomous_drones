import json
import os
import threading
from datetime import datetime, timezone
from typing import Any

from telemetry import get_meter

MISSION_EVENT_DEBUG = os.getenv("VLM_LOG_MISSION_EVENTS", "0") == "1"


class MissionState:
    def __init__(self, mission_name: str = "drone_mission"):
        self._lock = threading.Lock()

        self.mission_name: str = mission_name
        self.start_time: str = datetime.now(timezone.utc).isoformat()
        self.event_log: list[dict[str, Any]] = []
        self.metadata: dict[str, Any] = {}
        self._next_event_id: int = 0

        self.position: dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.position_history: list[dict[str, Any]] = []
        self._max_position_history: int = 100

        meter = get_meter("mission_state")
        self._events_counter = meter.create_counter(
            name="mission.events_logged",
            description="Total mission events logged",
            unit="1",
        )

    def log_event(self, actor: str, action: str, data: Any = None) -> dict:
        with self._lock:
            event = {
                "id": self._next_event_id,
                "actor": actor,
                "action": action,
                "data": data,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._next_event_id += 1
            self.event_log.append(event)
        self._events_counter.add(1, {"actor": actor, "action": action})

        if MISSION_EVENT_DEBUG:
            from advanced_logger import get_logger
            get_logger("mission_state").debug(f"Event #{event['id']} [{actor}] {action}")

        return event

    def get_recent_events(self, n: int = 10) -> list[dict]:
        with self._lock:
            return list(self.event_log[-n:])

    def get_events_by_actor(self, actor: str) -> list[dict]:
        with self._lock:
            return [e for e in self.event_log if e["actor"] == actor]

    def get_events_by_action(self, action: str) -> list[dict]:
        with self._lock:
            return [e for e in self.event_log if e["action"] == action]

    @property
    def total_events(self) -> int:
        with self._lock:
            return len(self.event_log)

    def update_position(self, x: float, y: float, z: float = 0.0) -> None:
        with self._lock:
            self.position = {"x": round(x, 4), "y": round(y, 4), "z": round(z, 4)}
            self.position_history.append({
                "x": self.position["x"],
                "y": self.position["y"],
                "z": self.position["z"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            if len(self.position_history) > self._max_position_history:
                self.position_history = self.position_history[-self._max_position_history:]

    def get_recent_positions(self, n: int = 10) -> list[dict]:
        with self._lock:
            return list(self.position_history[-n:])

    def to_payload(self, max_events: int | None = None) -> dict:
        with self._lock:
            events = list(self.event_log)
            if max_events is not None:
                events = events[-max_events:]
            position_copy = dict(self.position)

        return {
            "mission_name": self.mission_name,
            "start_time": self.start_time,
            "total_events": len(events),
            "events_included": len(events),
            "current_position": position_copy,
            "metadata": self.metadata,
            "events": events,
        }

    def to_json(self, max_events: int | None = None, indent: int = 2) -> str:
        return json.dumps(self.to_payload(max_events), indent=indent, ensure_ascii=False)

    def clear_old_events(self, keep_last_n: int = 50) -> int:
        # Los IDs siguen siendo monotónicos tras la poda; no se reinician.
        with self._lock:
            if len(self.event_log) <= keep_last_n:
                return 0
            removed = len(self.event_log) - keep_last_n
            self.event_log = self.event_log[-keep_last_n:]
            return removed

    def set_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    def __repr__(self) -> str:
        return (
            f"MissionState(name='{self.mission_name}', "
            f"events={self.total_events}, "
            f"started='{self.start_time}')"
        )


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
