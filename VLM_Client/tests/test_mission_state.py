"""Tests for mission_state.py — MissionState class."""
import json
import threading
import time

import pytest

from mission_state import MissionState


@pytest.fixture
def state():
    return MissionState("test_mission")


class TestInit:
    def test_mission_name_stored(self, state):
        assert state.mission_name == "test_mission"

    def test_starts_with_no_events(self, state):
        assert state.total_events == 0
        assert state.event_log == []

    def test_default_position_is_origin(self, state):
        assert state.position == {"x": 0.0, "y": 0.0, "z": 0.0}

    def test_start_time_is_set(self, state):
        assert state.start_time is not None
        assert "T" in state.start_time


class TestLogEvent:
    def test_event_is_appended(self, state):
        state.log_event("drone", "takeoff")
        assert state.total_events == 1

    def test_event_fields_correct(self, state):
        evt = state.log_event("vlm", "frame_received", {"frame_id": 1})
        assert evt["actor"] == "vlm"
        assert evt["action"] == "frame_received"
        assert evt["data"] == {"frame_id": 1}
        assert "timestamp" in evt

    def test_event_ids_are_monotonic(self, state):
        e1 = state.log_event("a", "x")
        e2 = state.log_event("b", "y")
        e3 = state.log_event("c", "z")
        assert e1["id"] == 0
        assert e2["id"] == 1
        assert e3["id"] == 2

    def test_event_without_data(self, state):
        evt = state.log_event("drone", "landed")
        assert evt["data"] is None

    def test_multiple_events_increment_total(self, state):
        for i in range(5):
            state.log_event("agent", f"step_{i}")
        assert state.total_events == 5


class TestGetRecentEvents:
    def test_returns_last_n(self, state):
        for i in range(10):
            state.log_event("a", f"action_{i}")
        recent = state.get_recent_events(3)
        assert len(recent) == 3
        assert recent[-1]["action"] == "action_9"

    def test_returns_all_if_fewer_than_n(self, state):
        state.log_event("a", "only_one")
        recent = state.get_recent_events(10)
        assert len(recent) == 1

    def test_returns_copy_not_reference(self, state):
        state.log_event("a", "ev")
        recent = state.get_recent_events(5)
        recent.clear()
        assert state.total_events == 1


class TestGetEventsByActor:
    def test_filters_by_actor(self, state):
        state.log_event("drone", "cmd")
        state.log_event("vlm", "frame")
        state.log_event("drone", "cmd2")
        drone_events = state.get_events_by_actor("drone")
        assert len(drone_events) == 2
        assert all(e["actor"] == "drone" for e in drone_events)

    def test_empty_when_no_match(self, state):
        state.log_event("drone", "cmd")
        assert state.get_events_by_actor("agent") == []


class TestGetEventsByAction:
    def test_filters_by_action(self, state):
        state.log_event("drone", "takeoff")
        state.log_event("vlm", "frame")
        state.log_event("drone", "takeoff")
        takeoffs = state.get_events_by_action("takeoff")
        assert len(takeoffs) == 2

    def test_empty_when_no_match(self, state):
        state.log_event("drone", "takeoff")
        assert state.get_events_by_action("landing") == []


class TestUpdatePosition:
    def test_position_is_updated(self, state):
        state.update_position(1.5, 2.5, 0.8)
        assert state.position == {"x": 1.5, "y": 2.5, "z": 0.8}

    def test_coordinates_are_rounded_to_4_decimals(self, state):
        state.update_position(1.123456, -2.987654, 0.333333)
        assert state.position["x"] == pytest.approx(1.1235, abs=1e-4)
        assert state.position["y"] == pytest.approx(-2.9877, abs=1e-4)

    def test_z_defaults_to_zero(self, state):
        state.update_position(1.0, 2.0)
        assert state.position["z"] == 0.0

    def test_position_history_grows(self, state):
        state.update_position(0.0, 0.0)
        state.update_position(1.0, 0.0)
        assert len(state.position_history) == 2

    def test_position_history_capped_at_100(self, state):
        for i in range(120):
            state.update_position(float(i), 0.0)
        assert len(state.position_history) == 100

    def test_get_recent_positions_returns_last_n(self, state):
        for i in range(10):
            state.update_position(float(i), 0.0)
        recent = state.get_recent_positions(3)
        assert len(recent) == 3
        assert recent[-1]["x"] == pytest.approx(9.0)


class TestToPayload:
    def test_payload_structure(self, state):
        state.log_event("drone", "takeoff")
        payload = state.to_payload()
        assert payload["mission_name"] == "test_mission"
        assert "start_time" in payload
        assert "current_position" in payload
        assert "events" in payload

    def test_max_events_limits_events(self, state):
        for i in range(10):
            state.log_event("a", f"ev{i}")
        payload = state.to_payload(max_events=3)
        assert len(payload["events"]) == 3
        assert payload["events"][-1]["action"] == "ev9"

    def test_total_events_reflects_included(self, state):
        for i in range(5):
            state.log_event("a", f"ev{i}")
        payload = state.to_payload(max_events=2)
        assert payload["total_events"] == 2


class TestToJson:
    def test_output_is_valid_json(self, state):
        state.log_event("drone", "takeoff")
        js = state.to_json()
        parsed = json.loads(js)
        assert parsed["mission_name"] == "test_mission"


class TestClearOldEvents:
    def test_removes_excess_events(self, state):
        for i in range(10):
            state.log_event("a", f"ev{i}")
        removed = state.clear_old_events(keep_last_n=4)
        assert removed == 6
        assert state.total_events == 4

    def test_no_removal_when_below_threshold(self, state):
        state.log_event("a", "only")
        removed = state.clear_old_events(keep_last_n=10)
        assert removed == 0
        assert state.total_events == 1

    def test_keeps_latest_events(self, state):
        for i in range(5):
            state.log_event("a", f"ev{i}")
        state.clear_old_events(keep_last_n=2)
        remaining = state.get_recent_events(10)
        assert remaining[0]["action"] == "ev3"
        assert remaining[1]["action"] == "ev4"

    def test_event_ids_not_reset_after_pruning(self, state):
        for i in range(5):
            state.log_event("a", f"ev{i}")
        state.clear_old_events(keep_last_n=2)
        new_evt = state.log_event("a", "new")
        assert new_evt["id"] == 5


class TestMetadata:
    def test_update_last_command(self, state):
        state.update_last_command(0.5, -0.3)
        assert state.metadata["last_movement"] == pytest.approx(0.5, abs=0.001)
        assert state.metadata["last_rotation"] == pytest.approx(-0.3, abs=0.001)

    def test_get_last_rotation_none_initially(self, state):
        assert state.get_last_rotation() is None

    def test_get_last_rotation_after_update(self, state):
        state.update_last_command(0.0, 0.7)
        assert state.get_last_rotation() == pytest.approx(0.7, abs=0.001)

    def test_set_metadata_arbitrary_key(self, state):
        state.set_metadata("phase", "landing")
        assert state.metadata["phase"] == "landing"


class TestThreadSafety:
    def test_concurrent_log_event_no_duplicates(self, state):
        errors = []

        def log_many():
            try:
                for _ in range(50):
                    state.log_event("thread", "action")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=log_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert state.total_events == 200
        ids = [e["id"] for e in state.event_log]
        assert len(ids) == len(set(ids)), "Duplicate event IDs detected"


class TestRepr:
    def test_repr_contains_mission_name(self, state):
        assert "test_mission" in repr(state)
