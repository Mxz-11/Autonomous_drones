import json
import os
import tempfile

import pytest

from mission_state import MissionState
from hybrid_memory import HybridMemory


@pytest.fixture
def summary_path():
    f = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    f.close()
    os.unlink(f.name) 
    yield f.name
    if os.path.exists(f.name):
        os.unlink(f.name)


@pytest.fixture
def state():
    return MissionState("test_hybrid")


@pytest.fixture
def memory(state, summary_path):
    return HybridMemory(state, summary_path=summary_path, recent_events_count=10)


class TestInit:
    def test_starts_with_empty_summary(self, memory):
        assert memory.strategic_summary == ""

    def test_events_summarized_count_zero(self, memory):
        assert memory.events_summarized_count == 0

    def test_load_summary_returns_false_when_no_file(self, memory):
        assert memory.last_summary_update is None

    def test_load_existing_summary_file(self, state, summary_path):
        data = {
            "summary": "Drone navigating forward.",
            "last_update": "2024-01-01T00:00:00+00:00",
            "events_summarized": 5,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        mem = HybridMemory(state, summary_path=summary_path)
        assert mem.strategic_summary == "Drone navigating forward."
        assert mem.events_summarized_count == 5

    def test_invalid_json_does_not_crash(self, state, summary_path):
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("not_json{{{{")
        mem = HybridMemory(state, summary_path=summary_path)
        assert mem.strategic_summary == ""


class TestUpdateSummaryManual:
    def test_sets_strategic_summary(self, memory, state):
        memory.update_summary_manual("Mission in progress.")
        assert memory.strategic_summary == "Mission in progress."

    def test_sets_last_update_timestamp(self, memory):
        memory.update_summary_manual("Test summary.")
        assert memory.last_summary_update is not None

    def test_events_summarized_updated(self, memory, state):
        state.log_event("drone", "takeoff")
        state.log_event("vlm", "frame")
        memory.update_summary_manual("Updated.")
        assert memory.events_summarized_count == 2

    def test_persists_to_disk(self, memory, summary_path):
        memory.update_summary_manual("Persisted summary.")
        assert os.path.exists(summary_path)
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["summary"] == "Persisted summary."


class TestPersistenceRoundtrip:
    def test_roundtrip_preserves_summary(self, state, summary_path):
        mem1 = HybridMemory(state, summary_path=summary_path)
        mem1.update_summary_manual("Original summary text.")

        mem2 = HybridMemory(state, summary_path=summary_path)
        assert mem2.strategic_summary == "Original summary text."

    def test_roundtrip_preserves_events_count(self, state, summary_path):
        for _ in range(3):
            state.log_event("drone", "action")
        mem1 = HybridMemory(state, summary_path=summary_path)
        mem1.update_summary_manual("Summary.")

        mem2 = HybridMemory(state, summary_path=summary_path)
        assert mem2.events_summarized_count == 3


class TestGetContext:
    def test_returns_dict_with_required_keys(self, memory):
        ctx = memory.get_context()
        assert "strategic_summary" in ctx
        assert "current_position" in ctx
        assert "recent_events" in ctx
        assert "mission_info" in ctx

    def test_summary_fallback_when_empty(self, memory):
        ctx = memory.get_context()
        assert "Sin resumen" in ctx["strategic_summary"]

    def test_includes_current_position(self, memory, state):
        state.update_position(1.5, -2.0, 0.8)
        ctx = memory.get_context()
        assert ctx["current_position"]["x"] == pytest.approx(1.5)

    def test_recent_events_list(self, memory, state):
        state.log_event("drone", "takeoff")
        ctx = memory.get_context()
        assert isinstance(ctx["recent_events"], list)
        assert len(ctx["recent_events"]) >= 1

    def test_mission_info_contains_name(self, memory):
        ctx = memory.get_context()
        assert ctx["mission_info"]["name"] == "test_hybrid"


class TestGetContextText:
    def test_contains_position_section(self, memory):
        text = memory.get_context_text()
        assert "POSICIÓN ACTUAL" in text

    def test_contains_summary_section(self, memory):
        text = memory.get_context_text()
        assert "RESUMEN ESTRATÉGICO" in text

    def test_contains_events_section(self, memory):
        text = memory.get_context_text()
        assert "EVENTOS RECIENTES" in text

    def test_contains_mission_info_section(self, memory):
        text = memory.get_context_text()
        assert "INFO MISIÓN" in text

    def test_returns_string(self, memory):
        assert isinstance(memory.get_context_text(), str)


class TestShouldUpdateSummary:
    def test_false_when_few_new_events(self, memory, state):
        state.log_event("drone", "action")
        assert memory.should_update_summary(every_n_events=10) is False

    def test_true_when_enough_new_events(self, memory, state):
        for _ in range(5):
            state.log_event("drone", "action")
        assert memory.should_update_summary(every_n_events=3) is True

    def test_boundary_exactly_n_events(self, memory, state):
        for _ in range(5):
            state.log_event("drone", "action")
        assert memory.should_update_summary(every_n_events=5) is True

    def test_resets_after_manual_update(self, memory, state):
        for _ in range(10):
            state.log_event("drone", "action")
        memory.update_summary_manual("Refreshed.")
        assert memory.should_update_summary(every_n_events=10) is False


class TestRepr:
    def test_repr_is_string(self, memory):
        assert isinstance(repr(memory), str)

    def test_repr_contains_summary_len(self, memory):
        memory.update_summary_manual("hello")
        assert "summary_len=5" in repr(memory)
