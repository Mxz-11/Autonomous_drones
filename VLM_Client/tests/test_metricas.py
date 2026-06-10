import json
import tempfile
from pathlib import Path

import pytest

from metricas import (
    load_session,
    extract_model,
    extract_latency,
    extract_tokens,
    extract_drift_max,
    extract_frames,
    extract_errors,
    extract_format_compliance,
    extract_obstacle_detection,
    extract_semantic_hits,
    is_session_completed,
    check_no_collision,
    check_altitude_range,
    check_valid_commands,
)


def _write_jsonl(entries: list[dict]) -> Path:
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    for e in entries:
        f.write(json.dumps(e) + "\n")
    f.close()
    return Path(f.name)


def _llm_entry(latency: float, model: str = "test-model",
               completion: int = 50, prompt: int = 200) -> dict:
    return {
        "category": "llm",
        "data": {
            "latency_s": latency,
            "model": model,
            "completion_tokens": completion,
            "prompt_tokens": prompt,
        },
    }


def _agent_entry(movement: float, rotation: float, raw: str = "") -> dict:
    return {
        "category": "agent",
        "data": {"movement": movement, "rotation": rotation, "raw_response": raw},
    }


class TestLoadSession:
    def test_reads_valid_jsonl(self):
        p = _write_jsonl([{"a": 1}, {"b": 2}])
        entries = load_session(p)
        assert len(entries) == 2

    def test_skips_blank_lines(self):
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        f.write('{"x": 1}\n\n{"y": 2}\n')
        f.close()
        entries = load_session(Path(f.name))
        assert len(entries) == 2

    def test_skips_malformed_json(self):
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        f.write('{"ok": 1}\nnot_json\n{"ok2": 2}\n')
        f.close()
        entries = load_session(Path(f.name))
        assert len(entries) == 2

    def test_empty_file(self):
        p = _write_jsonl([])
        assert load_session(p) == []


class TestExtractModel:
    def test_from_llm_category_field(self):
        entries = [_llm_entry(1.0, model="llama-3")]
        assert extract_model(entries) == "llama-3"

    def test_from_message_pattern(self):
        entries = [{"message": "Modelo decisión: gpt-4o"}]
        assert extract_model(entries) == "gpt-4o"

    def test_returns_unknown_when_no_info(self):
        assert extract_model([{"message": "nothing useful"}]) == "unknown"

    def test_empty_list_returns_unknown(self):
        assert extract_model([]) == "unknown"


class TestExtractLatency:
    def test_computes_correct_mean(self):
        entries = [_llm_entry(1.0), _llm_entry(3.0)]
        result = extract_latency(entries)
        assert result["mean"] == pytest.approx(2.0)
        assert result["n"] == 2

    def test_min_max(self):
        entries = [_llm_entry(0.5), _llm_entry(2.5), _llm_entry(1.0)]
        r = extract_latency(entries)
        assert r["min"] == pytest.approx(0.5)
        assert r["max"] == pytest.approx(2.5)

    def test_empty_entries_returns_zeros(self):
        r = extract_latency([])
        assert r == {"mean": 0.0, "min": 0.0, "max": 0.0, "n": 0}

    def test_non_llm_entries_ignored(self):
        entries = [{"category": "frame", "data": {"latency_s": 99.0}}]
        r = extract_latency(entries)
        assert r["n"] == 0


class TestExtractTokens:
    def test_mean_completion_tokens(self):
        entries = [_llm_entry(1.0, completion=100), _llm_entry(1.0, completion=200)]
        r = extract_tokens(entries)
        assert r["completion_per_call"] == pytest.approx(150.0)

    def test_mean_prompt_tokens(self):
        entries = [_llm_entry(1.0, prompt=300), _llm_entry(1.0, prompt=500)]
        r = extract_tokens(entries)
        assert r["prompt_per_call"] == pytest.approx(400.0)

    def test_total_calls(self):
        entries = [_llm_entry(1.0)] * 5
        r = extract_tokens(entries)
        assert r["total_calls"] == 5

    def test_empty_returns_zeros(self):
        r = extract_tokens([])
        assert r["completion_per_call"] == 0.0
        assert r["total_calls"] == 0


class TestExtractDriftMax:
    def test_finds_max_drift(self):
        entries = [
            {"logger": "prompt_debug", "message": "Y-drift: LEFT 1.23m"},
            {"logger": "prompt_debug", "message": "Y-drift: RIGHT 3.45m"},
        ]
        assert extract_drift_max(entries) == pytest.approx(3.45)

    def test_no_drift_returns_zero(self):
        entries = [{"logger": "other", "message": "Y-drift: LEFT 5.0m"}]
        assert extract_drift_max(entries) == 0.0

    def test_empty_entries(self):
        assert extract_drift_max([]) == 0.0


class TestExtractFrames:
    def test_reads_from_stats_entry(self):
        entries = [{"category": "stats", "data": {"frames_processed": 42}}]
        assert extract_frames(entries) == 42

    def test_counts_frame_category_as_fallback(self):
        entries = [
            {"category": "frame"},
            {"category": "frame"},
            {"category": "other"},
        ]
        assert extract_frames(entries) == 2

    def test_stats_entry_takes_priority(self):
        entries = [
            {"category": "stats", "data": {"frames_processed": 10}},
            {"category": "frame"},
        ]
        assert extract_frames(entries) == 10


class TestExtractErrors:
    def test_counts_error_level_entries(self):
        entries = [
            {"level": "ERROR", "message": "oops"},
            {"level": "INFO", "message": "ok"},
            {"level": "ERROR", "message": "bad"},
        ]
        assert extract_errors(entries) >= 2

    def test_detects_broken_pipe_keyword(self):
        entries = [{"level": "INFO", "message": "broken pipe occurred"}]
        assert extract_errors(entries) >= 1

    def test_empty_returns_zero(self):
        assert extract_errors([]) == 0


class TestExtractFormatCompliance:
    def test_all_valid(self):
        entries = [
            _agent_entry(0.5, 0.1, raw="movement=0.5 rotation=0.1"),
            _agent_entry(0.3, 0.0, raw="movement=0.3 rotation=0.0"),
        ]
        assert extract_format_compliance(entries) == pytest.approx(100.0)

    def test_half_valid(self):
        entries = [
            _agent_entry(0.5, 0.1, raw="movement=0.5"),
            _agent_entry(0.3, 0.0, raw="I don't know"),
        ]
        assert extract_format_compliance(entries) == pytest.approx(50.0)

    def test_no_agent_entries_returns_zero(self):
        entries = [{"category": "llm", "data": {}}]
        assert extract_format_compliance(entries) == 0.0


class TestExtractSemanticHits:
    def _make_prompt_entry(self, resp_text: str) -> dict:
        return {"logger": "prompt_debug", "message": f"resp={resp_text} | extra"}

    def test_hits_on_keyword_present(self):
        entries = [self._make_prompt_entry("I see a barrel ahead")]
        pct = extract_semantic_hits(entries, ["barrel"])
        assert pct == pytest.approx(100.0)

    def test_miss_when_keyword_absent(self):
        entries = [self._make_prompt_entry("clear path forward")]
        pct = extract_semantic_hits(entries, ["barrel"])
        assert pct == pytest.approx(0.0)

    def test_avoid_keyword_excludes_entry(self):
        entries = [self._make_prompt_entry("barrel but no real obstacle")]
        pct = extract_semantic_hits(entries, ["barrel"], avoid_keywords=["no real"])
        assert pct == pytest.approx(0.0)

    def test_empty_entries_returns_zero(self):
        assert extract_semantic_hits([], ["anything"]) == 0.0


class TestIsSessionCompleted:
    def test_completed_when_tail_is_zero_movement(self):
        entries = [
            _agent_entry(0.5, 0.1),
            _agent_entry(0.0, 0.0),
            _agent_entry(0.0, 0.0),
        ]
        assert is_session_completed(entries) is True

    def test_not_completed_when_still_moving(self):
        entries = [
            _agent_entry(0.5, 0.1),
            _agent_entry(0.5, 0.1),
        ]
        assert is_session_completed(entries) is False

    def test_not_completed_when_too_few_decisions(self):
        entries = [_agent_entry(0.0, 0.0)]
        assert is_session_completed(entries) is False

    def test_empty_entries_not_completed(self):
        assert is_session_completed([]) is False


class TestCheckNoCollision:
    def test_no_collision_keywords_returns_true(self):
        entries = [{"message": "moving forward fine"}]
        assert check_no_collision(entries) is True

    def test_crash_keyword_returns_false(self):
        entries = [{"message": "crash detected"}]
        assert check_no_collision(entries) is False

    def test_collision_in_prompt_debug_response(self):
        entries = [
            {"logger": "prompt_debug", "message": "resp=drone crashed into wall | extra"}
        ]
        assert check_no_collision(entries) is False

    def test_empty_entries_returns_true(self):
        assert check_no_collision([]) is True


class TestCheckAltitudeRange:
    def _frame_entry(self, z: float) -> dict:
        return {"category": "frame", "data": {"pos_z": z}}

    def test_all_within_range(self):
        entries = [self._frame_entry(0.8), self._frame_entry(1.5)]
        r = check_altitude_range(entries)
        assert r["ok"] is True

    def test_below_min_fails(self):
        entries = [self._frame_entry(0.1)]
        r = check_altitude_range(entries)
        assert r["ok"] is False

    def test_above_max_fails(self):
        entries = [self._frame_entry(3.0)]
        r = check_altitude_range(entries)
        assert r["ok"] is False

    def test_no_data_returns_none(self):
        r = check_altitude_range([{"category": "other"}])
        assert r["ok"] is None


class TestCheckValidCommands:
    def test_all_valid_commands(self):
        entries = [
            _agent_entry(0.5, 0.3),
            _agent_entry(0.8, -0.5),
        ]
        r = check_valid_commands(entries)
        assert r["rate"] == pytest.approx(100.0)
        assert r["ok"] is True

    def test_invalid_commands_detected(self):
        entries = [
            _agent_entry(5.0, 0.0),
            _agent_entry(0.5, 0.3),
        ]
        r = check_valid_commands(entries)
        assert r["rate"] < 100.0

    def test_no_data_returns_none(self):
        r = check_valid_commands([])
        assert r["ok"] is None
