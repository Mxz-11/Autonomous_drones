import math
from unittest.mock import patch

import numpy as np
import pytest

import navigator as nav


class TestParseMovement:
    def test_basic_format_equals(self):
        m, r = nav.parse_movement("movement=0.5 rotation=-0.3")
        assert m == pytest.approx(0.5)
        assert r == pytest.approx(-0.3)

    def test_colon_separator(self):
        m, r = nav.parse_movement("movement: 0.8 rotation: 0.2")
        assert m == pytest.approx(0.8)
        assert r == pytest.approx(0.2)

    def test_case_insensitive(self):
        m, r = nav.parse_movement("MOVEMENT=0.3 ROTATION=0.1")
        assert m == pytest.approx(0.3)
        assert r == pytest.approx(0.1)

    def test_movement_clamped_to_0_1(self):
        m, _ = nav.parse_movement("movement=5.0 rotation=0.0")
        assert m == pytest.approx(1.0)

    def test_movement_negative_clamped_to_zero(self):
        m, _ = nav.parse_movement("movement=-0.5 rotation=0.0")
        assert m == pytest.approx(0.0)

    def test_rotation_clamped_to_minus1_1(self):
        _, r = nav.parse_movement("movement=0.5 rotation=99.0")
        assert r == pytest.approx(1.0)

    def test_rotation_clamped_negative(self):
        _, r = nav.parse_movement("movement=0.5 rotation=-99.0")
        assert r == pytest.approx(-1.0)

    def test_missing_movement_defaults_to_zero(self):
        m, r = nav.parse_movement("rotation=0.5")
        assert m == pytest.approx(0.0)
        assert r == pytest.approx(0.5)

    def test_missing_rotation_defaults_to_zero(self):
        m, r = nav.parse_movement("movement=0.7")
        assert m == pytest.approx(0.7)
        assert r == pytest.approx(0.0)

    def test_empty_string_returns_zeros(self):
        m, r = nav.parse_movement("")
        assert m == 0.0
        assert r == 0.0

    def test_scientific_notation(self):
        m, r = nav.parse_movement("movement=1e-1 rotation=-2e-1")
        assert m == pytest.approx(0.1)
        assert r == pytest.approx(-0.2)


class TestNormalizeAngle:
    def test_zero_stays_zero(self):
        assert nav._normalize_angle(0.0) == pytest.approx(0.0)

    def test_pi_normalizes_to_pi_or_minus_pi(self):
        result = nav._normalize_angle(math.pi)
        assert abs(result) == pytest.approx(math.pi, abs=1e-9)

    def test_beyond_pi_wraps_negative(self):
        result = nav._normalize_angle(math.pi + 0.1)
        assert result < 0

    def test_beyond_minus_pi_wraps_positive(self):
        result = nav._normalize_angle(-math.pi - 0.1)
        assert result > 0

    def test_small_angle_preserved(self):
        assert nav._normalize_angle(0.5) == pytest.approx(0.5)


class TestComputeRuleBasedControl:
    def test_no_target_returns_forward_search(self):
        with patch.object(nav, "TARGET_X", None):
            m, r, vz, phase = nav.compute_rule_based_control(0.0, 0.0)
        assert m == pytest.approx(0.5)
        assert r == pytest.approx(0.0)
        assert phase == "fwd_search"

    def test_at_target_returns_landing(self):
        with patch.object(nav, "TARGET_X", 5.0), \
             patch.object(nav, "TARGET_Y", 0.0), \
             patch.object(nav, "TARGET_X_REACHED_TOL", 0.35), \
             patch.object(nav, "ARRIVAL_HOVER", False):
            m, r, vz, phase = nav.compute_rule_based_control(4.9, 0.0)
        assert phase == "landing"
        assert m == pytest.approx(0.0)

    def test_hover_on_arrival_when_hover_mode(self):
        with patch.object(nav, "TARGET_X", 5.0), \
             patch.object(nav, "TARGET_Y", 0.0), \
             patch.object(nav, "TARGET_X_REACHED_TOL", 0.35), \
             patch.object(nav, "ARRIVAL_HOVER", True):
            m, r, vz, phase = nav.compute_rule_based_control(4.9, 0.0)
        assert phase == "arrived"
        assert m == 0.0 and r == 0.0 and vz == 0.0

    def test_far_from_target_full_speed(self):
        with patch.object(nav, "TARGET_X", 20.0), \
             patch.object(nav, "TARGET_Y", 0.0), \
             patch.object(nav, "TARGET_X_SLOWDOWN_RADIUS", 3.0), \
             patch.object(nav, "TARGET_X_REACHED_TOL", 0.35):
            m, r, vz, phase = nav.compute_rule_based_control(0.0, 0.0)
        assert m == pytest.approx(0.8)
        assert phase == "cruise"

    def test_y_error_generates_rotation(self):
        with patch.object(nav, "TARGET_X", 20.0), \
             patch.object(nav, "TARGET_Y", 0.0), \
             patch.object(nav, "TARGET_X_SLOWDOWN_RADIUS", 3.0), \
             patch.object(nav, "TARGET_X_REACHED_TOL", 0.35):
            _, r, _, _ = nav.compute_rule_based_control(0.0, 5.0)
        assert r < 0

    def test_small_y_error_no_rotation(self):
        with patch.object(nav, "TARGET_X", 20.0), \
             patch.object(nav, "TARGET_Y", 0.0), \
             patch.object(nav, "TARGET_X_SLOWDOWN_RADIUS", 3.0), \
             patch.object(nav, "TARGET_X_REACHED_TOL", 0.35):
            _, r, _, _ = nav.compute_rule_based_control(0.0, 0.1)
        assert r == pytest.approx(0.0)


class TestApplyControlGuardrails:
    def test_latency_fallback_ignores_llm(self):
        with patch.object(nav, "TARGET_X", None), \
             patch.object(nav, "DECISION_MAX_LATENCY_S", 5.0):
            m, r, vz, reason = nav.apply_control_guardrails(
                0.9, 0.8, 0.0, 0.0, llm_latency=10.0
            )
        assert "latency_fallback" in reason

    def test_no_target_trusts_llm_completely(self):
        with patch.object(nav, "TARGET_X", None), \
             patch.object(nav, "DECISION_MAX_LATENCY_S", 60.0):
            m, r, vz, reason = nav.apply_control_guardrails(
                0.7, -0.4, 0.0, 0.0, llm_latency=0.5
            )
        assert m == pytest.approx(0.7)
        assert r == pytest.approx(-0.4)
        assert reason == "llm_search"

    def test_output_is_clamped(self):
        with patch.object(nav, "TARGET_X", None), \
             patch.object(nav, "DECISION_MAX_LATENCY_S", 60.0):
            m, r, vz, _ = nav.apply_control_guardrails(
                2.0, 5.0, 0.0, 0.0, llm_latency=0.1
            )
        assert 0.0 <= m <= 1.0
        assert -1.0 <= r <= 1.0


class TestEstimateObstacleAvoidance:
    @pytest.fixture
    def clear_frame(self):
        return np.full((120, 160, 3), 200, dtype=np.uint8)

    @pytest.fixture
    def dark_center_frame(self):
        frame = np.full((120, 160, 3), 200, dtype=np.uint8)
        frame[:, 60:100, :] = 5
        return frame

    def test_clear_frame_not_blocked(self, clear_frame):
        blocked, _, _, _ = nav.estimate_obstacle_avoidance(clear_frame)
        assert not blocked

    def test_clear_frame_speed_scale_one(self, clear_frame):
        _, _, speed, _ = nav.estimate_obstacle_avoidance(clear_frame)
        assert speed == pytest.approx(1.0)

    def test_sensor_clear_overrides_cv(self, dark_center_frame):
        with patch.object(nav, "OBSTACLE_DIST_CLEAR_M", 1.40):
            blocked, _, _, _ = nav.estimate_obstacle_avoidance(
                dark_center_frame, dist_m=2.0
            )
        assert not blocked

    def test_sensor_close_overrides_cv(self, clear_frame):
        with patch.object(nav, "OBSTACLE_DIST_CLOSE_M", 0.70):
            blocked, _, speed, _ = nav.estimate_obstacle_avoidance(
                clear_frame, dist_m=0.3
            )
        assert blocked
        assert speed < 1.0

    def test_avoid_rotation_nonzero_when_blocked(self, dark_center_frame):
        with patch.object(nav, "OBSTACLE_CENTER_THRESHOLD", 0.01):
            blocked, avoid_rot, _, _ = nav.estimate_obstacle_avoidance(dark_center_frame)
        if blocked:
            assert avoid_rot != 0.0

    def test_avoid_rotation_within_bounds(self, dark_center_frame):
        with patch.object(nav, "OBSTACLE_DIST_CLOSE_M", 0.70):
            _, avoid_rot, _, _ = nav.estimate_obstacle_avoidance(
                dark_center_frame, dist_m=0.3
            )
        assert -1.0 <= avoid_rot <= 1.0

    def test_empty_roi_does_not_crash(self):
        tiny = np.zeros((1, 1, 3), dtype=np.uint8)
        blocked, avoid_rot, speed, score = nav.estimate_obstacle_avoidance(tiny)
        assert isinstance(blocked, bool)


class TestCommittedObstacleAvoidance:
    def setup_method(self):
        nav._obs_commit_dir = 0.0
        nav._obs_commit_remaining = 0

    def test_no_obstacle_resets_commit(self):
        clear = np.full((120, 160, 3), 200, dtype=np.uint8)
        blocked, _, _, _ = nav.get_committed_obstacle_avoidance(clear)
        assert nav._obs_commit_remaining == 0

    def test_obstacle_sets_commit_frames(self):
        clear = np.full((120, 160, 3), 200, dtype=np.uint8)
        with patch("navigator.estimate_obstacle_avoidance",
                   return_value=(True, 0.5, 0.6, 0.4)):
            nav.get_committed_obstacle_avoidance(clear)
        assert nav._obs_commit_remaining == nav._OBS_COMMIT_FRAMES

    def test_direction_is_held_while_committed(self):
        clear = np.full((120, 160, 3), 200, dtype=np.uint8)
        nav._obs_commit_dir = 0.7
        nav._obs_commit_remaining = 3
        with patch("navigator.estimate_obstacle_avoidance",
                   return_value=(True, -0.7, 0.5, 0.4)):
            _, avoid_rot, _, _ = nav.get_committed_obstacle_avoidance(clear)
        assert avoid_rot == pytest.approx(0.7)
