import importlib
import os
from types import SimpleNamespace

import pytest


def _reload_config(env_overrides: dict) -> SimpleNamespace:
    old_env = {k: os.environ.get(k) for k in env_overrides}
    for k, v in env_overrides.items():
        os.environ[k] = v
    try:
        import mission_config
        importlib.reload(mission_config)
        snapshot = SimpleNamespace(
            **{k: v for k, v in vars(mission_config).items() if not k.startswith("_")}
        )
        return snapshot
    finally:
        for k, old_v in old_env.items():
            if old_v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old_v
        importlib.reload(mission_config)


class TestSpeedDivisor:
    def test_default_is_two(self):
        cfg = _reload_config({})
        assert cfg.SPEED_DIVISOR == 2.0

    def test_custom_value(self):
        cfg = _reload_config({"VLM_SPEED_DIVISOR": "4.0"})
        assert cfg.SPEED_DIVISOR == 4.0

    def test_invalid_falls_back_to_default(self):
        cfg = _reload_config({"VLM_SPEED_DIVISOR": "not_a_number"})
        assert cfg.SPEED_DIVISOR == 2.0

    def test_below_one_is_clamped_to_one(self):
        cfg = _reload_config({"VLM_SPEED_DIVISOR": "0.1"})
        assert cfg.SPEED_DIVISOR == 1.0


class TestTargetCoordinates:
    def test_defaults_are_none(self):
        cfg = _reload_config({})
        assert cfg.TARGET_X is None
        assert cfg.TARGET_Y is None

    def test_env_vars_are_ignored(self):
        cfg = _reload_config({"DRONE_TARGET_X": "5.0", "DRONE_TARGET_Y": "-3.5"})
        assert cfg.TARGET_X is None
        assert cfg.TARGET_Y is None
        assert cfg.STALE_TARGET_ENV == {"DRONE_TARGET_X": "5.0", "DRONE_TARGET_Y": "-3.5"}

    def test_cli_flag_parsed(self):
        import mission_config
        assert mission_config._parse_target_arg(["--target", "5,-3.5"]) == (5.0, -3.5)
        assert mission_config._parse_target_arg(["--target=9,-5"]) == (9.0, -5.0)

    def test_invalid_cli_falls_back_to_none(self):
        import mission_config
        assert mission_config._parse_target_arg(["--target", "bad"]) == (None, None)
        assert mission_config._parse_target_arg([]) == (None, None)


class TestSyncLlmMode:
    def test_default_is_sync(self):
        cfg = _reload_config({})
        assert cfg.SYNC_LLM_MODE is True

    def test_async_mode_enabled_by_flag(self):
        for val in ("1", "true", "yes", "on"):
            cfg = _reload_config({"VLM_ASYNC_MODE": val})
            assert cfg.SYNC_LLM_MODE is False, f"Expected async for VLM_ASYNC_MODE={val}"


class TestObstacleEnvVars:
    def test_avoidance_enabled_by_default(self):
        cfg = _reload_config({})
        assert cfg.OBSTACLE_AVOIDANCE_ENABLED is True

    def test_avoidance_disabled(self):
        cfg = _reload_config({"VLM_OBSTACLE_AVOID": "0"})
        assert cfg.OBSTACLE_AVOIDANCE_ENABLED is False

    def test_cruise_z_default(self):
        cfg = _reload_config({})
        assert cfg.OBSTACLE_ALT_CRUISE_Z == pytest.approx(0.60)

    def test_cruise_z_custom(self):
        cfg = _reload_config({"VLM_CRUISE_Z": "1.20"})
        assert cfg.OBSTACLE_ALT_CRUISE_Z == pytest.approx(1.20)


class TestControlRateConstants:
    def test_period_is_inverse_of_rate(self):
        import mission_config as cfg
        assert cfg.CONTROL_PERIOD_S == pytest.approx(1.0 / cfg.CONTROL_RATE_HZ)

    def test_max_forward_positive(self):
        import mission_config as cfg
        assert cfg.MAX_FORWARD > 0

    def test_max_descend_negative(self):
        import mission_config as cfg
        assert cfg.MAX_DESCEND < 0

    def test_max_ascend_positive(self):
        import mission_config as cfg
        assert cfg.MAX_ASCEND > 0
