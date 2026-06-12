from __future__ import annotations

import os
import sys

WEBOTS_IP = "127.0.0.1"
WEBOTS_PORT = 9002
SOCKET_TIMEOUT_S = 10.0
MAX_FORWARD = 0.25
MAX_YAW = 0.3
MAX_DESCEND = -0.18
MAX_ASCEND = 0.22

try:
    SPEED_DIVISOR = max(1.0, float(os.environ.get("VLM_SPEED_DIVISOR", "2.0")))

except ValueError:
    SPEED_DIVISOR = 2.0

CONTROL_RATE_HZ = 2.0
CONTROL_PERIOD_S = 1.0 / CONTROL_RATE_HZ
SYNC_LLM_MODE = os.environ.get("VLM_ASYNC_MODE", "0").strip().lower() not in ("1", "true", "yes", "on")

def _parse_xy_arg(flag: str, argv: list[str] | None = None) -> tuple[float | None, float | None]:
    args = argv if argv is not None else sys.argv[1:]
    val = None
    for i, arg in enumerate(args):
        if arg == flag and i + 1 < len(args):
            val = args[i + 1]
            break

        if arg.startswith(flag + "="):
            val = arg.split("=", 1)[1]
            break

    if not val:
        return None, None

    try:
        x_s, y_s = val.split(",")
        return float(x_s), float(y_s)

    except ValueError:
        return None, None


def _parse_target_arg(argv: list[str] | None = None) -> tuple[float | None, float | None]:
    return _parse_xy_arg("--target", argv)


TARGET_X, TARGET_Y = _parse_target_arg()
STALE_TARGET_ENV = {k: v for k, v in os.environ.items() if k.startswith("DRONE_TARGET_")}

TARGET_X_SLOWDOWN_RADIUS = (min(3.0, max(1.0, TARGET_X * 0.35)) if TARGET_X is not None else 3.0)
TARGET_X_REACHED_TOL = 0.35
WAYPOINT_CLEAR_X, WAYPOINT_CORRIDOR_Y = _parse_xy_arg("--waypoint")


def phase_target_y(pos_x, target_y, clear_x, corridor_y):
    if (clear_x is not None and corridor_y is not None
            and pos_x is not None and pos_x < clear_x):
        return corridor_y

    return target_y


def effective_target_y(pos_x: float | None) -> float | None:
    return phase_target_y(pos_x, TARGET_Y, WAYPOINT_CLEAR_X, WAYPOINT_CORRIDOR_Y)
ROT_CORRECTION_THRESHOLD_Y = 0.8
HIGH_Y_ERROR_THRESHOLD = 2.0
DECISION_MAX_LATENCY_S = 60.0
USE_LLM_GUIDANCE = True
ARRIVAL_RADIUS = 0.6

HEADING_MIN_STEP = 0.02
HEADING_KP = 0.45
HEADING_SMOOTH_ALPHA = 0.25
HEADING_DEADBAND_RAD = 0.15

GUARDRAIL_OVERRIDE_ROT = 0.50

OBSTACLE_AVOIDANCE_ENABLED = os.environ.get("VLM_OBSTACLE_AVOID", "1").strip().lower() in ("1", "true", "yes", "on",)
OBSTACLE_CENTER_THRESHOLD = 0.18
OBSTACLE_EDGE_THRESHOLD = 0.18
OBSTACLE_DARK_THRESHOLD = 0.30
OBSTACLE_STRONG_SCORE = 0.33
try:
    OBSTACLE_ALT_CRUISE_Z = float(os.environ.get("VLM_CRUISE_Z", "0.60"))

except ValueError:
    OBSTACLE_ALT_CRUISE_Z = 0.60

OBSTACLE_ALT_MAX_Z = 1.80

ARRIVAL_HOVER = os.environ.get("VLM_ARRIVAL_HOVER", "0").strip().lower() in ("1", "true", "yes", "on",)

POSITION_ENABLED = True
POSITION_PACKET_FLOATS = 3

SUMMARY_UPDATE_INTERVAL = 20
MAX_RECENT_EVENTS = 6
RESET_STRATEGIC_SUMMARY_ON_START = True
ENABLE_SUMMARY_UPDATES = True

DEFAULT_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")
DEFAULT_DIRECT_PROMPT = os.path.join(DEFAULT_PROMPT_DIR, "mision1.md")
VLM_USER_CTX_CHARS = 0

DIST_SENSOR_ENABLED = os.environ.get("DIST_SENSOR", "1").strip() not in ("0", "false", "no", "off")
DOWN_CAM_ENABLED    = os.environ.get("DOWN_CAM",    "1").strip() not in ("0", "false", "no", "off")

OBSTACLE_DIST_CLOSE_M = 0.70
OBSTACLE_DIST_CLEAR_M = 1.40
