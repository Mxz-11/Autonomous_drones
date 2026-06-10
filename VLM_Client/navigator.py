import math
import re

import numpy as np

from mission_config import (
    TARGET_X, TARGET_Y,
    TARGET_X_SLOWDOWN_RADIUS, TARGET_X_REACHED_TOL,
    ROT_CORRECTION_THRESHOLD_Y, HIGH_Y_ERROR_THRESHOLD,
    DECISION_MAX_LATENCY_S, ARRIVAL_RADIUS,
    HEADING_MIN_STEP, HEADING_KP, HEADING_SMOOTH_ALPHA,
    HEADING_DEADBAND_RAD, GUARDRAIL_OVERRIDE_ROT,
    MAX_ASCEND, MAX_DESCEND,
    OBSTACLE_CENTER_THRESHOLD, OBSTACLE_EDGE_THRESHOLD,
    OBSTACLE_DARK_THRESHOLD, OBSTACLE_STRONG_SCORE,
    OBSTACLE_ALT_MAX_Z, OBSTACLE_ALT_CRUISE_Z,
    OBSTACLE_DIST_CLOSE_M, OBSTACLE_DIST_CLEAR_M,
    POSITION_PACKET_FLOATS,
    ARRIVAL_HOVER,
)


_FLOAT_RE = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"


def parse_movement(answer: str) -> tuple[float, float]:
    s = str(answer).lower().strip()
    m = re.search(rf"movement\s*[:=]\s*{_FLOAT_RE}", s)
    r = re.search(rf"rotation\s*[:=]\s*{_FLOAT_RE}", s)
    movement = float(m.group(1)) if m else 0.0
    rotation = float(r.group(1)) if r else 0.0
    return max(0.0, min(1.0, movement)), max(-1.0, min(1.0, rotation))


def compute_rule_based_control(pos_x: float, pos_y: float) -> tuple[float, float, float, str]:
    if TARGET_X is None:
        return 0.5, 0.0, 0.0, "fwd_search"

    dx = TARGET_X - pos_x

    if dx <= TARGET_X_REACHED_TOL:
        if ARRIVAL_HOVER:
            return 0.0, 0.0, 0.0, "arrived"
        
        landing_rotation = max(-0.3, min(0.3, -0.15 * (pos_y - TARGET_Y)))
        return 0.0, landing_rotation, MAX_DESCEND, "landing"

    if dx <= TARGET_X_SLOWDOWN_RADIUS:
        ratio = dx / TARGET_X_SLOWDOWN_RADIUS
        movement = max(0.2, min(0.55, 0.55 * ratio))

    else:
        movement = 0.8

    y_error = pos_y - TARGET_Y
    rotation = 0.0 if abs(y_error) <= 0.35 else max(-0.65, min(0.65, -0.12 * y_error))

    return movement, rotation, 0.0, "cruise"


def apply_control_guardrails(llm_movement: float, llm_rotation: float, pos_x: float, pos_y: float, llm_latency: float, mission_state=None) -> tuple[float, float, float, str]:
    if mission_state is not None and TARGET_X is not None:
        rb_movement, rb_rotation, rb_vz, phase = compute_gps_guided_control(mission_state)

    else:
        rb_movement, rb_rotation, rb_vz, phase = compute_rule_based_control(pos_x, pos_y)

    if llm_latency > DECISION_MAX_LATENCY_S:
        return rb_movement, rb_rotation, rb_vz, f"{phase}_latency_fallback"

    if TARGET_X is None:
        return (max(0.0, min(1.0, llm_movement)), max(-1.0, min(1.0, llm_rotation)), 0.0, "llm_search")

    if phase in ("landing", "gps_landing", "arrived", "gps_arrived"):
        return rb_movement, rb_rotation, rb_vz, f"{phase}_guardrail"

    movement = llm_movement
    rotation = llm_rotation

    if rotation * rb_rotation < 0 and abs(rb_rotation) >= GUARDRAIL_OVERRIDE_ROT:
        rotation = rb_rotation

    if rb_movement < movement:
        movement = rb_movement

    elif TARGET_X - pos_x > TARGET_X_REACHED_TOL:
        movement = max(movement, rb_movement * 0.6)

    return (max(0.0, min(1.0, movement)), max(-1.0, min(1.0, rotation)), 0.0, "llm_guarded",)

_last_rotation: float = 0.0


def _normalize_angle(rad: float) -> float:
    return math.atan2(math.sin(rad), math.cos(rad))


def compute_gps_guided_control(mission_state) -> tuple[float, float, float, str]:
    global _last_rotation
    pos = mission_state.position
    x, y, z = pos["x"], pos["y"], pos.get("z", 0.0)

    if TARGET_X is None:
        _last_rotation = 0.0
        return 0.5, 0.0, 0.0, "fwd_search"

    dx = TARGET_X - x
    dy = TARGET_Y - y
    dist = math.hypot(dx, dy)

    if dist <= ARRIVAL_RADIUS:
        _last_rotation = 0.0
        if ARRIVAL_HOVER:
            return 0.0, 0.0, 0.0, "gps_arrived"
        
        return 0.0, 0.0, MAX_DESCEND, "gps_landing"

    history = mission_state.get_recent_positions(4)
    heading_known = False
    heading_now = 0.0
    if len(history) >= 2:
        for i in range(len(history) - 1, 0, -1):
            hx = history[i]["x"] - history[i - 1]["x"]
            hy = history[i]["y"] - history[i - 1]["y"]
            if math.hypot(hx, hy) >= HEADING_MIN_STEP:
                heading_now = math.atan2(hy, hx)
                heading_known = True
                break

    desired_heading = math.atan2(dy, dx)

    if heading_known:
        err = _normalize_angle(desired_heading - heading_now)
        if abs(err) < HEADING_DEADBAND_RAD:
            err = 0.0

        raw_rotation = max(-0.9, min(0.9, HEADING_KP * err))
        phase = "gps_track_heading"
    else:
        y_sign = math.copysign(1.0, dy) if abs(dy) > 0.1 else 0.0
        lateral_factor = min(1.0, abs(dy) / 4.0)
        raw_rotation = y_sign * 0.55 * lateral_factor
        phase = "gps_bootstrap_heading"

    rotation = HEADING_SMOOTH_ALPHA * _last_rotation + (1.0 - HEADING_SMOOTH_ALPHA) * raw_rotation
    _last_rotation = rotation

    if dist > 12.0:
        movement = 0.90

    elif dist > 6.0:
        movement = 0.70

    elif dist > 2.5:
        movement = 0.50

    else:
        movement = 0.28

    if heading_known:
        abs_err = abs(_normalize_angle(desired_heading - heading_now))
        if abs_err > 1.2:
            movement = min(movement, 0.18)

        elif abs_err > 0.6:
            movement = min(movement, 0.32)


    vz = 0.0
    if POSITION_PACKET_FLOATS == 3:
        cruise_z = getattr(compute_gps_guided_control, "_cruise_z", 0.55)
        if z < cruise_z - 0.15 and phase != "gps_landing":
            vz = min(MAX_ASCEND, 0.08 + (cruise_z - z) * 0.20)

        elif z > cruise_z + 0.25 and phase != "gps_landing":
            vz = max(-0.06, -(z - cruise_z) * 0.10)

    return movement, rotation, vz, phase


_obs_commit_dir: float = 0.0
_obs_commit_remaining: int = 0
_OBS_COMMIT_FRAMES = 6


def get_committed_obstacle_avoidance(frame_rgb: np.ndarray, dist_m: float | None = None) -> tuple[bool, float, float, float]:
    global _obs_commit_dir, _obs_commit_remaining
    blocked, avoid_rot, speed_scale, score = estimate_obstacle_avoidance(frame_rgb, dist_m)

    if blocked:
        if _obs_commit_remaining <= 0:
            _obs_commit_dir = avoid_rot
            _obs_commit_remaining = _OBS_COMMIT_FRAMES

        else:
            avoid_rot = _obs_commit_dir
            _obs_commit_remaining -= 1

    else:
        _obs_commit_remaining = 0

    return blocked, avoid_rot, speed_scale, score


def estimate_obstacle_avoidance(frame_rgb: np.ndarray, dist_m: float | None = None) -> tuple[bool, float, float, float]:
    h, w, _ = frame_rgb.shape
    gray = frame_rgb.astype(np.float32).mean(axis=2) / 255.0

    y0, y1 = int(h * 0.45), int(h * 0.92)
    x0, x1 = int(w * 0.12), int(w * 0.88)
    roi = gray[y0:y1, x0:x1]

    if roi.size == 0:
        sensor_blocked = (dist_m is not None and math.isfinite(dist_m) and dist_m < OBSTACLE_DIST_CLOSE_M)
        return sensor_blocked, 0.0, (0.3 if sensor_blocked else 1.0), 0.0

    gy, gx = np.gradient(roi)
    edge_mag = np.hypot(gx, gy)
    edge_map = (edge_mag > OBSTACLE_EDGE_THRESHOLD).astype(np.float32)
    dark_map = (roi < OBSTACLE_DARK_THRESHOLD).astype(np.float32)
    obs_map = 0.65 * edge_map + 0.35 * dark_map

    rw = roi.shape[1]
    c0, c1 = int(rw * 0.35), int(rw * 0.65)
    l0, l1 = 0, int(rw * 0.35)
    r0, r1 = int(rw * 0.65), rw

    center_score = float(obs_map[:, c0:c1].mean()) if c1 > c0 else 0.0
    left_score   = float(obs_map[:, l0:l1].mean()) if l1 > l0 else center_score
    right_score  = float(obs_map[:, r0:r1].mean()) if r1 > r0 else center_score

    sensor_valid = dist_m is not None and math.isfinite(dist_m) and dist_m > 0.0
    if sensor_valid:
        if dist_m >= OBSTACLE_DIST_CLEAR_M:
            return False, 0.0, 1.0, center_score
        
        sensor_blocked = dist_m < OBSTACLE_DIST_CLOSE_M
    else:
        sensor_blocked = False

    blocked = sensor_blocked or (center_score >= OBSTACLE_CENTER_THRESHOLD)
    if not blocked:
        return False, 0.0, 1.0, center_score

    if sensor_valid and dist_m < OBSTACLE_DIST_CLOSE_M:
        severity = min(1.0, max(0.0, 1.0 - dist_m / OBSTACLE_DIST_CLOSE_M))
        
    else:
        severity = min(1.0, max(0.0, (center_score - OBSTACLE_CENTER_THRESHOLD) / 0.35))

    turn_sign = -1.0 if left_score < right_score else 1.0
    avoid_rotation = turn_sign * min(0.95, 0.30 + 0.55 * severity)
    speed_scale = max(0.20, 1.0 - 0.70 * severity)

    return True, avoid_rotation, speed_scale, center_score
