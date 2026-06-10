import cv2
import numpy as np


def _blend_rect(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: tuple, alpha: float) -> None:
    h, w = img.shape[:2]
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    if x0 >= x1 or y0 >= y1:
        return
    
    roi = img[y0:y1, x0:x1]
    filled = np.full_like(roi, color, dtype=np.uint8)
    cv2.addWeighted(filled, alpha, roi, 1.0 - alpha, 0, roi)
    img[y0:y1, x0:x1] = roi


def _text(img: np.ndarray, text: str, x: int, y: int, scale: float = 0.38, color: tuple = (255, 255, 255), thickness: int = 1) -> None:
    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def annotate_frame(frame_rgb: np.ndarray, pos_x: float, pos_y: float, pos_z: float, obstacle_score: float = 0.0, obstacle_blocked: bool = False, phase: str = "cruise", target_x: float | None = None, target_y: float | None = None, dist_m: float | None = None) -> np.ndarray:
    img = frame_rgb.copy()
    h, w = img.shape[:2]

    BAR_H = 22
    BOT_H = 44
    FS = 0.38

    _blend_rect(img, 0, 0, w, BAR_H, (0, 0, 0), 0.67)

    if target_x is not None and target_x > 0:
        prog = min(1.0, max(0.0, pos_x / target_x))
        pb_w = int(w * 0.38)
        pb_h = 14
        pb_y0 = (BAR_H - pb_h) // 2
        pb_y1 = pb_y0 + pb_h
        cv2.rectangle(img, (4, pb_y0), (4 + pb_w, pb_y1), (180, 180, 180), 1)
        fill_w = max(0, int(pb_w * prog))

        if fill_w:
            bar_c = (80, 220, 80) if prog < 0.85 else (255, 200, 0)
            cv2.rectangle(img, (5, pb_y0 + 1), (5 + fill_w - 1, pb_y1 - 1), bar_c, -1)

        _text(img, f"X {pos_x:.1f}/{target_x:.0f}m ({prog*100:.0f}%)", 4 + pb_w + 6, pb_y1 - 2, FS)
    else:
        _text(img, f"X={pos_x:.1f}m", 6, BAR_H - 5, FS, color=(200, 200, 200))

    phase_label = phase.replace("_", " ").upper()
    phase_color = {
        "GPS LANDING":        (120, 120, 255),
        "LANDING GUARDRAIL":  (120, 120, 255),
        "APPROACH":           (50,  200, 255),
        "FINAL":              (50,  150, 255),
        "LLM GUARDED":        (255, 200, 100),
        "SEARCH":             (0,   220, 255),
    }.get(phase_label, (200, 255, 200))
    _text(img, phase_label, w // 2 - 42, BAR_H - 5, FS, color=phase_color)

    if dist_m is not None and dist_m >= 0:
        d_col = (80, 80, 255) if dist_m < 0.5 else ((0, 200, 255) if dist_m < 1.5 else (200, 255, 200))
        _text(img, f"D:{dist_m:.2f}m", w - 132, BAR_H - 5, FS, color=d_col)

    _text(img, f"Z={pos_z:.2f}m", w - 72, BAR_H - 5, FS)

    _blend_rect(img, 0, h - BOT_H, w, h, (0, 0, 0), 0.63)

    arrow_dir = 0
    drift_abs = 0.0
    if target_y is not None:
        drift = pos_y - target_y
        drift_abs = abs(drift)
        if drift_abs < 0.2:
            d_col   = (80, 220, 80)
            d_label = f"centered  Y={pos_y:.2f}m"

        elif drift_abs < 0.5:
            side      = "LEFT" if drift > 0 else "RIGHT"
            fix       = "turn RIGHT" if drift > 0 else "turn LEFT"
            d_col     = (0, 200, 255)
            d_label   = f"drift {side} {drift_abs:.2f}m  -> {fix}"
            arrow_dir = 1 if drift > 0 else -1

        else:
            side      = "LEFT" if drift > 0 else "RIGHT"
            fix       = "TURN RIGHT" if drift > 0 else "TURN LEFT"
            d_col     = (80, 80, 255)
            d_label   = f"DRIFT {side} {drift_abs:.2f}m  -> {fix}"
            arrow_dir = 1 if drift > 0 else -1

    else:
        d_col   = (180, 180, 180)
        d_label = f"Y={pos_y:.2f}m  [visual search]"

    _text(img, d_label, 6, h - BOT_H + 16, FS, color=d_col)

    if arrow_dir != 0 and drift_abs > 0.08:
        cx = w - 60
        cy = h - BOT_H + 12
        arrow_len = min(50, int(50 * drift_abs / 1.2))
        ax = cx + arrow_dir * arrow_len
        cv2.arrowedLine(img, (cx, cy), (ax, cy), d_col, 2, tipLength=0.35)

    fb_dir = 0
    if target_x is not None:
        dx = target_x - pos_x
        dx_abs = abs(dx)
        if dx_abs < 0.2:
            x_col   = (80, 220, 80)
            x_label = f"at target X  ({dx:+.2f}m)"

        elif dx > 0:
            x_col   = (200, 255, 200) if dx_abs < 0.5 else (0, 200, 255)
            x_label = f"target ahead {dx_abs:.2f}m  -> go FORWARD"
            fb_dir  = 1

        else:
            x_col   = (0, 200, 255) if dx_abs < 0.5 else (80, 80, 255)
            x_label = f"OVERSHOOT {dx_abs:.2f}m  -> go BACKWARD"
            fb_dir  = -1
    else:
        x_col   = (180, 180, 180)
        x_label = "no X target  [visual search]"

    _text(img, x_label, 6, h - BOT_H + 36, FS, color=x_col)

    if fb_dir != 0:
        cx = w - 60
        cy = h - BOT_H + 28
        if fb_dir > 0:
            cv2.arrowedLine(img, (cx, cy + 5), (cx, cy - 5), x_col, 2, tipLength=0.6)
        else:
            cv2.arrowedLine(img, (cx, cy - 5), (cx, cy + 5), x_col, 2, tipLength=0.6)

    if obstacle_blocked and obstacle_score > 0.18:
        severity = min(1.0, (obstacle_score - 0.14) / 0.22)
        alpha = 0.14 + 0.31 * severity
        rx0, rx1 = int(w * 0.35), int(w * 0.65)
        ry0, ry1 = int(h * 0.45), int(h * 0.92)
        _blend_rect(img, rx0, ry0, rx1, ry1, (80, 50, 255), alpha)
        cv2.rectangle(img, (rx0, ry0), (rx1, ry1), (80, 80, 255), 2)
        _text(img, f"OBSTACLE {obstacle_score:.2f}",
              rx0 + 4, ry0 + 20, 0.45, color=(100, 100, 255))

    return img


def annotate_down_frame(frame_rgb: np.ndarray, pos_x: float, pos_y: float, pos_z: float,) -> np.ndarray:
    img = frame_rgb.copy()
    h, w = img.shape[:2]
    BAR_H = 20

    _blend_rect(img, 0, 0, w, BAR_H, (0, 0, 0), 0.67)
    _text(img, f"DOWN | Z={pos_z:.2f}m  ({pos_x:.1f},{pos_y:.1f})", 4, BAR_H - 4, 0.38, color=(255, 210, 100))

    cx, cy = w // 2, h // 2
    arm = min(w, h) // 6
    cv2.line(img, (cx - arm, cy), (cx + arm, cy), (255, 200, 0), 1)
    cv2.line(img, (cx, cy - arm), (cx, cy + arm), (255, 200, 0), 1)
    cv2.circle(img, (cx, cy), 5, (255, 200, 0), 1)

    BOT_H = 16
    _blend_rect(img, 0, h - BOT_H, w, h, (0, 0, 0), 0.55)
    _text(img, "GROUND VIEW", 4, h - BOT_H + 12, 0.38, color=(255, 210, 100))

    return img
