#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluar_logs.py — Evaluación de sesiones de navegación autónoma de drones con VLMs.

Lee los logs JSONL generados por el cliente VLM (logs/evaluacion/mision1..4,
ignorando logs/prompts), calcula métricas de inferencia y de sesión, ejecuta
validaciones automáticas y genera tablas comparativas agrupadas por modelo y
misión, con exportación a CSV y JSON.

Uso:
    python3 evaluar_logs.py --logs-dir logs/evaluacion
    python3 evaluar_logs.py --logs-dir logs/evaluacion --export-csv out/ --export-json out/resultados.json

Métricas de inferencia (por llamada al modelo):
    - Tasa de cumplimiento de formato (%): respuestas de las que se pudo
      parsear un comando válido (se descuentan "respuesta sin comando" y
      fallos de sincronización con respuesta vacía).
    - Aciertos semánticos (%): respuestas parseadas cuyo texto es coherente
      con el objetivo de la misión (palabras clave por misión, ver
      MISSION_CONFIG). Es una métrica proxy basada en léxico.
    - Tasa de detección de obstáculos (%): de los frames con obstáculo
      presente (distancia frontal <= obstacle_present_m), fracción en la que
      el modelo reacciona (menciona el obstáculo, gira o frena). La distancia
      frontal se reconstruye geométricamente a partir del GPS y de la
      posición de los obstáculos estáticos de cada mundo de Webots
      (WORLD_OBSTACLES), porque la línea [PROMPT] del log se trunca a 80
      caracteres y solo conserva FrontDist en el frame 1. Si el log contiene
      FrontDist explícito, ese valor tiene prioridad.
    - Tokens por llamada (prompt / completion / total).
    - Latencia media de inferencia (s).

Métricas de sesión:
    - Tasa de sesiones completadas (%): la sesión cerró con SESSION SUMMARY /
      snapshot de estadísticas (no quedó truncada).
    - Tasa de sesiones con éxito (%): según el criterio por misión (ver
      MISSION_CONFIG["success"]).
    - Frames procesados por sesión.
    - Deriva lateral máxima (m): máximo |Y-drift| reportado en telemetría
      (fallback: máx |Y - target_y| en misiones con objetivo).
    - Errores de sistema por sesión: campo `errors` del snapshot + fallos de
      sincronización LLM + reconexiones de socket (desglosados).

Validaciones automáticas (por sesión):
    - Ausencia de colisiones (proxy): FrontDist < collision_m en algún frame,
      posición GPS dentro de la huella de un obstáculo del mundo, o caída de
      Z por debajo de z_min_vuelo lejos de la zona de aterrizaje.
    - Permanencia dentro de límites de altura: z_min_vuelo <= Z <= z_max en
      crucero (se exime el descenso final si la misión permite aterrizar).
    - Validez de comandos generados: movement/rotation numéricos y dentro de
      los rangos permitidos.
    - Alcance del objetivo final: criterio configurable por misión.

NOTA IMPORTANTE: los logs no contienen marcadores explícitos de éxito,
colisión ni fin de misión (todas las sesiones terminan con "Interrumpido por
usuario"), por lo que estas validaciones son heurísticas derivadas de la
telemetría. Los umbrales se ajustan en MISSION_CONFIG. Con la opción
--observados se imprime la comparación con los resultados anotados a mano.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics as st
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ───────────────────────────── Configuración ─────────────────────────────

# Alias cortos de modelo para las tablas.
MODEL_ALIASES = {
    "ggml-org/smolvlm2-2.2b-instruct": "Smol",
    "google/gemma-4-e4b": "Gemma",
    "qwen/qwen3-vl-8b": "Qwen",
}

# Configuración por misión. Todos los umbrales son ajustables.
#   success.mode:
#     "reach_and_land" → dist XY al objetivo <= tol  Y  Z final <= z_land
#     "reach"          → dist XY al objetivo <= tol
#     "detect_and_reach" → dist XY a `target` <= tol  Y  la última respuesta
#                          declara detección positiva (keywords sin negación)
#     "progress"       → X máxima alcanzada >= target_x - margin
MISSION_CONFIG = {
    "mision1": {
        "success": {"mode": "reach_and_land", "tol": 1.0, "z_land": 0.25},
        "semantic_keywords": ["helipad", "helipuerto", "land", "aterriz", "green", "verde", "target"],
        "obstacle_keywords": ["obstacle", "obstácul", "wall", "muro", "avoid", "esquiv", "turn", "gir"],
    },
    "mision2": {
        # El barril está en (4, 2) según el propio prompt de la misión.
        "success": {"mode": "detect_and_reach", "tol": 1.0, "target": (4.0, 2.0),
                    "detect_keywords": ["barrel", "barril"],
                    # Negaciones: incluye ecos del prompt ("Approach the red
                    # barrel…") que no son una detección real del modelo.
                    "negations": ["not visible", "no visible", "not found", "searching",
                                  "buscando", "approach", "aproxim", "you are",
                                  "sync_gps_fallback"]},
        "semantic_keywords": ["barrel", "barril", "red", "rojo", "cylinder", "cilindro"],
        "obstacle_keywords": ["obstacle", "obstácul", "wall", "muro", "avoid", "esquiv", "shelf", "estanter"],
    },
    "mision3": {
        # Slalom: se considera éxito recorrer el circuito (progreso en X).
        # El recorrido alternativo con objetivo (4, 2) exige además llegar
        # y aterrizar sobre el punto (override por objetivo).
        "success": {"mode": "progress", "margin": 1.5,
                    "overrides": [{"target": (4.0, 2.0),
                                   "mode": "reach_and_land", "tol": 1.0, "z_land": 0.25}]},
        "semantic_keywords": ["obstacle", "obstácul", "wall", "muro", "slalom", "gap", "hueco",
                              "pylon", "poste", "column", "columna", "clear"],
        "obstacle_keywords": ["obstacle", "obstácul", "wall", "muro", "avoid", "esquiv", "turn", "gir",
                              "left", "right", "izquierda", "derecha"],
    },
    "mision4": {
        "success": {"mode": "reach", "tol": 1.0},
        "semantic_keywords": ["arch", "arco", "opening", "apertura", "gate", "puerta", "red", "rojo"],
        "obstacle_keywords": ["obstacle", "obstácul", "arch", "arco", "avoid", "esquiv", "turn", "gir"],
    },
    # Valores por defecto para misiones no listadas.
    "_default": {
        "success": {"mode": "reach", "tol": 1.5},
        "semantic_keywords": ["target", "objetivo"],
        "obstacle_keywords": ["obstacle", "obstácul", "avoid", "esquiv"],
    },
}

# Umbrales globales de seguridad / validación.
SAFETY = {
    "z_max": 3.0,             # límite superior de altura (m)
    "z_min_vuelo": 0.15,      # por debajo de esto en crucero ≈ contacto con el suelo
    "landing_frac": 0.85,     # último 15% de frames: se permite descender (aterrizaje)
    "collision_m": 0.5,       # FrontDist por debajo de esto ≈ colisión inminente/contacto
    "obstacle_present_m": 2.5,# FrontDist por debajo de esto ⇒ hay obstáculo delante
    "mov_range": (-1.0, 1.0), # rango válido de movement
    "rot_range": (-1.57, 1.57),  # rango válido de rotation (rad/factor)
}

# ───────────── Geometría de los mundos (reconstrucción de FrontDist) ─────────────
# La línea [PROMPT] del jsonl trunca el user_text a 80 caracteres, por lo que
# FrontDist solo sobrevive en el frame 1. Para poder medir la detección de
# obstáculos y las colisiones se reconstruye la distancia frontal con la
# posición GPS de cada frame, el rumbo estimado por desplazamiento y los
# obstáculos estáticos de los mundos de Webots (proyecto_webots/worlds).
#
# Cada obstáculo es una caja alineada con los ejes:
#   {x, y: centro; hx, hy: semiejes; z_lo, z_hi: extensión vertical}
# Flags:
#   waypoint=True → puerta que la misión obliga a atravesar: cuenta para
#                   colisiones pero no para la tasa de detección de obstáculos.
#   target=True   → objetivo de la misión (acercarse es correcto): ídem.

def _box(x, y, hx, hy, z_lo, z_hi, **flags) -> dict:
    return {"x": x, "y": y, "hx": hx, "hy": hy, "z_lo": z_lo, "z_hi": z_hi, **flags}


WORLD_OBSTACLES = {
    # test1.wbt (versión actual): pasarela con puertas, contenedores y postes.
    "test1": [
        _box(2.5, 0.7, 0.035, 0.035, 0.0, 1.0, waypoint=True),    # WP1_LEFT
        _box(2.5, -0.7, 0.035, 0.035, 0.0, 1.0, waypoint=True),   # WP1_RIGHT
        _box(2.5, 0.0, 0.035, 0.735, 1.0, 1.07, waypoint=True),   # WP1_TOP
        _box(5.0, 1.2, 1.2, 0.5, 0.0, 2.4),                       # CONTAINER_1+2
        _box(5.0, 0.0, 1.2, 0.4, 0.0, 1.2),                       # CONTAINER_3
        _box(4.3, -1.6, 0.3, 0.3, 0.0, 0.6),                      # WBOX_1
        _box(5.7, -1.6, 0.3, 0.3, 0.0, 1.2),                      # WBOX_2+3
        _box(7.5, 1.45, 0.2, 0.45, 0.0, 0.8),                     # CARD_WALL_L
        _box(7.5, -1.45, 0.2, 0.45, 0.0, 0.8),                    # CARD_WALL_R
        _box(7.5, 0.7, 0.035, 0.035, 0.0, 1.0, waypoint=True),    # WP2_LEFT
        _box(7.5, -0.7, 0.035, 0.035, 0.0, 1.0, waypoint=True),   # WP2_RIGHT
        _box(7.5, 0.0, 0.035, 0.735, 1.0, 1.07, waypoint=True),   # WP2_TOP
        _box(8.8, 2.5, 0.05, 0.05, 0.0, 3.0),                     # POST_1
        _box(9.8, -2.5, 0.05, 0.05, 0.0, 3.0),                    # POST_2
        _box(11.0, 0.7, 0.035, 0.035, 0.0, 1.0, waypoint=True),   # WP3_LEFT
        _box(11.0, -0.7, 0.035, 0.035, 0.0, 1.0, waypoint=True),  # WP3_RIGHT
        _box(11.0, 0.0, 0.035, 0.735, 1.0, 1.07, waypoint=True),  # WP3_TOP
    ],
    # test2.wbt: almacén con bidón objetivo, estanterías, pilares y muros.
    "test2": [
        _box(-2.05, 0.0, 0.075, 11.0, 0.0, 3.0),                  # WALL_BACK
        _box(6.0, 10.0, 8.0, 0.075, 0.0, 3.0),                    # WALL_LEFT
        _box(6.0, -10.0, 8.0, 0.075, 0.0, 3.0),                   # WALL_RIGHT
        _box(4.0, 2.0, 0.3, 0.3, 0.0, 0.86, target=True),         # BIDON_ROJO
        _box(4.0, -2.0, 0.35, 0.35, 0.0, 0.7),                    # CAJA_AZUL
        _box(8.0, 0.0, 0.6, 0.4, 0.0, 0.21),                      # PALLET_VERDE
        _box(5.0, 5.5, 3.05, 0.25, 0.0, 1.8),                     # SHELF_L
        _box(5.0, -5.5, 3.05, 0.25, 0.0, 1.8),                    # SHELF_R
        _box(10.0, 3.0, 0.3, 0.3, 0.0, 1.2),                      # IND_BOX_1+2
        _box(10.0, -3.0, 0.3, 0.3, 0.0, 1.2),                     # IND_BOX_3+4
        _box(0.5, 4.5, 0.1, 0.1, 0.0, 3.0),                       # PILLAR_1
        _box(0.5, -4.5, 0.1, 0.1, 0.0, 3.0),                      # PILLAR_2
        _box(9.0, 4.5, 0.1, 0.1, 0.0, 3.0),                       # PILLAR_3
        _box(9.0, -4.5, 0.1, 0.1, 0.0, 3.0),                      # PILLAR_4
    ],
    # test3.wbt (versión actual): sala con muro divisorio y cubo azul objetivo.
    "test3": [
        _box(-3.05, 0.0, 0.075, 9.0, 0.0, 3.0),                   # WALL_BACK
        _box(5.0, 8.0, 9.0, 0.075, 0.0, 3.0),                     # WALL_LEFT
        _box(5.0, -8.0, 9.0, 0.075, 0.0, 3.0),                    # WALL_RIGHT
        _box(12.05, 0.0, 0.075, 9.0, 0.0, 3.0),                   # WALL_FRONT
        _box(5.0, -2.0, 0.075, 2.0, 0.0, 2.5),                    # MURO_LATERAL
        _box(9.0, -5.0, 0.45, 0.45, 0.0, 0.9, target=True),       # CAJA_AZUL
    ],
    # test1.wbt (versión committeada en git, usada en misión 4): arco rojo en
    # X=4 (coincide con la escena descrita en prompts/mision4.md) y resto del
    # circuito desplazado.
    "test1_arco": [
        _box(4.0, 0.7, 0.035, 0.035, 0.0, 1.0),                   # arco: poste izq.
        _box(4.0, -0.7, 0.035, 0.035, 0.0, 1.0),                  # arco: poste dcho.
        _box(4.0, 0.0, 0.035, 0.735, 1.0, 1.07),                  # arco: barra superior
        _box(8.0, 1.2, 1.2, 0.5, 0.0, 2.4),                       # CONTAINER_1+2
        _box(8.0, 0.0, 1.2, 0.4, 0.0, 1.2),                       # CONTAINER_3
        _box(7.0, -1.6, 0.3, 0.3, 0.0, 0.6),                      # WBOX_1
        _box(9.0, -1.6, 0.3, 0.3, 0.0, 1.2),                      # WBOX_2+3
        _box(12.0, 1.45, 0.2, 0.45, 0.0, 0.8),                    # CARD_WALL_L
        _box(12.0, -1.45, 0.2, 0.45, 0.0, 0.8),                   # CARD_WALL_R
        _box(12.0, 0.7, 0.035, 0.035, 0.0, 1.0, waypoint=True),   # WP2_LEFT
        _box(12.0, -0.7, 0.035, 0.035, 0.0, 1.0, waypoint=True),  # WP2_RIGHT
        _box(12.0, 0.0, 0.035, 0.735, 1.0, 1.07, waypoint=True),  # WP2_TOP
        _box(16.0, 2.5, 0.05, 0.05, 0.0, 3.0),                    # POST_1
        _box(19.0, -2.5, 0.05, 0.05, 0.0, 3.0),                   # POST_2
    ],
}

# Misión → mundo. La variante de mision3 con objetivo (4, 2) se ejecutó en un
# mundo que no está archivado (ni en el working tree ni en git), así que esas
# sesiones quedan sin geometría: su tasa de detección de obstáculos es None.
MISSION_WORLDS = {
    "mision1": "test1",
    "mision2": "test2",
    "mision4": "test1_arco",
}


def obstacles_for(mission: str, target_x, target_y) -> list[dict]:
    if mission == "mision3":
        if target_x is not None and abs(target_x - 9.0) < 1e-6:
            return WORLD_OBSTACLES["test3"]
        return []
    return WORLD_OBSTACLES.get(MISSION_WORLDS.get(mission, ""), [])


# Resultados observados manualmente (imagen del usuario), para comparación.
# {mision: {modelo: (ejecuciones, exitos)}}
OBSERVADOS = {
    "mision1": {"Smol": (4, 3), "Gemma": (4, 0), "Qwen": (4, 0)},
    "mision2": {"Smol": (5, 0), "Gemma": (5, 1), "Qwen": (5, 2)},
    "mision3": {"Smol": (3, 0), "Gemma": (4, 4), "Qwen": (4, 0)},
    "mision4": {"Smol": (3, 0), "Gemma": (3, 2), "Qwen": (3, 0)},
}

# ─────────────────────────── Expresiones regulares ───────────────────────────

RE_GPS = re.compile(r"GPS: X=(-?[\d.]+) Y=(-?[\d.]+) Z=(-?[\d.]+)")
RE_FRONTDIST = re.compile(r"FrontDist: ([\d.]+)m")
RE_YDRIFT = re.compile(r"Y-drift: (centered|(?:LEFT|RIGHT)?\s*-?[\d.]+m)")
RE_SIN_COMANDO = re.compile(r"respuesta sin comando", re.I)
RE_SYNC_FALLO = re.compile(r"LLM sync falló", re.I)
RE_FRAME_PROMPT = re.compile(r"\[PROMPT\] frame=(\d+)")


# ───────────────────────────── Estructuras ─────────────────────────────

@dataclass
class SessionMetrics:
    session_id: str
    file: str
    mission: str
    model: str
    model_alias: str
    # Inferencia
    llm_calls: int = 0
    format_fail_no_cmd: int = 0
    format_fail_empty: int = 0
    format_compliance_pct: Optional[float] = None
    semantic_hits: int = 0
    semantic_total: int = 0
    semantic_pct: Optional[float] = None
    obstacle_frames: int = 0
    obstacle_detected: int = 0
    obstacle_detection_pct: Optional[float] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tokens_per_call: Optional[float] = None
    avg_latency_s: Optional[float] = None
    # Sesión
    completed: bool = False
    success: bool = False
    frames_processed: int = 0
    commands_sent: int = 0
    duration_s: Optional[float] = None
    max_lateral_drift_m: Optional[float] = None
    errors_snapshot: int = 0
    sync_failures: int = 0
    reconnects: int = 0
    system_errors_total: int = 0
    # Validaciones
    target_x: Optional[float] = None
    target_y: Optional[float] = None
    final_x: Optional[float] = None
    final_y: Optional[float] = None
    final_z: Optional[float] = None
    max_x: Optional[float] = None
    min_z: Optional[float] = None
    max_z: Optional[float] = None
    dist_to_target_m: Optional[float] = None
    no_collision: bool = True
    within_altitude: bool = True
    commands_valid: bool = True
    invalid_commands: int = 0
    goal_reached: bool = False
    validations_passed: int = 0


def mission_cfg(mission: str) -> dict:
    return MISSION_CONFIG.get(mission, MISSION_CONFIG["_default"])


def contains_any(text: str, keywords) -> bool:
    t = text.lower()
    return any(k.lower() in t for k in keywords)


def _dist_to_box(px: float, py: float, ob: dict) -> float:
    dx = max(abs(px - ob["x"]) - ob["hx"], 0.0)
    dy = max(abs(py - ob["y"]) - ob["hy"], 0.0)
    return math.hypot(dx, dy)


# Cono frontal del estimador geométrico: el rumbo se deriva del desplazamiento
# GPS, así que se admite ±40° de incertidumbre frente al sensor real (un rayo).
FRONT_CONE_DEG = 40.0
HEADING_MIN_DISP_M = 0.08   # desplazamiento mínimo para considerar fiable el rumbo
HEADING_LOOKAHEAD = 5       # frames hacia delante para acumular desplazamiento


def estimate_front_obstacles(gps_track, frame_of_gps, obstacles) -> dict:
    """Distancia frontal reconstruida por frame: {frame_idx: (dist_m, obstáculo)}.

    Para cada posición GPS se estima el rumbo con el desplazamiento hacia los
    siguientes frames (los mundos arrancan mirando +X) y se busca el obstáculo
    más cercano cuya cara visible caiga dentro del cono frontal y cuyo rango
    vertical solape la altura del dron.
    """
    out: dict[int, tuple[float, dict]] = {}
    if not obstacles or not gps_track:
        return out
    cos_lim = math.cos(math.radians(FRONT_CONE_DEG))
    heading = (1.0, 0.0)
    for i, (x, y, z) in enumerate(gps_track):
        for j in range(i + 1, min(i + 1 + HEADING_LOOKAHEAD, len(gps_track))):
            dx, dy = gps_track[j][0] - x, gps_track[j][1] - y
            n = math.hypot(dx, dy)
            if n >= HEADING_MIN_DISP_M:
                heading = (dx / n, dy / n)
                break
        best = None
        for ob in obstacles:
            if ob.get("waypoint") or ob.get("target"):
                continue  # no cuentan como obstáculo (y no deben tapar a los demás)
            if not (ob["z_lo"] - 0.05 <= z <= ob["z_hi"] + 0.05):
                continue
            d = _dist_to_box(x, y, ob)
            if d > SAFETY["obstacle_present_m"]:
                continue
            if d > 1e-6:
                cx = min(max(x, ob["x"] - ob["hx"]), ob["x"] + ob["hx"])
                cy = min(max(y, ob["y"] - ob["hy"]), ob["y"] + ob["hy"])
                vx, vy = cx - x, cy - y
                n = math.hypot(vx, vy)
                if n and (vx * heading[0] + vy * heading[1]) / n < cos_lim:
                    continue
            if best is None or d < best[0]:
                best = (d, ob)
        if best is not None:
            f = frame_of_gps[i] if i < len(frame_of_gps) else i
            if f not in out or best[0] < out[f][0]:
                out[f] = best
    return out


# ───────────────────────────── Parseo de sesión ─────────────────────────────

def parse_session(path: Path, mission: str) -> SessionMetrics:
    cfg = mission_cfg(mission)
    sm = SessionMetrics(session_id=path.stem.split("_")[-1], file=path.name,
                        mission=mission, model="?", model_alias="?")

    latencies: list[float] = []
    gps_track: list[tuple[float, float, float]] = []   # (x, y, z)
    drifts: list[float] = []
    front_dists: list[tuple[int, float]] = []          # (índice frame, dist)
    responses: list[dict] = []                          # comandos parseados (agent)
    frame_of_gps: list[int] = []
    snapshot: dict = {}
    last_frame_idx = 0

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg = ev.get("message", "") or ""
            data = ev.get("data") or {}
            cat = ev.get("category")

            # Modelo y objetivo
            if "model" in data:
                sm.model = data["model"]
            if "target_x" in data:
                sm.target_x = float(data["target_x"])
                sm.target_y = float(data.get("target_y", 0.0))

            # Llamadas LLM (tokens / latencia)
            if cat == "llm":
                sm.llm_calls += 1
                sm.prompt_tokens += int(data.get("prompt_tokens", 0))
                sm.completion_tokens += int(data.get("completion_tokens", 0))
                sm.total_tokens += int(data.get("total_tokens", 0))
                if "latency_s" in data:
                    latencies.append(float(data["latency_s"]))

            # Fallos de formato
            if RE_SIN_COMANDO.search(msg):
                sm.format_fail_no_cmd += 1
            if RE_SYNC_FALLO.search(msg):
                sm.format_fail_empty += 1
                sm.sync_failures += 1

            # Reconexiones / sistema
            if cat == "system" and "reconnect" in msg:
                sm.reconnects += 1

            # Telemetría embebida en los mensajes de prompt
            m = RE_FRAME_PROMPT.search(msg)
            if m:
                last_frame_idx = int(m.group(1))
            g = RE_GPS.search(msg)
            if g:
                x, y, z = (float(v) for v in g.groups())
                gps_track.append((x, y, z))
                frame_of_gps.append(last_frame_idx)
            fdm = RE_FRONTDIST.search(msg)
            if fdm:
                front_dists.append((last_frame_idx, float(fdm.group(1))))
            dm = RE_YDRIFT.search(msg)
            if dm:
                raw = dm.group(1)
                if raw == "centered":
                    drifts.append(0.0)
                else:
                    num = re.search(r"-?[\d.]+", raw)
                    if num:
                        drifts.append(abs(float(num.group(0))))

            # Comandos parseados (eventos agent con movement/rotation)
            if cat == "agent" and "movement" in data:
                responses.append({
                    "frame": int(data.get("frame_id", last_frame_idx)),
                    "movement": data.get("movement"),
                    "rotation": data.get("rotation"),
                    "raw": data.get("raw_response", "") or "",
                })

            # Snapshot final de estadísticas
            if cat == "stats" and "Session stats snapshot" in msg and data:
                snapshot = data

    # ── Sesión completada y contadores del snapshot ──
    sm.completed = bool(snapshot)
    sm.frames_processed = int(snapshot.get("frames_processed", len(gps_track)))
    sm.commands_sent = int(snapshot.get("commands_sent", len(responses)))
    sm.errors_snapshot = int(snapshot.get("errors", 0))
    sm.duration_s = snapshot.get("session_elapsed_s")
    sm.system_errors_total = sm.errors_snapshot + sm.sync_failures + sm.reconnects

    # ── Métricas de inferencia ──
    if sm.llm_calls:
        fails = min(sm.format_fail_no_cmd + sm.format_fail_empty, sm.llm_calls)
        sm.format_compliance_pct = 100.0 * (sm.llm_calls - fails) / sm.llm_calls
        sm.tokens_per_call = sm.total_tokens / sm.llm_calls
    if latencies:
        sm.avg_latency_s = st.fmean(latencies)
    elif snapshot.get("avg_llm_latency_s"):
        sm.avg_latency_s = float(snapshot["avg_llm_latency_s"])

    # Aciertos semánticos: respuestas parseadas coherentes con la misión.
    parsed_with_text = [r for r in responses if r["raw"].strip()]
    sm.semantic_total = len(parsed_with_text)
    sm.semantic_hits = sum(1 for r in parsed_with_text
                           if contains_any(r["raw"], cfg["semantic_keywords"]))
    if sm.semantic_total:
        sm.semantic_pct = 100.0 * sm.semantic_hits / sm.semantic_total

    # Detección de obstáculos: frames con distancia frontal <= umbral. La
    # distancia se reconstruye geométricamente (estimate_front_obstacles); el
    # FrontDist del log, cuando existe, tiene prioridad. Las puertas que la
    # misión obliga a atravesar y el objetivo no cuentan como obstáculo.
    obstacles = obstacles_for(mission, sm.target_x, sm.target_y)
    geo_front = estimate_front_obstacles(gps_track, frame_of_gps, obstacles)
    front_events = {f: (d, ob) for f, (d, ob) in geo_front.items()
                    if not (ob.get("waypoint") or ob.get("target"))}
    for fidx, dist in front_dists:
        if dist <= SAFETY["obstacle_present_m"] and (
                fidx not in front_events or dist < front_events[fidx][0]):
            front_events[fidx] = (dist, {})
    resp_by_frame = {r["frame"]: r for r in responses}
    for fidx in sorted(front_events):
        sm.obstacle_frames += 1
        r = resp_by_frame.get(fidx) or resp_by_frame.get(fidx + 1)
        if r is None:
            continue
        mentions = contains_any(r["raw"], cfg["obstacle_keywords"])
        reacts = (abs(r.get("rotation") or 0.0) >= 0.1) or ((r.get("movement") or 1.0) <= 0.2)
        if mentions or reacts:
            sm.obstacle_detected += 1
    if sm.obstacle_frames:
        sm.obstacle_detection_pct = 100.0 * sm.obstacle_detected / sm.obstacle_frames

    # ── Telemetría agregada ──
    if gps_track:
        sm.final_x, sm.final_y, sm.final_z = gps_track[-1]
        sm.max_x = max(p[0] for p in gps_track)
        sm.min_z = min(p[2] for p in gps_track)
        sm.max_z = max(p[2] for p in gps_track)
    if drifts:
        sm.max_lateral_drift_m = max(drifts)
    elif gps_track and sm.target_y is not None:
        sm.max_lateral_drift_m = max(abs(p[1] - sm.target_y) for p in gps_track)

    # ── Validaciones automáticas ──
    n = len(gps_track)
    landing_start = int(n * SAFETY["landing_frac"]) if n else 0

    # 1) Colisiones (proxy): FrontDist registrado, posición dentro de la
    #    huella de un obstáculo del mundo (con margen por el radio del dron,
    #    incluidas puertas y objetivo), o contacto con el suelo en crucero.
    collided = any(d < SAFETY["collision_m"] for _, d in front_dists)
    if not collided:
        for x, y, z in gps_track:
            if any(ob["z_lo"] <= z <= ob["z_hi"]
                   and abs(x - ob["x"]) <= ob["hx"] + 0.06
                   and abs(y - ob["y"]) <= ob["hy"] + 0.06
                   for ob in obstacles):
                collided = True
                break
    if not collided:
        for i, (x, y, z) in enumerate(gps_track):
            if z < SAFETY["z_min_vuelo"] and i < landing_start:
                collided = True  # contacto con el suelo fuera de la fase de aterrizaje
                break
    sm.no_collision = not collided

    # 2) Límites de altura
    alt_ok = True
    for i, (x, y, z) in enumerate(gps_track):
        if z > SAFETY["z_max"]:
            alt_ok = False
            break
        if z < SAFETY["z_min_vuelo"] and i < landing_start:
            alt_ok = False
            break
    sm.within_altitude = alt_ok

    # 3) Validez de comandos
    lo_m, hi_m = SAFETY["mov_range"]
    lo_r, hi_r = SAFETY["rot_range"]
    for r in responses:
        mv, rt = r.get("movement"), r.get("rotation")
        ok = (isinstance(mv, (int, float)) and isinstance(rt, (int, float))
              and math.isfinite(mv) and math.isfinite(rt)
              and lo_m <= mv <= hi_m and lo_r <= rt <= hi_r)
        if not ok:
            sm.invalid_commands += 1
    sm.commands_valid = sm.invalid_commands == 0

    # 4) Alcance del objetivo (criterio por misión, con overrides por objetivo)
    sc = cfg["success"]
    for ov in sc.get("overrides", []):
        tx, ty = ov["target"]
        if (sm.target_x is not None and abs(sm.target_x - tx) < 1e-6
                and abs((sm.target_y or 0.0) - ty) < 1e-6):
            sc = ov
            break
    if sm.target_x is not None and sm.final_x is not None:
        sm.dist_to_target_m = math.dist((sm.final_x, sm.final_y), (sm.target_x, sm.target_y))
    mode = sc["mode"]
    if mode == "reach_and_land":
        sm.goal_reached = (sm.dist_to_target_m is not None
                           and sm.dist_to_target_m <= sc["tol"]
                           and sm.final_z is not None and sm.final_z <= sc["z_land"])
    elif mode == "reach":
        sm.goal_reached = (sm.dist_to_target_m is not None
                           and sm.dist_to_target_m <= sc["tol"])
    elif mode == "progress":
        tx = sm.target_x if sm.target_x is not None else None
        sm.goal_reached = (tx is not None and sm.max_x is not None
                           and sm.max_x >= tx - sc["margin"])
    elif mode == "detect_and_reach":
        tx, ty = sc["target"]
        dist = (math.dist((sm.final_x, sm.final_y), (tx, ty))
                if sm.final_x is not None else None)
        sm.dist_to_target_m = dist
        last_text = responses[-1]["raw"].lower() if responses else ""
        positive = (contains_any(last_text, sc["detect_keywords"])
                    and not contains_any(last_text, sc["negations"]))
        sm.goal_reached = dist is not None and dist <= sc["tol"] and positive

    sm.success = sm.goal_reached
    sm.validations_passed = sum([sm.no_collision, sm.within_altitude,
                                 sm.commands_valid, sm.goal_reached])
    return sm


# ───────────────────────────── Agregación ─────────────────────────────

NUMERIC_FIELDS = [
    ("format_compliance_pct", "Cumpl. formato (%)"),
    ("semantic_pct", "Aciertos semánticos (%)"),
    ("obstacle_detection_pct", "Detección obstáculos (%)"),
    ("tokens_per_call", "Tokens/llamada"),
    ("avg_latency_s", "Latencia media (s)"),
    ("frames_processed", "Frames/sesión"),
    ("max_lateral_drift_m", "Deriva lateral máx (m)"),
    ("system_errors_total", "Errores sistema/sesión"),
]


def aggregate(sessions: list[SessionMetrics]) -> dict:
    """Agrupa por (misión, modelo) y calcula media/máx/mín de cada métrica."""
    groups: dict[tuple[str, str], list[SessionMetrics]] = {}
    for s in sessions:
        groups.setdefault((s.mission, s.model_alias), []).append(s)

    out = {}
    for (mission, model), rows in sorted(groups.items()):
        entry = {
            "mision": mission, "modelo": model,
            "ejecuciones": len(rows),
            "sesiones_completadas": sum(r.completed for r in rows),
            "tasa_completadas_pct": 100.0 * sum(r.completed for r in rows) / len(rows),
            "sesiones_exitosas": sum(r.success for r in rows),
            "tasa_exito_pct": 100.0 * sum(r.success for r in rows) / len(rows),
            "validaciones": {
                "sin_colisiones": sum(r.no_collision for r in rows),
                "dentro_limites_altura": sum(r.within_altitude for r in rows),
                "comandos_validos": sum(r.commands_valid for r in rows),
                "objetivo_alcanzado": sum(r.goal_reached for r in rows),
            },
            "metricas": {},
        }
        for fname, label in NUMERIC_FIELDS:
            vals = [getattr(r, fname) for r in rows if getattr(r, fname) is not None]
            entry["metricas"][fname] = {
                "label": label,
                "media": st.fmean(vals) if vals else None,
                "max": max(vals) if vals else None,
                "min": min(vals) if vals else None,
                "n": len(vals),
            }
        out[f"{mission}/{model}"] = entry
    return out


# ───────────────────────────── Presentación ─────────────────────────────

def fmt(v, nd=2):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def print_table(headers, rows, title=None):
    widths = [len(h) for h in headers]
    srows = [[fmt(c) for c in r] for r in rows]
    for r in srows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))
    line = "  ".join("─" * w for w in widths)
    if title:
        print(f"\n{title}")
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print(line)
    for r in srows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))


def print_report(sessions, agg, show_observed: bool):
    models = sorted({s.model_alias for s in sessions})
    missions = sorted({s.mission for s in sessions})

    # Tabla 1: resumen de sesiones por misión × modelo
    rows = []
    for mi in missions:
        for mo in models:
            e = agg.get(f"{mi}/{mo}")
            if not e:
                continue
            rows.append([mi, mo, e["ejecuciones"],
                         f'{e["sesiones_completadas"]}/{e["ejecuciones"]}',
                         f'{e["tasa_completadas_pct"]:.0f}%',
                         f'{e["sesiones_exitosas"]}/{e["ejecuciones"]}',
                         f'{e["tasa_exito_pct"]:.0f}%'])
    print_table(["Misión", "Modelo", "Ejec.", "Completadas", "%Compl.", "Éxitos", "%Éxito"],
                rows, "═ RESUMEN DE SESIONES (por misión y modelo) ═")

    # Tabla 2: métricas de inferencia y sesión (media [mín–máx])
    for fname, label in NUMERIC_FIELDS:
        rows = []
        for mi in missions:
            for mo in models:
                e = agg.get(f"{mi}/{mo}")
                if not e:
                    continue
                m = e["metricas"][fname]
                rows.append([mi, mo, fmt(m["media"]), fmt(m["min"]), fmt(m["max"])])
        print_table(["Misión", "Modelo", "Media", "Mín", "Máx"], rows, f"═ {label} ═")
        if fname == "obstacle_detection_pct" and any(r[2] == "—" for r in rows):
            print("  (—: sesiones sin geometría de mundo conocida, ver "
                  "MISSION_WORLDS/obstacles_for, o sin ningún frame con "
                  f"obstáculo a <= {SAFETY['obstacle_present_m']} m)")

    # Tabla 3: validaciones automáticas
    rows = []
    for mi in missions:
        for mo in models:
            e = agg.get(f"{mi}/{mo}")
            if not e:
                continue
            v, n = e["validaciones"], e["ejecuciones"]
            rows.append([mi, mo,
                         f'{v["sin_colisiones"]}/{n}',
                         f'{v["dentro_limites_altura"]}/{n}',
                         f'{v["comandos_validos"]}/{n}',
                         f'{v["objetivo_alcanzado"]}/{n}'])
    print_table(["Misión", "Modelo", "Sin colisión", "Altura OK", "Cmds válidos", "Objetivo"],
                rows, "═ VALIDACIONES AUTOMÁTICAS (sesiones que pasan / total) ═")

    # Tabla 4 (opcional): comparación con resultados observados a mano
    if show_observed:
        rows = []
        for mi in missions:
            for mo in models:
                e = agg.get(f"{mi}/{mo}")
                obs = OBSERVADOS.get(mi, {}).get(mo)
                if not e or not obs:
                    continue
                auto = e["sesiones_exitosas"]
                rows.append([mi, mo, f"{obs[1]}/{obs[0]}",
                             f'{auto}/{e["ejecuciones"]}',
                             "✓" if auto == obs[1] and e["ejecuciones"] == obs[0] else "✗"])
        print_table(["Misión", "Modelo", "Éxitos observados", "Éxitos automáticos", "Coincide"],
                    rows, "═ CONTRASTE: criterio automático vs. observación manual ═")
        print("\n  Si alguna fila no coincide, ajusta MISSION_CONFIG (umbral, modo de éxito)\n"
              "  o usa etiquetas manuales: el criterio automático es una heurística.")


# ───────────────────────────── Exportación ─────────────────────────────

def export_csv(sessions, agg, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    f1 = outdir / "sesiones.csv"
    fields = list(SessionMetrics.__dataclass_fields__.keys())
    with f1.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for s in sessions:
            w.writerow(asdict(s))

    f2 = outdir / "agregado_modelo_mision.csv"
    with f2.open("w", newline="", encoding="utf-8") as fh:
        cols = ["mision", "modelo", "ejecuciones", "tasa_completadas_pct", "tasa_exito_pct",
                "val_sin_colisiones", "val_altura", "val_comandos", "val_objetivo"]
        for fname, _ in NUMERIC_FIELDS:
            cols += [f"{fname}_media", f"{fname}_min", f"{fname}_max"]
        w = csv.writer(fh)
        w.writerow(cols)
        for _key, e in agg.items():
            row = [e["mision"], e["modelo"], e["ejecuciones"],
                   round(e["tasa_completadas_pct"], 2), round(e["tasa_exito_pct"], 2),
                   e["validaciones"]["sin_colisiones"], e["validaciones"]["dentro_limites_altura"],
                   e["validaciones"]["comandos_validos"], e["validaciones"]["objetivo_alcanzado"]]
            for fname, _ in NUMERIC_FIELDS:
                m = e["metricas"][fname]
                row += [None if m["media"] is None else round(m["media"], 4),
                        m["min"], m["max"]]
            w.writerow(row)
    return [f1, f2]


def export_json(sessions, agg, outfile: Path):
    outfile.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {"safety": SAFETY,
                   "missions": {k: v for k, v in MISSION_CONFIG.items() if k != "_default"}},
        "sesiones": [asdict(s) for s in sessions],
        "agregado": agg,
        "observados_manual": OBSERVADOS,
    }
    outfile.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                       encoding="utf-8")
    return outfile


# ───────────────────────────── Programa principal ─────────────────────────────

def discover_sessions(logs_dir: Path):
    """Recorre misión*/  bajo logs_dir, ignorando cualquier carpeta 'prompts'."""
    if not logs_dir.is_dir():
        sys.exit(f"ERROR: no existe el directorio {logs_dir}")
    files = []
    for sub in sorted(logs_dir.iterdir()):
        if not sub.is_dir():
            continue
        if sub.name.lower() == "prompts":
            continue
        if not sub.name.lower().startswith("mision"):
            continue
        for f in sorted(sub.glob("*.jsonl")):
            if "prompts" in f.parts:
                continue
            files.append((sub.name, f))
    if not files:
        sys.exit(f"ERROR: no se encontraron .jsonl en {logs_dir}/mision*/")
    return files


def main():
    ap = argparse.ArgumentParser(description="Evaluación de logs JSONL de navegación con VLMs")
    ap.add_argument("--logs-dir", default="logs/evaluacion",
                    help="Directorio raíz con mision1..4 (def: logs/evaluacion)")
    ap.add_argument("--export-csv", metavar="DIR", help="Exportar CSVs a este directorio")
    ap.add_argument("--export-json", metavar="FICHERO", help="Exportar JSON completo")
    ap.add_argument("--observados", action="store_true",
                    help="Mostrar contraste con los resultados observados a mano")
    ap.add_argument("--quiet", action="store_true", help="No imprimir tablas")
    args = ap.parse_args()

    files = discover_sessions(Path(args.logs_dir))
    sessions = []
    for mission, f in files:
        s = parse_session(f, mission)
        s.model_alias = MODEL_ALIASES.get(s.model, s.model)
        sessions.append(s)

    agg = aggregate(sessions)

    if not args.quiet:
        print(f"Analizadas {len(sessions)} sesiones en {len({s.mission for s in sessions})} misiones "
              f"({', '.join(sorted({s.model_alias for s in sessions}))}).")
        print_report(sessions, agg, args.observados)

    if args.export_csv:
        paths = export_csv(sessions, agg, Path(args.export_csv))
        print("\nCSV exportados:", ", ".join(str(p) for p in paths))
    if args.export_json:
        p = export_json(sessions, agg, Path(args.export_json))
        print("JSON exportado:", p)


if __name__ == "__main__":
    main()