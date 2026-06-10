#!/usr/bin/env python3

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any


_OBSTACLE_KW = frozenset([
    "barrel", "bidón", "wall", "pared", "obstacle", "obstáculo",
    "blocking", "bloqueando", "block", "crate", "barril",
    "collision", "colisión", "caja", "box", "object ahead",
])

_COLLISION_KW = frozenset([
    "crash", "collision", "colision", "colisión", "choque",
    "hit", "impact", "stuck", "atascado", "bloqueado",
    "broken pipe", "brokenpipe",
])

_ALT_MIN = 0.5
_ALT_MAX = 2.5
_STOP_MOV_TOL  = 0.05
_STOP_ROT_TOL  = 0.05
_STOP_MIN_FRAMES = 2 

def load_session(path: Path) -> list[dict[str, Any]]:
    entries = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))

                except json.JSONDecodeError:
                    pass

    return entries


def extract_model(entries: list[dict]) -> str:
    for e in entries:
        m = re.search(r"Modelo decisión:\s*(\S+)", e.get("message", ""))
        if m:
            return m.group(1)
        
    for e in entries:
        if e.get("category") == "llm" and "model" in e.get("data", {}):
            return e["data"]["model"]
        
    return "unknown"


def extract_latency(entries: list[dict]) -> dict[str, float]:
    lats = [
        e["data"]["latency_s"]
        for e in entries
        if e.get("category") == "llm" and "latency_s" in e.get("data", {})
    ]
    if not lats:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    
    return {"mean": round(mean(lats), 3), "min": round(min(lats), 3),
            "max": round(max(lats), 3), "n": len(lats)}


def extract_tokens(entries: list[dict]) -> dict[str, float]:
    compl, prompt = [], []
    for e in entries:
        if e.get("category") == "llm":
            d = e.get("data", {})
            if "completion_tokens" in d:
                compl.append(d["completion_tokens"])

            if "prompt_tokens" in d:
                prompt.append(d["prompt_tokens"])

    return {
        "completion_per_call": round(mean(compl), 1) if compl else 0.0,
        "prompt_per_call":     round(mean(prompt), 1) if prompt else 0.0,
        "total_calls":         len(compl),
    }


def extract_drift_max(entries: list[dict]) -> float:
    pattern = re.compile(r"Y-drift:\s*(LEFT|RIGHT)\s+([0-9]+(?:\.[0-9]+)?)m")
    drifts = []
    for e in entries:
        msg = e.get("message", "")
        if "prompt_debug" not in e.get("logger", ""):
            continue

        m = pattern.search(msg)
        if m:
            drifts.append(float(m.group(2)))

    return round(max(drifts), 3) if drifts else 0.0


def extract_frames(entries: list[dict]) -> int:
    for e in entries:
        if e.get("category") == "stats" and "frames_processed" in e.get("data", {}):
            return e["data"]["frames_processed"]
        
    return sum(1 for e in entries if e.get("category") == "frame")


def extract_errors(entries: list[dict]) -> int:
    count = sum(1 for e in entries if e.get("level") == "ERROR")
    critical_kw = ("brokenpipe", "broken pipe", "parseerror", "parse error", "context_window", "context window")
    for e in entries:
        msg = e.get("message", "").lower()
        if any(kw in msg for kw in critical_kw):
            count += 1

    return count


def extract_format_compliance(entries: list[dict]) -> float:
    llm_frames = [
        e for e in entries
        if e.get("category") == "agent"
        and e.get("data", {}).get("raw_response", "")
    ]
    if not llm_frames:
        return 0.0
    
    valid = sum(1 for e in llm_frames
        if "movement=" in e["data"]["raw_response"]
    )
    return round(100 * valid / len(llm_frames), 1)


def extract_obstacle_detection(entries: list[dict]) -> float:
    prompt_entries = [e for e in entries if "prompt_debug" in e.get("logger", "")]
    if not prompt_entries:
        return 0.0
    
    detected = 0
    resp_pattern = re.compile(r"resp=(.+?)(?:\s+\||\s*$)", re.IGNORECASE)
    for e in prompt_entries:
        m = resp_pattern.search(e.get("message", ""))
        if m:
            resp_lower = m.group(1).lower()
            if any(kw in resp_lower for kw in _OBSTACLE_KW):
                detected += 1

    return round(100 * detected / len(prompt_entries), 1)


def extract_semantic_hits(entries: list[dict], target_keywords: list[str], avoid_keywords: list[str] | None = None) -> float:
    avoid_keywords = avoid_keywords or []
    prompt_entries = [e for e in entries if "prompt_debug" in e.get("logger", "")]
    if not prompt_entries:
        return 0.0
    
    resp_pattern = re.compile(r"resp=(.+?)(?:\s+\||\s*$)", re.IGNORECASE)
    hits = 0
    for e in prompt_entries:
        m = resp_pattern.search(e.get("message", ""))
        if not m:
            continue

        resp_lower = m.group(1).lower()
        has_target = any(kw.lower() in resp_lower for kw in target_keywords)
        has_avoid  = any(kw.lower() in resp_lower for kw in avoid_keywords)
        if has_target and not has_avoid:
            hits += 1

    return round(100 * hits / len(prompt_entries), 1)


def is_session_completed(entries: list[dict]) -> bool:
    decision_entries = [
        e for e in entries
        if e.get("category") == "agent" and "movement" in e.get("data", {})
    ]
    if len(decision_entries) < _STOP_MIN_FRAMES:
        return False
    
    tail = decision_entries[-_STOP_MIN_FRAMES:]
    return all(
        abs(e["data"].get("movement", 1)) <= _STOP_MOV_TOL and
        abs(e["data"].get("rotation", 1)) <= _STOP_ROT_TOL
        for e in tail
    )


def check_no_collision(entries: list[dict]) -> bool:
    resp_pattern = re.compile(r"resp=(.+?)(?:\s+\||\s*$)", re.IGNORECASE)
    for e in entries:
        msg = e.get("message", "").lower()
        if any(kw in msg for kw in _COLLISION_KW):
            return False
        
        if "prompt_debug" in e.get("logger", ""):
            m = resp_pattern.search(e.get("message", ""))
            if m and any(kw in m.group(1).lower() for kw in _COLLISION_KW):
                return False
            
    return True


def extract_altitudes(entries: list[dict]) -> list[float]:
    altitudes = [
        e["data"]["pos_z"]
        for e in entries
        if e.get("category") == "frame" and "pos_z" in e.get("data", {})
    ]
    if not altitudes:
        gps_z = re.compile(r"GPS:\s*X=[^\s]+ Y=[^\s]+ Z=([0-9]+(?:\.[0-9]+)?)")
        for e in entries:
            if "prompt_debug" not in e.get("logger", ""):
                continue
            m = gps_z.search(e.get("message", ""))
            if m:
                altitudes.append(float(m.group(1)))
    return altitudes


def check_altitude_range(entries: list[dict]) -> dict[str, Any]:
    altitudes = extract_altitudes(entries)
    if not altitudes:
        return {"ok": None, "min": None, "max": None, "note": "sin datos"}
    z_min, z_max = min(altitudes), max(altitudes)
    ok = z_min >= _ALT_MIN and z_max <= _ALT_MAX
    return {"ok": ok, "min": round(z_min, 3), "max": round(z_max, 3), "note": ""}


def check_valid_commands(entries: list[dict]) -> dict[str, Any]:
    decision_entries = [
        e for e in entries
        if e.get("category") == "agent" and "movement" in e.get("data", {})
    ]
    if not decision_entries:
        return {"ok": None, "rate": None, "note": "sin datos"}
    valid = sum(
        1 for e in decision_entries
        if -1.0 <= e["data"]["movement"] <= 1.0
        and -1.0 <= e["data"].get("rotation", 0) <= 1.0
    )
    rate = round(100 * valid / len(decision_entries), 1)
    return {"ok": rate > 90.0, "rate": rate, "note": ""}


def analyse_session(path: Path) -> dict[str, Any]:
    entries = load_session(path)
    latency = extract_latency(entries)
    tokens  = extract_tokens(entries)
    alt     = check_altitude_range(entries)
    cmds    = check_valid_commands(entries)
    completed = is_session_completed(entries)

    return {
        "file":             path.name,
        "model":            extract_model(entries),
        "frames":           extract_frames(entries),
        "latency_mean_s":   latency["mean"],
        "latency_min_s":    latency["min"],
        "latency_max_s":    latency["max"],
        "compl_tok_call":   tokens["completion_per_call"],
        "prompt_tok_call":  tokens["prompt_per_call"],
        "drift_max_m":      extract_drift_max(entries),
        "obstacle_det_pct": extract_obstacle_detection(entries),
        "format_ok_pct":    extract_format_compliance(entries),
        "errors":           extract_errors(entries),
        "completed":        completed,
        "c2_no_collision":  check_no_collision(entries),
        "c3_alt_ok":        alt["ok"],
        "c3_alt_min":       alt["min"],
        "c3_alt_max":       alt["max"],
        "c4_valid_cmds_pct": cmds["rate"],
        "c4_ok":            cmds["ok"],
    }


def collect_paths(args_paths: list[str]) -> list[Path]:
    paths = []
    for arg in args_paths:
        p = Path(arg)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.jsonl")))
        elif p.is_file():
            paths.append(p)
        else:
            print(f"[WARN] No encontrado: {p}", file=sys.stderr)
    return paths


def print_table(results: list[dict]) -> None:
    cols = [
        ("Sesión",        "file",              30),
        ("Modelo",        "model",             22),
        ("Frames",        "frames",             6),
        ("Lat.med(s)",    "latency_mean_s",     9),
        ("Tok/call",      "compl_tok_call",     8),
        ("DriftMax(m)",   "drift_max_m",        10),
        ("ObstDet%",      "obstacle_det_pct",   8),
        ("FmtOK%",        "format_ok_pct",      6),
        ("Errs",          "errors",             4),
        ("Complet.",      "completed",          7),
        ("C3.AltOK",      "c3_alt_ok",          8),
        ("C4.Cmd%",       "c4_valid_cmds_pct",  7),
    ]
    header = "  ".join(f"{label:<{w}}" for label, _, w in cols)
    print(header)
    print("-" * len(header))
    for r in results:
        row = []
        for _, key, w in cols:
            val = r.get(key)
            if isinstance(val, bool):
                val = "SI" if val else "NO"
            elif val is None:
                val = "—"
            row.append(f"{str(val):<{w}}")
        print("  ".join(row))


def write_csv(results: list[dict], path: Path) -> None:
    if not results:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"CSV guardado: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extrae métricas de sesiones .jsonl (analisis_modelos.md §2.3)"
    )
    parser.add_argument(
        "paths", nargs="+",
        help="Fichero(s) .jsonl o directorio(s) con sesiones"
    )
    parser.add_argument(
        "--csv", metavar="FICHERO.csv", default=None,
        help="Exportar resultados a CSV"
    )
    args = parser.parse_args()

    paths = collect_paths(args.paths)
    if not paths:
        print("No se encontraron ficheros .jsonl.", file=sys.stderr)
        sys.exit(1)

    results = []
    for p in paths:
        try:
            results.append(analyse_session(p))
        except Exception as exc:
            print(f"[ERROR] {p.name}: {exc}", file=sys.stderr)

    print_table(results)

    if len(results) > 1:
        print()
        print(f"Total sesiones analizadas: {len(results)}")
        completed = sum(1 for r in results if r["completed"])
        print(f"Completadas:               {completed} / {len(results)}")
        c4_ok = [r for r in results if r.get("c4_ok")]
        print(f"C4 comandos válidos >90%:  {len(c4_ok)} / {len(results)}")
        drifts = [r["drift_max_m"] for r in results if r["drift_max_m"] > 0]
        if drifts:
            print(f"Deriva lateral media:      {round(mean(drifts), 3)}m")
            print(f"Deriva lateral máxima:     {round(max(drifts), 3)}m")

    if args.csv:
        write_csv(results, Path(args.csv))


if __name__ == "__main__":
    main()
