import logging
import math
import time
import traceback

from mission_config import (
    WEBOTS_IP, WEBOTS_PORT,
    MAX_FORWARD, MAX_YAW, MAX_ASCEND,
    SPEED_DIVISOR, CONTROL_PERIOD_S,
    SYNC_LLM_MODE,
    USE_LLM_GUIDANCE,
    POSITION_ENABLED, POSITION_PACKET_FLOATS,
    SUMMARY_UPDATE_INTERVAL, MAX_RECENT_EVENTS,
    RESET_STRATEGIC_SUMMARY_ON_START, ENABLE_SUMMARY_UPDATES,
    DECISION_MAX_LATENCY_S, CONTROL_RATE_HZ,
    OBSTACLE_AVOIDANCE_ENABLED, OBSTACLE_CENTER_THRESHOLD,
    OBSTACLE_ALT_CRUISE_Z, OBSTACLE_ALT_MAX_Z,
    OBSTACLE_STRONG_SCORE,
    TARGET_X, TARGET_Y, TARGET_X_SLOWDOWN_RADIUS,
    DIST_SENSOR_ENABLED, DOWN_CAM_ENABLED,
)
from llm_config import ( LMSTUDIO_OPENAI_BASE, get_decision_llm, get_summary_llm, probe_openai_compatible_server)
from mission_state import MissionState
from hybrid_memory import HybridMemory
from vision_pipeline import encode_both_profiles, encode_single
from image_annotator import annotate_frame, annotate_down_frame
from navigator import (parse_movement, apply_control_guardrails, compute_gps_guided_control, estimate_obstacle_avoidance, get_committed_obstacle_avoidance)
from webots_comm import connect_to_webots, receive_frame
from llm_invoker import (invoke_direct_llm, _llm_worker, _PROMPT_LOG_DIR)
from prompt_loader import load_prompt, parse_prompt_arg
from telemetry import init_telemetry, get_tracer, get_meter
from advanced_logger import MissionLogger, get_logger


def create_agent_system(system_prompt: str):
    logger = get_logger("init")
    logger.info("=" * 60)
    logger.info("  INICIALIZANDO SISTEMA VLM CLIENT")
    logger.info("=" * 60)

    logger.info("[1/3] Configurando modelos LLM…")
    decision_llm = get_decision_llm()
    summary_llm = get_summary_llm()
    _dm = getattr(decision_llm, "model_name", None) or getattr(decision_llm, "model", "?")
    _sm = getattr(summary_llm, "model_name", None) or getattr(summary_llm, "model", "?")
    logger.info("      Modelo decisión: %s", _dm)
    logger.info("      Modelo resumen:  %s", _sm)
    logger.info("      API: %s", LMSTUDIO_OPENAI_BASE)
    probe_ok, probe_detail = probe_openai_compatible_server()
    if probe_ok:
        logger.info("      Probe OK → %s", probe_detail)

    else:
        logger.error("      Probe fallido: %s", probe_detail)

    logger.info("[2/3] Creando estado de misión…")
    mission_state = MissionState("drone_landing_mission_x27")
    mission_state.set_metadata("mission_type", "x27_landing")
    mission_state.set_metadata("webots_port", WEBOTS_PORT)
    mission_state.set_metadata("position_packet_floats", POSITION_PACKET_FLOATS)

    logger.info("[3/3] Configurando memoria híbrida…")
    hybrid_memory = HybridMemory(mission_state=mission_state, recent_events_count=MAX_RECENT_EVENTS)
    if RESET_STRATEGIC_SUMMARY_ON_START:
        hybrid_memory.update_summary_manual("")
        logger.info("      Resumen estratégico previo reiniciado.")

    logger.info("      Buffer eventos: %d", MAX_RECENT_EVENTS)

    logger.info("Sistema inicializado.")
    logger.info("=" * 60)
    return mission_state, hybrid_memory, decision_llm


def main():
    system_prompt = load_prompt(parse_prompt_arg())

    init_telemetry("vlm_client")
    tracer = get_tracer("main_loop")
    meter = get_meter("drone_metrics")

    frame_counter      = meter.create_counter("drone.frames_processed", unit="1")
    decision_histogram = meter.create_histogram("drone.decision_latency", unit="s")
    pos_x_gauge        = meter.create_gauge("drone.position_x", unit="m")
    pos_y_gauge        = meter.create_gauge("drone.position_y", unit="m")
    pos_z_gauge        = meter.create_gauge("drone.position_z", unit="m")

    mission_state, hybrid_memory, decision_llm = create_agent_system(system_prompt)

    if USE_LLM_GUIDANCE and SYNC_LLM_MODE:
        mission_mode = "direct_llm_sync"

    elif USE_LLM_GUIDANCE:
        mission_mode = "direct_llm_async"

    else:
        mission_mode = "gps_autopilot"

    data = {"mode": mission_mode, "sync_llm": SYNC_LLM_MODE, "webots_ip": WEBOTS_IP, "webots_port": WEBOTS_PORT}
    mission_state.log_event("system", "mission_started", data)

    use_async_worker = USE_LLM_GUIDANCE and not SYNC_LLM_MODE
    if use_async_worker:
        _llm_worker.start(decision_llm=decision_llm, mission_state=mission_state, system_prompt=system_prompt)

    sock = connect_to_webots()
    mission_state.log_event("system", "webots_connected", {"ip": WEBOTS_IP, "port": WEBOTS_PORT})

    frame_id = 0
    next_control_tick = time.perf_counter()
    _consecutive_llm_failures = 0

    ml = MissionLogger()
    ml.log_system("Bucle principal iniciado", data={"mode": mission_mode, "sync": SYNC_LLM_MODE})

    if USE_LLM_GUIDANCE and SYNC_LLM_MODE:
        mode_label = "LLM Directo SÍNCRONO"

    elif USE_LLM_GUIDANCE:
        mode_label = "LLM Directo ASÍNCRONO"

    else:
        mode_label = "GPS Autopilot"

    print(f"\n[LOOP] Iniciando bucle principal…")
    print(f"[MODE] {mode_label}")
    print(f"[PROMPT] {parse_prompt_arg()}")
    if SPEED_DIVISOR > 1.0:
        print(f"[SLOW-MO] Velocidad ÷{SPEED_DIVISOR:.1f}")

    print(f"[PROMPT-LOG] {_PROMPT_LOG_DIR}\n")

    while True:
        try:
            now = time.perf_counter()
            if now < next_control_tick:
                time.sleep(next_control_tick - now)

            next_control_tick = max(next_control_tick + CONTROL_PERIOD_S, time.perf_counter())

            with tracer.start_as_current_span("frame_receive") as span:
                try:
                    frame_rgb, w, h, pos_x, pos_y, pos_z, dist_m, down_frame_rgb = receive_frame(sock, position_floats=POSITION_PACKET_FLOATS, position_enabled=POSITION_ENABLED)

                except ConnectionError as e:
                    ml.log_connection_event("reconnect", {"reason": str(e)})
                    sock.close()
                    sock = connect_to_webots()
                    continue

                if not (math.isfinite(pos_x) and math.isfinite(pos_y) and math.isfinite(pos_z)):
                    get_logger("gps").warning("GPS inválido (NaN/Inf) — usando última posición.")
                    prev = mission_state.position
                    pos_x, pos_y, pos_z = prev["x"], prev["y"], prev["z"]

                else:
                    mission_state.update_position(pos_x, pos_y, pos_z)

                if DIST_SENSOR_ENABLED and math.isfinite(dist_m):
                    mission_state.set_metadata("dist_m", round(dist_m, 3))

                frame_id += 1
                frame_counter.add(1)
                pos_x_gauge.set(pos_x)
                pos_y_gauge.set(pos_y)
                pos_z_gauge.set(pos_z)
                span.set_attribute("frame.id", frame_id)
                span.set_attribute("drone.pos_x", pos_x)
                span.set_attribute("drone.pos_y", pos_y)
                span.set_attribute("drone.pos_z", pos_z)
                span.set_attribute("drone.dist_m", round(dist_m, 3))
                ml.log_frame_received(frame_id, w, h, pos_x, pos_y, pos_z)

            if OBSTACLE_AVOIDANCE_ENABLED:
                _obs_blocked, _obs_avoid_rot, _obs_speed_scale, _obs_score = (get_committed_obstacle_avoidance(frame_rgb, dist_m if DIST_SENSOR_ENABLED else None))
            else:
                _obs_blocked, _obs_avoid_rot, _obs_speed_scale, _obs_score = (False, 0.0, 1.0, 0.0)
            _ann_phase = (
                "search" if TARGET_X is None
                else "approach" if pos_x >= TARGET_X - TARGET_X_SLOWDOWN_RADIUS
                else "cruise"
            )
            annotated_frame = annotate_frame(frame_rgb, pos_x, pos_y, pos_z, obstacle_score=_obs_score, obstacle_blocked=_obs_blocked, phase=_ann_phase, target_x=TARGET_X, target_y=TARGET_Y, dist_m=dist_m if DIST_SENSOR_ENABLED else None)

            down_b64: str | None = None
            if DOWN_CAM_ENABLED:
                ann_down = annotate_down_frame(down_frame_rgb, pos_x, pos_y, pos_z)
                down_b64 = encode_single(ann_down)

            with tracer.start_as_current_span("agent_invoke") as span:
                t0 = time.time()

                if USE_LLM_GUIDANCE and SYNC_LLM_MODE:
                    img_b64, img_b64_fb = encode_both_profiles(annotated_frame)
                    try:
                        movement, rotation, answer = invoke_direct_llm(decision_llm, mission_state, system_prompt, img_b64, frame_id, img_b64_fb=img_b64_fb, down_b64=down_b64)
                        latency = time.time() - t0
                        if not answer:
                            raise RuntimeError("LLM returned empty response")
                        
                        _consecutive_llm_failures = 0
                        guarded_movement, guarded_rotation, guarded_vz, guard_reason = (apply_control_guardrails(movement, rotation, pos_x, pos_y, latency, mission_state=mission_state))
                        phase = "llm_sync"

                    except Exception as sync_exc:
                        _consecutive_llm_failures += 1
                        latency = time.time() - t0
                        get_logger("sync_llm").warning("LLM sync falló (intento %d, %.1fs): %s", _consecutive_llm_failures, latency, sync_exc)
                        movement, rotation, guarded_vz, phase = compute_gps_guided_control(mission_state)
                        answer = f"sync_gps_fallback (fail #{_consecutive_llm_failures})"
                        guarded_movement, guarded_rotation, guard_reason = movement, rotation, phase

                elif USE_LLM_GUIDANCE and not SYNC_LLM_MODE:
                    img_b64, img_b64_fb = encode_both_profiles(annotated_frame)
                    _llm_worker.submit(img_b64, frame_id, img_b64_fb, down_b64=down_b64)
                    llm_result = _llm_worker.get_result()
                    result_age = frame_id - llm_result.frame_id
                    max_stale = int(DECISION_MAX_LATENCY_S * CONTROL_RATE_HZ) + 24
                    if llm_result.frame_id == 0 or result_age > max_stale:
                        movement, rotation, guarded_vz, phase = compute_gps_guided_control(mission_state)
                        reason = "cold_start" if llm_result.frame_id == 0 else f"stale_{result_age}f"
                        answer = f"gps_fallback ({reason})"
                        latency = time.time() - t0
                        guarded_movement, guarded_rotation, guard_reason = movement, rotation, phase

                    else:
                        movement, rotation, answer, latency = (llm_result.movement, llm_result.rotation, llm_result.answer, llm_result.latency_s)
                        guarded_movement, guarded_rotation, guarded_vz, guard_reason = (apply_control_guardrails(movement, rotation, pos_x, pos_y, latency, mission_state=mission_state))
                        phase = "llm_async"

                else:
                    movement, rotation, guarded_vz, phase = compute_gps_guided_control(mission_state)
                    answer = f"gps_controller phase={phase}"
                    latency = time.time() - t0
                    guarded_movement, guarded_rotation, guard_reason = movement, rotation, phase

                decision_histogram.record(latency, attributes={"decision.phase": phase})
                span.set_attribute("decision.phase", phase)
                span.set_attribute("decision.latency_s", round(latency, 3))

                data = {"frame_id": frame_id, "raw_movement": round(movement, 4), "raw_rotation": round(rotation, 4), "guarded_movement": round(guarded_movement, 4), "guarded_rotation": round(guarded_rotation, 4), "guarded_vz": round(guarded_vz, 4), "reason": guard_reason, "latency_s": round(latency, 4)}
                mission_state.log_event("controller", "guardrail_applied", data)

                movement = guarded_movement
                rotation = guarded_rotation
                vz = guarded_vz

                if OBSTACLE_AVOIDANCE_ENABLED:
                    blocked, avoid_rotation, speed_scale, obstacle_score = (_obs_blocked, _obs_avoid_rot, _obs_speed_scale, _obs_score)
                    if blocked and phase != "gps_landing":
                        rotation = avoid_rotation
                        movement = max(0.0, min(1.0, movement * speed_scale))
                        if POSITION_PACKET_FLOATS == 2:
                            vz = max(vz, MAX_ASCEND * 0.7)

                        elif pos_z < OBSTACLE_ALT_MAX_Z:
                            strength = min(1.0, max(0.0, (obstacle_score - OBSTACLE_CENTER_THRESHOLD) / max(0.05, OBSTACLE_STRONG_SCORE - OBSTACLE_CENTER_THRESHOLD)))
                            vz = max(vz, 0.08 + strength * (MAX_ASCEND - 0.08))
                        guard_reason += "+obstacle"
                        span.set_attribute("obstacle.blocked", True)
                        span.set_attribute("obstacle.score", round(obstacle_score, 4))

                    else:
                        _cruise_phases = ("gps_landing", "arrived", "gps_arrived", "landing_guardrail")
                        if phase not in _cruise_phases and POSITION_PACKET_FLOATS == 3 and pos_z > OBSTACLE_ALT_CRUISE_Z:
                            vz = min(vz, -0.06)

                        span.set_attribute("obstacle.blocked", False)

                ml.log_agent_decision(frame_id, movement, rotation, latency, f"{answer} | guard={guard_reason}")
                mission_state.update_last_command(movement, rotation)

            vx = (movement * MAX_FORWARD) / SPEED_DIVISOR
            vy = 0.0
            vz = vz / SPEED_DIVISOR
            yaw = (rotation * MAX_YAW) / SPEED_DIVISOR
            cmd = f"{vx} {vy} {vz} {yaw}\n"

            with tracer.start_as_current_span("command_send"):
                try:
                    sock.send(cmd.encode())
                    ml.log_command_sent(frame_id, vx, vy, vz, yaw)
                    data = {"frame_id": frame_id, "vx": vx, "vy": vy, "vz": vz, "yaw": yaw, "pos_x": round(pos_x, 4), "pos_y": round(pos_y, 4), "pos_z": round(pos_z, 4)}
                    mission_state.log_event("drone", "command_sent", data)

                except BrokenPipeError:
                    sock.close()
                    sock = connect_to_webots()

            if ENABLE_SUMMARY_UPDATES and hybrid_memory.should_update_summary(SUMMARY_UPDATE_INTERVAL):
                ml.log_system("Actualizando resumen estratégico…")
                try:
                    hybrid_memory.update_summary(get_summary_llm())
                    
                except Exception as e:
                    ml.log_error("hybrid_memory", f"Error actualizando resumen: {e}")

            if mission_state.total_events > 200:
                mission_state.clear_old_events(keep_last_n=100)

        except (ConnectionError, TimeoutError) as e:
            ml.log_connection_event("connection_error", {"error": str(e)})
            try:
                sock.close()

            except Exception:
                pass

            sock = connect_to_webots()

        except KeyboardInterrupt:
            ml.log_system("Interrumpido por usuario", level=logging.WARNING)
            if use_async_worker:
                _llm_worker.stop()
                
            hybrid_memory.save_summary()
            ml.log_session_summary()
            break

        except Exception as e:
            ml.log_error("main_loop", e, traceback.format_exc())
            time.sleep(1)


if __name__ == "__main__":
    main()
