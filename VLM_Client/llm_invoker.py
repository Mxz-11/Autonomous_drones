import json
import os
import queue
import re
import threading
import time
from dataclasses import dataclass

from langchain_core.messages import SystemMessage, HumanMessage

from mission_config import DECISION_MAX_LATENCY_S, TARGET_Y as _TARGET_Y
from vision_pipeline import jpeg_b64_to_data_url, is_context_window_error
from advanced_logger import MissionLogger, get_logger, extract_token_usage
from llm_config import LMSTUDIO_OPENAI_BASE, is_probably_network_llm_error
from navigator import parse_movement

_PROMPT_LOG_DIR = os.path.join(os.path.dirname(__file__), "logs", "prompts")
os.makedirs(_PROMPT_LOG_DIR, exist_ok=True)
_LLM_NET_ERR_LOG_INTERVAL_S = 20.0
_llm_net_err_last_ts: float = 0.0


def build_vlm_user_text(frame_id: int, pos: dict, mission_state=None) -> str:
    x, y, z = pos["x"], pos["y"], pos["z"]

    if _TARGET_Y is not None:
        dy = y - _TARGET_Y
        if dy > 0.1:
            drift = f"LEFT {dy:.2f}m"

        elif dy < -0.1:
            drift = f"RIGHT {abs(dy):.2f}m"

        else:
            drift = "centered"

    else:
        drift = None

    line = f"Frame #{frame_id} | GPS: X={x:.2f} Y={y:.2f} Z={z:.2f}"
    if drift is not None:
        line += f" | Y-drift: {drift}"

    if mission_state is not None:
        history = mission_state.get_recent_positions(4)
        if len(history) >= 2:
            track = " → ".join(f"({p['x']:.1f},{p['y']:.1f})" for p in history[-4:])
            line += f" | Track: {track}"

        last_rot = mission_state.get_last_rotation()
        if last_rot is not None:
            line += f" | PrevRot: {last_rot:+.2f}"

        dist_m = mission_state.metadata.get("dist_m")
        if dist_m is not None:
            line += f" | FrontDist: {dist_m:.2f}m"

    return line


def _log_llm_request(frame_id: int, mode: str, system_prompt: str, user_text: str, img_b64: str, response_text: str, latency_s: float) -> None:
    import base64 as _b64

    logger = get_logger("prompt_debug")
    ts = time.strftime("%H%M%S")

    img_path = os.path.join(_PROMPT_LOG_DIR, f"frame_{frame_id:05d}.jpg")
    try:
        with open(img_path, "wb") as f:
            f.write(_b64.b64decode(img_b64))

    except Exception as e:
        logger.warning("No se pudo guardar imagen frame %d: %s", frame_id, e)
        img_path = "<error>"

    prompt_data = {
        "frame_id": frame_id,
        "timestamp": ts,
        "mode": mode,
        "system_prompt": system_prompt,
        "user_text": user_text,
        "image_file": os.path.basename(img_path),
        "image_b64_length": len(img_b64),
        "response": response_text,
        "latency_s": round(latency_s, 3),
    }
    prompt_path = os.path.join(_PROMPT_LOG_DIR, f"frame_{frame_id:05d}_prompt.json")
    try:
        with open(prompt_path, "w") as f:
            json.dump(prompt_data, f, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.warning("No se pudo guardar prompt frame %d: %s", frame_id, e)

    logger.info(
        "[PROMPT] frame=%d mode=%s | sys=%d chars | user=%s | img=%d bytes b64 | resp=%s | %.1fs",
        frame_id, mode,
        len(system_prompt),
        user_text[:80],
        len(img_b64),
        response_text[:100].replace("\n", " "),
        latency_s,
    )


def _log_throttled_llm_network_error(logger_name: str, exc: BaseException) -> None:
    global _llm_net_err_last_ts
    log = get_logger(logger_name)
    now = time.time()
    if now - _llm_net_err_last_ts >= _LLM_NET_ERR_LOG_INTERVAL_S:
        _llm_net_err_last_ts = now
        log.error("Sin conexión al LLM (%s). base=%s — comprueba que LM Studio está corriendo.", exc, LMSTUDIO_OPENAI_BASE)

    else:
        log.debug("LLM (red, suprimido): %s", exc)


def invoke_direct_llm(decision_llm, mission_state, system_prompt: str, img_b64: str, frame_id: int, img_b64_fb: str | None = None, down_b64: str | None = None) -> tuple[float, float, str]:
    pos = mission_state.position
    user_text = build_vlm_user_text(frame_id, pos, mission_state)

    def _messages(b64: str, down: str | None = None):
        if down:
            content: list = [
                {"type": "text", "text": "Image 1 — Front camera (forward-facing):"},
                {"type": "image_url", "image_url": {"url": jpeg_b64_to_data_url(b64)}},
                {"type": "text", "text": "Image 2 — Down camera (ground-facing, landing view):"},
                {"type": "image_url", "image_url": {"url": jpeg_b64_to_data_url(down)}},
            ]

        else:
            content: list = [
                {"type": "image_url", "image_url": {"url": jpeg_b64_to_data_url(b64)}},
            ]

        content.append({"type": "text", "text": user_text})
        return [SystemMessage(content=system_prompt), HumanMessage(content=content)]

    try:
        t0 = time.time()
        try:
            response = decision_llm.invoke(_messages(img_b64, down_b64))

        except Exception as e1:
            if img_b64_fb and is_context_window_error(e1):
                get_logger("invoke_direct").warning("Context overflow; reintentando frame reducido.")
                response = decision_llm.invoke(_messages(img_b64_fb))

            else:
                raise e1
            
        latency = time.time() - t0
        answer = (response.content or "").strip()
        movement, rotation = parse_movement(answer)

        # respuesta sin línea de comando, repetir el comando anterior
        if answer and not re.search(r"(movement|rotation)\s*[:=]", answer.lower()):
            prev_m = mission_state.metadata.get("last_movement")
            prev_r = mission_state.metadata.get("last_rotation")
            if prev_m is not None and prev_r is not None:
                movement, rotation = float(prev_m), float(prev_r)
            get_logger("invoke_direct").warning(
                "Frame %d: respuesta sin comando (¿truncada por max_tokens?); "
                "repitiendo comando anterior mov=%.2f rot=%.2f",
                frame_id, movement, rotation,
            )

        tokens = extract_token_usage(response)
        model_id = getattr(decision_llm, "model_name", None) or getattr(decision_llm, "model", "direct_llm")
        MissionLogger().log_llm_call(model=str(model_id), prompt_tokens=tokens.get("prompt_tokens", 0), completion_tokens=tokens.get("completion_tokens", 0), total_tokens=tokens.get("total_tokens", 0), latency_s=latency, response_preview=answer)
        _log_llm_request(frame_id, "direct", system_prompt, user_text, img_b64, answer, latency)
        data = {"frame_id": frame_id, "movement": movement, "rotation": rotation}
        mission_state.log_event("vlm_direct", "decision_made", data)
        return movement, rotation, answer

    except Exception as e:
        if is_probably_network_llm_error(e):
            _log_throttled_llm_network_error("invoke_direct", e)

        else:
            get_logger("invoke_direct").error("LLM directo falló: %s", e, exc_info=True)

        return 0.0, 0.0, ""


@dataclass
class LLMResult:
    movement: float = 0.8
    rotation: float = 0.0
    answer: str = "movement=0.8, rotation=0.0"
    latency_s: float = 0.0
    frame_id: int = 0
    phase: str = "llm_pending"


class LLMWorker:
    def __init__(self):
        self._queue: queue.Queue[dict | None] = queue.Queue(maxsize=1)
        self.last_result: LLMResult = LLMResult()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self.is_busy = False
        self._total_inferences = 0
        self._total_dropped = 0
        self._decision_llm = None
        self._mission_state = None
        self._system_prompt: str = ""

    def start(self, decision_llm, mission_state, system_prompt: str) -> None:
        self._decision_llm = decision_llm
        self._mission_state = mission_state
        self._system_prompt = system_prompt
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="llm-worker", daemon=True)
        self._thread.start()
        get_logger("llm_worker").info("LLM Worker arrancado.")

    def submit(self, img_b64: str, frame_id: int, img_b64_fb: str | None = None, down_b64: str | None = None) -> bool:
        if self._stop_event.is_set():
            return False
        
        payload = {"img_b64": img_b64, "frame_id": frame_id, "img_b64_fb": img_b64_fb, "down_b64": down_b64}
        if self._queue.full():
            try:
                self._queue.get_nowait()
                self._total_dropped += 1

            except Exception:
                pass
        try:
            self._queue.put_nowait(payload)
            return True
        
        except Exception:
            return False

    def get_result(self) -> LLMResult:
        with self._lock:
            return LLMResult(movement=self.last_result.movement, rotation=self.last_result.rotation, answer=self.last_result.answer, latency_s=self.last_result.latency_s, frame_id=self.last_result.frame_id, phase=self.last_result.phase)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        try:
            self._queue.put_nowait(None)

        except Exception:
            pass

        if self._thread is not None:
            self._thread.join(timeout=timeout)

        get_logger("llm_worker").info("LLM Worker detenido. Inferencias: %d, Frames descartados: %d", self._total_inferences, self._total_dropped)

    def _run(self) -> None:
        logger = get_logger("llm_worker")
        while not self._stop_event.is_set():
            try:
                payload = self._queue.get(timeout=1.0)

            except Exception:
                continue

            if payload is None:
                break

            self.is_busy = True
            try:
                t0 = time.time()
                movement, rotation, answer = invoke_direct_llm(self._decision_llm, self._mission_state, self._system_prompt,payload["img_b64"], payload["frame_id"], img_b64_fb=payload.get("img_b64_fb"), down_b64=payload.get("down_b64"))
                latency = time.time() - t0
                with self._lock:
                    self.last_result = LLMResult(movement=movement, rotation=rotation, answer=answer, latency_s=latency, frame_id=payload["frame_id"], phase="llm_async")

                self._total_inferences += 1
            except Exception as e:
                logger.error("Error en LLM Worker (frame %d): %s", payload["frame_id"], e)

            finally:
                self.is_busy = False


_llm_worker = LLMWorker()
