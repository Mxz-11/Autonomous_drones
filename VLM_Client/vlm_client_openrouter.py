import socket
import struct
import time
import base64
import io
import json
import requests
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import os
import re

load_dotenv()

# ================= CONFIG =================
WEBOTS_IP = "127.0.0.1"
WEBOTS_PORT = 9002

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPEN_ROUTER_APIKEY")
MODEL_ID = "nvidia/nemotron-nano-12b-v2-vl:free"

FPS = 3  # frames por segundo al VLM

# Factor divisor de velocidad: ralentiza el dron para compensar latencia LLM.
# 1.0 = velocidad normal, 2.0 = mitad de velocidad, 5.0 = un quinto.
# Configurable con: export VLM_SPEED_DIVISOR=2
try:
    SPEED_DIVISOR = max(1.0, float(os.environ.get("VLM_SPEED_DIVISOR", "2.0")))
except ValueError:
    SPEED_DIVISOR = 2.0

# ---- Prompt debug log: guarda imagen + prompt + respuesta por frame ----
_PROMPT_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "prompts")
os.makedirs(_PROMPT_LOG_DIR, exist_ok=True)

SYSTEM_PROMPT_TEXT = (
    "You are controlling a drone. "
    "Decide how the drone should move using two values:\n"
    "- movement: a number between 0 and 1 indicating forward motion.\n"
    "- rotation: a number between -1 and 1 indicating rotation "
    "(negative = left, positive = right).\n\n"
    "Mission:\n"
    "The drone should continuously fly following a figure-eight trajectory.\n"
    "A figure-eight consists of two connected smooth curves in opposite directions.\n\n"
    "Behavior rules:\n"
    "- Always maintain forward movement.\n"
    "- Introduce small rotation while moving forward to create curved paths.\n"
    "- Alternate the rotation direction over time to form two opposite loops.\n"
    "- Do not fly straight for long periods.\n\n"
    "Respond ONLY in this format:\n"
    "movement=<value>, rotation=<value>"
)


def _log_llm_request(frame_id, img_b64, user_text, response_text, latency_s):
    """Guarda imagen PNG + prompt + respuesta del frame para depuración."""
    ts = time.strftime("%H%M%S")
    img_path = os.path.join(_PROMPT_LOG_DIR, f"frame_{frame_id:05d}.png")
    try:
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(img_b64))
    except Exception as e:
        print(f"[WARN] No se pudo guardar imagen frame {frame_id}: {e}")
        img_path = "<error>"

    prompt_path = os.path.join(_PROMPT_LOG_DIR, f"frame_{frame_id:05d}_prompt.json")
    prompt_data = {
        "frame_id": frame_id,
        "timestamp": ts,
        "mode": "openrouter_direct",
        "model": MODEL_ID,
        "system_prompt": SYSTEM_PROMPT_TEXT,
        "user_text": user_text,
        "image_file": os.path.basename(img_path),
        "image_b64_length": len(img_b64),
        "response": response_text,
        "latency_s": round(latency_s, 3),
    }
    try:
        with open(prompt_path, "w") as f:
            json.dump(prompt_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[WARN] No se pudo guardar prompt frame {frame_id}: {e}")
# =========================================


# ================= SOCKET =================
def connect_to_webots():
    """Connect to Webots with retry"""
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((WEBOTS_IP, WEBOTS_PORT))
            print("[OK] Connected to Webots")
            return sock
        except ConnectionRefusedError:
            print("[WARN] Connection refused, retrying in 2s...")
            time.sleep(2)

sock = connect_to_webots()


def recv_exact(n):
    data = b""
    while len(data) < n:
        try:
            packet = sock.recv(n - len(data))
            if not packet:
                raise ConnectionError("Socket closed")
            data += packet
        except Exception as e:
            raise ConnectionError(f"recv failed: {e}")
    return data


last_time = 0
frame_id = 0

print(f"[INIT] Prompts/imágenes guardados en: {_PROMPT_LOG_DIR}")
if SPEED_DIVISOR > 1.0:
    print(f"[SLOW-MO] Velocidad dividida x{SPEED_DIVISOR:.1f} (VLM_SPEED_DIVISOR)")

# ================= LOOP ===================
while True:
    try:
        # ---- Pedir frame (protocolo pull-based del controlador) ----
        try:
            sock.send(b"FRAME\n")
        except BrokenPipeError:
            print("[WARN] BrokenPipe on FRAME request, reconnecting...")
            sock.close()
            sock = connect_to_webots()
            continue

        # ---- Leer header ----
        header = recv_exact(8)
        w, h = struct.unpack("ii", header)

        # Validar header (evitar MemoryError por valores basura)
        if w <= 0 or h <= 0 or w > 2000 or h > 2000:
            print(f"[ERROR] Invalid header: {w}x{h}, reconnecting...")
            sock.close()
            sock = connect_to_webots()
            continue

        img_bytes = recv_exact(w * h * 4)

        # ---- Leer posición GPS (3 floats = 12 bytes) ----
        pos_bytes = recv_exact(12)
        gps_x, gps_y, gps_z = struct.unpack("fff", pos_bytes)

        frame_id += 1

        # ---- Procesar frame ----
        frame = np.frombuffer(img_bytes, np.uint8).reshape((h, w, 4))
        frame = frame[:, :, :3]  # RGBA -> RGB
        image = Image.fromarray(frame)

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        user_text = f"#{frame_id} X={gps_x:.2f} Y={gps_y:.2f} Z={gps_z:.2f}"

        # ---- payload para OpenRouter ----
        payload = {
            "model": MODEL_ID,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": f"{SYSTEM_PROMPT_TEXT}\n\n{user_text}",
                        }
                    ]
                }
            ]
        }

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }

        # ---- llamada al VLM ----
        t_llm_start = time.time()
        answer = None
        try:
            r = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=60)
            resp = r.json()
        except Exception as e:
            print("[ERROR] OpenRouter request failed:", e)
            resp = {"error": str(e)}
        llm_latency = time.time() - t_llm_start

        # ---- parseo ROBUSTO de respuesta ----
        if "choices" in resp:
            try:
                answer = resp["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                answer = None
        elif "message" in resp and "content" in resp["message"]:
            answer = resp["message"]["content"]
        elif "content" in resp:
            answer = resp["content"]

        # Respuesta vacía o error del proveedor → logear y seguir sin mover
        if not answer:
            err_msg = str(resp)[:200]
            print(f"[WARN #{frame_id}] respuesta vacía/error del proveedor: {err_msg}")
            _log_llm_request(
                frame_id=frame_id,
                img_b64=img_b64,
                user_text=user_text,
                response_text=f"<error> {err_msg}",
                latency_s=llm_latency,
            )
            continue

        answer_lower = answer.lower().strip()

        # ---- parseo movement / rotation ----
        movement = 0.0
        rotation = 0.0

        m = re.search(r"movement\s*=\s*([0-9.]+)", answer_lower)
        if m:
            movement = float(m.group(1))

        r = re.search(r"rotation\s*=\s*(-?[0-9.]+)", answer_lower)
        if r:
            rotation = float(r.group(1))

        # ---- guardar prompt + imagen + respuesta para depuración ----
        _log_llm_request(
            frame_id=frame_id,
            img_b64=img_b64,
            user_text=user_text,
            response_text=answer,
            latency_s=llm_latency,
        )

        # ---- control continuo con divisor de velocidad (cámara lenta) ----
        MAX_FORWARD = 0.5   # velocidad máxima
        MAX_YAW = 0.8       # rotación máxima

        vx = (movement * MAX_FORWARD) / SPEED_DIVISOR
        vy = 0.0
        vz = 0.0
        yaw = (rotation * MAX_YAW) / SPEED_DIVISOR

        cmd = f"{vx} {vy} {vz} {yaw}\n"

        try:
            sock.send(cmd.encode())
            print(f"[VLM #{frame_id}] movement={movement:.2f}, rotation={rotation:.2f} "
                  f"({llm_latency:.1f}s) -> {cmd.strip()}")
        except BrokenPipeError:
            print("[WARN] BrokenPipe, reconnecting...")
            sock.close()
            sock = connect_to_webots()

    except ConnectionError as e:
        print(f"[ERROR] Connection lost: {e}, reconnecting...")
        try:
            sock.close()
        except:
            pass
        sock = connect_to_webots()
    except KeyboardInterrupt:
        print("\n[EXIT] User interrupted")
        break
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        time.sleep(1)
