import socket
import struct
import time
import base64
import io
import requests
import numpy as np
from PIL import Image

import re

# ================= CONFIG =================
WEBOTS_IP = "127.0.0.1"
WEBOTS_PORT = 9002

LMSTUDIO_URL = "http://localhost:1235/v1/chat/completions"
MODEL_ID = "qwen/qwen3-vl-8b"

FPS = 3  # frames por segundo al VLM
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

# ================= LOOP ===================
while True:
    try:
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
        
        # ---- Procesar frame ----
        frame = np.frombuffer(img_bytes, np.uint8).reshape((h, w, 4))
        frame = frame[:, :, :3]  # RGBA -> RGB
        image = Image.fromarray(frame)

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        # ---- payload para LM Studio ----
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
                            "text": (
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
                        }
                    ]
                }
            ]
        }


        # ---- llamada al VLM ----
        try:
            r = requests.post(LMSTUDIO_URL, json=payload, timeout=60)
            resp = r.json()
        except Exception as e:
            print("[ERROR] LM Studio request failed:", e)
            continue

        # ---- parseo ROBUSTO de respuesta ----
        if "choices" in resp:
            answer = resp["choices"][0]["message"]["content"]
        elif "message" in resp and "content" in resp["message"]:
            answer = resp["message"]["content"]
        elif "content" in resp:
            answer = resp["content"]
        else:
            print("[ERROR] Unknown LM Studio response format:")
            print(resp)
            continue

        answer = answer.lower().strip()

        # ---- parseo movement / rotation ----
        movement = 0.0
        rotation = 0.0

        m = re.search(r"movement\s*=\s*([0-9.]+)", answer)
        if m:
            movement = float(m.group(1))

        r = re.search(r"rotation\s*=\s*(-?[0-9.]+)", answer)
        if r:
            rotation = float(r.group(1))

        # ---- control continuo ----
        MAX_FORWARD = 0.5   # velocidad máxima
        MAX_YAW = 0.8       # rotación máxima

        vx = movement * MAX_FORWARD
        vy = 0.0
        vz = 0.0
        yaw = rotation * MAX_YAW

        cmd = f"{vx} {vy} {vz} {yaw}\n"
        
        try:
            sock.send(cmd.encode())
            print(f"[VLM] movement={movement:.2f}, rotation={rotation:.2f} -> {cmd.strip()}")
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
