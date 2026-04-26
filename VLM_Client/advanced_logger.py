"""
advanced_logger.py — Sistema de logs avanzado para VLM_Client.

Proporciona logging centralizado y estructurado para toda la arquitectura
del dron autónomo. Captura datos significativos como uso de tokens,
latencias de LLM, decisiones del agente, invocaciones de herramientas,
operaciones de memoria y eventos de conexión.

Arquitectura:
  ┌─────────────────────────────────────────────────────┐
  │                   MissionLogger                      │
  │                                                      │
  │  Console Handler ──► Salida coloreada y concisa      │
  │  File Handler    ──► JSONL estructurado (rotativo)   │
  │  Stats Tracker   ──► Contadores acumulativos         │
  └─────────────────────────────────────────────────────┘

Uso:
    from advanced_logger import MissionLogger, get_logger

    ml = MissionLogger()
    ml.log_llm_call("qwen3-vl-8b", 120, 45, 165, 1.5, "movement=0.5...")
    stats = ml.get_session_stats()
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any


# ================= COLORES PARA CONSOLA =================

class _Colors:
    """Códigos ANSI para salida coloreada en terminal."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"

    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"

    # Combinaciones para categorías
    SYSTEM  = "\033[92m"   # Verde
    LLM     = "\033[96m"   # Cyan
    AGENT   = "\033[94m"   # Azul
    TOOL    = "\033[95m"   # Magenta
    FRAME   = "\033[93m"   # Amarillo
    MEMORY  = "\033[97m"   # Blanco
    ERROR   = "\033[91m"   # Rojo
    STATS   = "\033[1;96m" # Cyan bold


# ================= FORMATTER PARA CONSOLA =================

class ColoredConsoleFormatter(logging.Formatter):
    """Formatter que aplica colores según el nivel y categoría del log."""

    LEVEL_COLORS = {
        logging.DEBUG:    _Colors.DIM,
        logging.INFO:     _Colors.WHITE,
        logging.WARNING:  _Colors.YELLOW,
        logging.ERROR:    _Colors.RED,
        logging.CRITICAL: f"{_Colors.BOLD}{_Colors.RED}",
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelno, _Colors.WHITE)

        # Usar color de categoría si está disponible
        category = getattr(record, "category", None)
        if category:
            cat_color = getattr(_Colors, category.upper(), None)
            if cat_color:
                color = cat_color

        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        level = record.levelname[0]  # D, I, W, E, C
        prefix = getattr(record, "prefix", "")

        if prefix:
            formatted = (
                f"{_Colors.DIM}{timestamp}{_Colors.RESET} "
                f"{color}{level} [{prefix}]{_Colors.RESET} "
                f"{record.getMessage()}"
            )
        else:
            formatted = (
                f"{_Colors.DIM}{timestamp}{_Colors.RESET} "
                f"{color}{level}{_Colors.RESET} "
                f"{record.getMessage()}"
            )

        return formatted


# ================= FORMATTER PARA ARCHIVO (JSONL) =================

class JsonLineFormatter(logging.Formatter):
    """Formatter que serializa cada log como una línea JSON."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Añadir campos extra si existen
        for attr in ("category", "prefix", "data"):
            val = getattr(record, attr, None)
            if val is not None:
                entry[attr] = val

        return json.dumps(entry, ensure_ascii=False)


# ================= MISSION LOGGER =================

class MissionLogger:
    """
    Logger centralizado para el sistema VLM_Client.

    Gestiona:
      - Salida a consola con colores por categoría
      - Archivo JSONL estructurado por sesión con rotación
      - Contadores acumulativos de tokens y métricas de sesión
    """

    _instance: "MissionLogger | None" = None

    def __new__(cls, *args, **kwargs):
        """Patrón singleton: una sola instancia por proceso."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        log_dir: str | None = None,
        console_level: int = logging.INFO,
        file_level: int = logging.INFO,
        max_file_bytes: int = 5 * 1024 * 1024,  # 5 MB
        backup_count: int = 10,
    ):
        if self._initialized:
            return
        self._initialized = True

        # ---- Sesión ----
        self.session_id = uuid.uuid4().hex[:12]
        self.session_start = datetime.now(timezone.utc)
        self.session_start_mono = time.monotonic()

        # ---- Contadores acumulativos ----
        self._stats = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "total_llm_calls": 0,
            "total_llm_latency_s": 0.0,
            "total_tool_calls": 0,
            "total_tool_latency_s": 0.0,
            "frames_processed": 0,
            "commands_sent": 0,
            "memory_updates": 0,
            "errors": 0,
        }
        # Muestreo para evitar logs por frame excesivos.
        self._log_every_n_frames = max(1, int(os.getenv("VLM_LOG_EVERY_N_FRAMES", "20")))

        # ---- Directorio de logs ----
        if log_dir is None:
            log_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "logs"
            )
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir

        # ---- Logger raíz del proyecto ----
        self._root_logger = logging.getLogger("vlm_client")
        self._root_logger.setLevel(logging.DEBUG)
        self._root_logger.propagate = False

        # Limpiar handlers previos (por si se reimporta)
        self._root_logger.handlers.clear()

        # ---- Console handler ----
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(ColoredConsoleFormatter())
        self._root_logger.addHandler(console_handler)

        # ---- File handler (JSONL, rotativo) ----
        session_ts = self.session_start.strftime("%Y%m%d_%H%M%S")
        log_filename = f"session_{session_ts}_{self.session_id}.jsonl"
        log_path = os.path.join(log_dir, log_filename)

        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_file_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(JsonLineFormatter())
        self._root_logger.addHandler(file_handler)

        self.log_path = log_path

        # ---- Log de inicio ----
        self._log("system", logging.INFO, "SYS",
                  f"Session started | id={self.session_id} | "
                  f"log_file={log_filename}")

    # ---------- Método interno de logging ----------

    def _log(
        self,
        category: str,
        level: int,
        prefix: str,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Escribe un log con categoría, prefijo y datos opcionales."""
        extra = {
            "category": category,
            "prefix": prefix,
            "data": data,
        }
        self._root_logger.log(level, message, extra=extra)

    # ================================================================
    # MÉTODOS ESPECIALIZADOS DE LOGGING
    # ================================================================

    def log_llm_call(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        latency_s: float,
        response_preview: str = "",
    ) -> None:
        """
        Registra una llamada a un modelo LLM con detalle de tokens.

        Args:
            model:             Nombre/ID del modelo.
            prompt_tokens:     Tokens de entrada (prompt).
            completion_tokens: Tokens de salida (respuesta).
            total_tokens:      Total de tokens consumidos.
            latency_s:         Latencia de la llamada en segundos.
            response_preview:  Primeros caracteres de la respuesta.
        """
        self._stats["total_prompt_tokens"] += prompt_tokens
        self._stats["total_completion_tokens"] += completion_tokens
        self._stats["total_tokens"] += total_tokens
        self._stats["total_llm_calls"] += 1
        self._stats["total_llm_latency_s"] += latency_s

        avg_latency = (
            self._stats["total_llm_latency_s"] / self._stats["total_llm_calls"]
        )

        preview = response_preview[:100].replace("\n", " ") if response_preview else ""

        self._log("llm", logging.INFO, "LLM",
                  f"{model} | tokens={prompt_tokens}+{completion_tokens}={total_tokens} | "
                  f"latency={latency_s:.2f}s | avg={avg_latency:.2f}s | "
                  f"session_tokens={self._stats['total_tokens']}",
                  data={
                      "model": model,
                      "prompt_tokens": prompt_tokens,
                      "completion_tokens": completion_tokens,
                      "total_tokens": total_tokens,
                      "latency_s": round(latency_s, 4),
                      "response_preview": preview,
                      "session_total_tokens": self._stats["total_tokens"],
                      "session_llm_calls": self._stats["total_llm_calls"],
                  })

    def log_agent_decision(
        self,
        frame_id: int,
        movement: float,
        rotation: float,
        latency_s: float,
        raw_response: str = "",
    ) -> None:
        """Registra una decisión de vuelo del agente."""
        if (
            frame_id > 3
            and frame_id % self._log_every_n_frames != 0
            and abs(rotation) < 0.2
            and movement > 0.05
        ):
            return

        preview = raw_response[:120].replace("\n", " ") if raw_response else ""

        self._log("agent", logging.INFO, "DECISION",
                  f"frame={frame_id} | mov={movement:.3f} rot={rotation:.3f} | "
                  f"latency={latency_s:.2f}s",
                  data={
                      "frame_id": frame_id,
                      "movement": round(movement, 4),
                      "rotation": round(rotation, 4),
                      "latency_s": round(latency_s, 4),
                      "raw_response": preview,
                  })

    def log_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        result: str = "",
        latency_s: float = 0.0,
    ) -> None:
        """Registra una invocación de herramienta del agente."""
        self._stats["total_tool_calls"] += 1
        self._stats["total_tool_latency_s"] += latency_s

        result_preview = str(result)[:100] if result else ""

        self._log("tool", logging.INFO, "TOOL",
                  f"{tool_name} | latency={latency_s:.3f}s | "
                  f"result={result_preview}",
                  data={
                      "tool_name": tool_name,
                      "args": args,
                      "result_preview": result_preview,
                      "latency_s": round(latency_s, 4),
                  })

    def log_frame_received(
        self,
        frame_id: int,
        width: int,
        height: int,
        pos_x: float = 0.0,
        pos_y: float = 0.0,
        pos_z: float = 0.0,
    ) -> None:
        """Registra la recepción de un frame de la cámara del dron."""
        self._stats["frames_processed"] += 1
        if frame_id > 3 and frame_id % self._log_every_n_frames != 0:
            return

        self._log("frame", logging.DEBUG, "FRAME",
                  f"#{frame_id:04d} | {width}x{height} | "
                  f"pos=({pos_x:.2f}, {pos_y:.2f}, {pos_z:.2f}) | "
                  f"total={self._stats['frames_processed']}",
                  data={
                      "frame_id": frame_id,
                      "width": width,
                      "height": height,
                      "pos_x": round(pos_x, 4),
                      "pos_y": round(pos_y, 4),
                      "pos_z": round(pos_z, 4),
                  })

    def log_command_sent(
        self,
        frame_id: int,
        vx: float,
        vy: float,
        vz: float,
        yaw: float,
    ) -> None:
        """Registra el envío de un comando de vuelo al dron."""
        self._stats["commands_sent"] += 1
        if (
            frame_id > 3
            and frame_id % self._log_every_n_frames != 0
            and abs(yaw) < 0.15
            and abs(vz) < 0.01
        ):
            return

        self._log("agent", logging.INFO, "CMD",
                  f"frame={frame_id} | vx={vx:.3f} vy={vy:.3f} "
                  f"vz={vz:.3f} yaw={yaw:.3f}",
                  data={
                      "frame_id": frame_id,
                      "vx": round(vx, 4),
                      "vy": round(vy, 4),
                      "vz": round(vz, 4),
                      "yaw": round(yaw, 4),
                  })

    def log_memory_update(
        self,
        summary_length: int,
        events_summarized: int,
        latency_s: float = 0.0,
    ) -> None:
        """Registra una actualización de la memoria/resumen estratégico."""
        self._stats["memory_updates"] += 1

        self._log("memory", logging.INFO, "MEM",
                  f"Summary updated | length={summary_length} chars | "
                  f"events_summarized={events_summarized} | "
                  f"latency={latency_s:.2f}s",
                  data={
                      "summary_length": summary_length,
                      "events_summarized": events_summarized,
                      "latency_s": round(latency_s, 4),
                  })

    def log_connection_event(
        self,
        event_type: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Registra eventos de conexión (connect, disconnect, error)."""
        level = logging.WARNING if "error" in event_type.lower() or "lost" in event_type.lower() else logging.INFO

        self._log("system", level, "CONN",
                  f"{event_type} | {details or ''}",
                  data={
                      "event_type": event_type,
                      "details": details,
                  })

    def log_system(
        self,
        message: str,
        level: int = logging.INFO,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Registra un evento genérico del sistema."""
        self._log("system", level, "SYS", message, data=data)

    def log_error(
        self,
        component: str,
        error: Exception | str,
        traceback_str: str = "",
    ) -> None:
        """Registra un error estructurado."""
        self._stats["errors"] += 1
        error_msg = str(error)[:300]

        self._log("error", logging.ERROR, component.upper(),
                  f"{error_msg}",
                  data={
                      "component": component,
                      "error": error_msg,
                      "traceback": traceback_str[:500] if traceback_str else "",
                  })

    # ================================================================
    # ESTADÍSTICAS DE SESIÓN
    # ================================================================

    def get_session_stats(self) -> dict[str, Any]:
        """
        Devuelve las estadísticas acumuladas de la sesión actual.

        Returns:
            Dict con contadores de tokens, latencias, frames, etc.
        """
        elapsed = time.monotonic() - self.session_start_mono
        stats = dict(self._stats)
        stats.update({
            "session_id": self.session_id,
            "session_start": self.session_start.isoformat(),
            "session_elapsed_s": round(elapsed, 2),
            "avg_llm_latency_s": round(
                stats["total_llm_latency_s"] / max(1, stats["total_llm_calls"]), 3
            ),
            "avg_tool_latency_s": round(
                stats["total_tool_latency_s"] / max(1, stats["total_tool_calls"]), 3
            ),
            "tokens_per_minute": round(
                stats["total_tokens"] / max(1, elapsed / 60), 1
            ),
        })
        return stats

    def log_session_summary(self) -> None:
        """Imprime un resumen completo de la sesión al log."""
        stats = self.get_session_stats()

        summary_lines = [
            "",
            "=" * 62,
            "  SESSION SUMMARY",
            "=" * 62,
            f"  Session ID:    {stats['session_id']}",
            f"  Duration:      {stats['session_elapsed_s']:.1f}s",
            f"  Log file:      {self.log_path}",
            "",
            "  ── LLM Usage ──",
            f"  Total calls:         {stats['total_llm_calls']}",
            f"  Prompt tokens:       {stats['total_prompt_tokens']}",
            f"  Completion tokens:   {stats['total_completion_tokens']}",
            f"  Total tokens:        {stats['total_tokens']}",
            f"  Tokens/min:          {stats['tokens_per_minute']}",
            f"  Avg latency:         {stats['avg_llm_latency_s']:.3f}s",
            f"  Total LLM time:      {stats['total_llm_latency_s']:.1f}s",
            "",
            "  ── Operations ──",
            f"  Frames processed:    {stats['frames_processed']}",
            f"  Commands sent:       {stats['commands_sent']}",
            f"  Tool calls:          {stats['total_tool_calls']}",
            f"  Memory updates:      {stats['memory_updates']}",
            f"  Errors:              {stats['errors']}",
            "=" * 62,
            "",
        ]

        for line in summary_lines:
            self._log("stats", logging.INFO, "STATS", line)

        # También guardar como JSON en el log
        self._log("stats", logging.INFO, "STATS",
                  "Session stats snapshot",
                  data=stats)


# ================= FUNCIÓN DE CONVENIENCIA =================

def get_logger(name: str = "vlm_client") -> logging.Logger:
    """
    Devuelve un logger hijo del logger raíz de VLM_Client.
    Útil para que cada módulo tenga su propio logger nombrado.

    Args:
        name: Nombre del sub-logger (ej: 'agent_tools', 'hybrid_memory').

    Returns:
        Logger de Python configurado.
    """
    return logging.getLogger(f"vlm_client.{name}")


# ================= EXTRACCIÓN DE TOKEN USAGE =================

def extract_token_usage(response) -> dict[str, int]:
    """
    Extrae el uso de tokens de una respuesta de LangChain/OpenAI.

    Intenta múltiples ubicaciones para máxima compatibilidad:
      - response.usage_metadata (LangChain >=0.3)
      - response.response_metadata.token_usage (LangChain + OpenAI)

    Args:
        response: Objeto de respuesta de un LLM LangChain.

    Returns:
        Dict con prompt_tokens, completion_tokens, total_tokens.
        Valores por defecto 0 si no se encuentran.
    """
    usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    # Intento 1: usage_metadata (LangChain >=0.3 usa input_tokens/output_tokens;
    #            LMStudio nativo usa prompt_tokens/completion_tokens — soportamos ambos)
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        um = response.usage_metadata
        if isinstance(um, dict):
            usage["prompt_tokens"] = um.get("prompt_tokens") or um.get("input_tokens", 0)
            usage["completion_tokens"] = um.get("completion_tokens") or um.get("output_tokens", 0)
            usage["total_tokens"] = um.get("total_tokens", 0)
        else:
            usage["prompt_tokens"] = getattr(um, "input_tokens", 0)
            usage["completion_tokens"] = getattr(um, "output_tokens", 0)
            usage["total_tokens"] = getattr(um, "total_tokens", 0)
        if usage["total_tokens"] == 0:
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        return usage

    # Intento 2: response_metadata.token_usage
    if hasattr(response, "response_metadata"):
        rm = response.response_metadata
        if isinstance(rm, dict):
            tu = rm.get("token_usage") or rm.get("usage", {})
            if tu:
                usage["prompt_tokens"] = tu.get("prompt_tokens", 0)
                usage["completion_tokens"] = tu.get("completion_tokens", 0)
                usage["total_tokens"] = tu.get("total_tokens", 0)
                if usage["total_tokens"] == 0:
                    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                return usage

    return usage