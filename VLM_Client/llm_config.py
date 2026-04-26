"""
Configuración LLM: ChatOpenAI → servidor local estilo OpenAI (LM Studio).

Variables de entorno:
  LMSTUDIO_OPENAI_BASE — URL base API OpenAI-compatible (ej. http://127.0.0.1:1234/v1)
  OPENAI_API_KEY       — clave; LM Studio suele aceptar cualquier string no vacío

LM Studio expone normalmente en otro puerto la API /api/v1/chat nativa;
este cliente usa /v1/chat/completions vía langchain_openai.ChatOpenAI.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

# ================= CONFIGURACIÓN =================

LMSTUDIO_OPENAI_BASE = os.environ.get(
    "LMSTUDIO_OPENAI_BASE",
    "http://127.0.0.1:1235/v1",
)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "lm-studio")

LMSTUDIO_NATIVE_BASE_URL = os.environ.get("LMSTUDIO_NATIVE_BASE", "http://localhost:1235")


def probe_openai_compatible_server(timeout: float = 5.0) -> tuple[bool, str]:
    """
    Comprueba reachability de la API estilo OpenAI (LM Studio muestra el URL al iniciar el servidor local).
    Debe responder GET {base}/models.
    """
    base = LMSTUDIO_OPENAI_BASE.rstrip("/")
    url = f"{base}/models"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code == 200:
            return True, url
        return False, f"{url} → HTTP {r.status_code}: {r.text[:400]!r}"
    except requests.exceptions.ConnectionError as e:
        return False, (
            f"No conecta a {url} ({e}). "
            "En LM Studio: cargar el modelo → Developer / Local Server → Start Server. "
            "Copia ese puerto en LMSTUDIO_OPENAI_BASE (ej. http://127.0.0.1:1234/v1). "
            "El puerto /api/v1/chat nativo (p. ej. 1235) no es este endpoint."
        )
    except requests.exceptions.Timeout:
        return False, f"Timeout esperando {url}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def is_probably_network_llm_error(exc: BaseException) -> bool:
    s = f"{type(exc).__name__} {exc}".lower()
    needles = (
        "connection error",
        "connection refused",
        "connecterror",
        "failed to establish",
        "name or service not known",
        "errno 61",
        "errno 111",
        "errno 99",
        "broken pipe",
        "timed out",
        "timeout",
        "unreachable",
    )
    return any(n in s for n in needles)
LMSTUDIO_CHAT_PATH = "/api/v1/chat"

# Modelos disponibles en LM Studio (descomentar para probar)
# DECISION_MODEL_ID = "ggml-org/smolvlm2-2.2b-instruct"
DECISION_MODEL_ID = "qwen/qwen3.5-9b"
# DECISION_MODEL_ID = "qwen/qwen3-vl-8b"
DECISION_TEMPERATURE = 0.1
DECISION_MAX_TOKENS = 128
DECISION_TIMEOUT_S = 90.0

SUMMARY_MODEL_ID = "qwen/qwen3-vl-8b"
# SUMMARY_MODEL_ID = "qwen/qwen3.5-9b"
SUMMARY_TEMPERATURE = 0.4
SUMMARY_MAX_TOKENS = 384
SUMMARY_TIMEOUT_S = 90.0


class LMStudioCompatChatOpenAI(ChatOpenAI):
    """
    ChatOpenAI adaptado para LM Studio: intenta bind_tools real para que
    ReAct funcione con modelos que soportan function-calling (ej. Qwen3).
    Si el servidor no lo soporta, cae a self con warning explícito.
    """

    _bind_tools_supported: bool = True

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        if not tools:
            return self
        try:
            bound = super().bind_tools(tools, **kwargs)
            self._bind_tools_supported = True
            return bound
        except Exception as e:
            import logging
            logging.getLogger("llm_config").warning(
                "bind_tools falló (%s). El modo agente ReAct NO usará herramientas. "
                "Si tu modelo soporta function-calling, revisa la configuración de LM Studio.",
                e,
            )
            self._bind_tools_supported = False
            return self


def _base_chat_openai(
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> LMStudioCompatChatOpenAI:
    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "base_url": LMSTUDIO_OPENAI_BASE,
        "api_key": OPENAI_API_KEY,
        "max_retries": 0,
    }
    # Compatibilidad versiones que aún aceptan openai_api_base
    try:
        return LMStudioCompatChatOpenAI(**kwargs)
    except TypeError:
        kwargs["openai_api_base"] = kwargs.pop("base_url")
        return LMStudioCompatChatOpenAI(**kwargs)


def get_decision_llm() -> LMStudioCompatChatOpenAI:
    """Retorna el cliente OpenAI-compatible (soporta image_url multimodal)."""
    return _base_chat_openai(
        DECISION_MODEL_ID,
        DECISION_TEMPERATURE,
        DECISION_MAX_TOKENS,
        DECISION_TIMEOUT_S,
    )


def get_summary_llm() -> LMStudioCompatChatOpenAI:
    return _base_chat_openai(
        SUMMARY_MODEL_ID,
        SUMMARY_TEMPERATURE,
        SUMMARY_MAX_TOKENS,
        SUMMARY_TIMEOUT_S,
    )


def get_custom_llm(
    model_id: str,
    temperature: float = 0.3,
    max_tokens: int = 256,
    base_url: Optional[str] = None,
) -> LMStudioCompatChatOpenAI:
    kwargs: dict[str, Any] = {
        "model": model_id,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": DECISION_TIMEOUT_S,
        "base_url": base_url or LMSTUDIO_OPENAI_BASE,
        "api_key": OPENAI_API_KEY,
        "max_retries": 1,
    }
    try:
        return LMStudioCompatChatOpenAI(**kwargs)
    except TypeError:
        kwargs["openai_api_base"] = kwargs.pop("base_url")
        return LMStudioCompatChatOpenAI(**kwargs)


# ================= API nativa LM Studio (opcional / diagnóstico) =================


class LMStudioChat(BaseChatModel):
    """BaseChatModel → POST /api/v1/chat (sin imágenes multimodales enriquecidas)."""

    model_name: str
    temperature: float = 0.3
    max_tokens: int = 256
    timeout: float = 60.0
    base_url: str = LMSTUDIO_NATIVE_BASE_URL

    @property
    def _llm_type(self) -> str:
        return "lmstudio-native"

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        parts: list[str] = []
        for block in content or []:
            if not isinstance(block, dict):
                parts.append(str(block))
                continue
            t = block.get("type", "")
            if t == "text":
                parts.append(block.get("text", ""))
            elif t == "image_url":
                url = block.get("image_url", {}).get("url", "")
                if url:
                    parts.append(f"[IMAGE:{url[:80]}…]")
        return "\n".join(parts)

    def _build_payload(self, messages: list[BaseMessage]) -> dict:
        system_prompt = ""
        human_turns: list[str] = []
        ai_turns: list[str] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                if not system_prompt:
                    system_prompt = self._content_to_text(msg.content)
            elif isinstance(msg, HumanMessage):
                human_turns.append(self._content_to_text(msg.content))
            elif isinstance(msg, AIMessage):
                ai_turns.append(self._content_to_text(msg.content))
        if len(human_turns) > 1:
            convo_parts: list[str] = []
            for i, h in enumerate(human_turns[:-1]):
                convo_parts.append(f"User: {h}")
                if i < len(ai_turns):
                    convo_parts.append(f"Assistant: {ai_turns[i]}")
            convo_parts.append(f"User: {human_turns[-1]}")
            input_text = "\n".join(convo_parts)
        elif human_turns:
            input_text = human_turns[-1]
        else:
            input_text = ""
        payload: dict[str, Any] = {
            "model": self.model_name,
            "input": input_text,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if system_prompt:
            payload["system_prompt"] = system_prompt
        return payload

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        endpoint = self.base_url.rstrip("/") + LMSTUDIO_CHAT_PATH
        payload = self._build_payload(messages)
        resp = requests.post(endpoint, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        content = ""
        if "output" in data:
            out_obj = data["output"]
            if isinstance(out_obj, list) and out_obj:
                # Buscar el último bloque "message" (no "reasoning")
                message_block = None
                for block in reversed(out_obj):
                    if isinstance(block, dict) and block.get("type") == "message":
                        message_block = block
                        break
                if message_block is not None:
                    content = message_block.get("content", "")
                elif isinstance(out_obj[0], dict):
                    # Fallback: si no hay bloque "message", usar el último
                    content = out_obj[-1].get("content", "")
                else:
                    content = str(out_obj)
            else:
                content = str(out_obj)
        elif "choices" in data:
            choices = data.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")
        # Parsear token usage: formato OpenAI ("usage") o nativo LM Studio ("stats")
        usage_raw = data.get("usage", {})
        if not usage_raw:
            stats = data.get("stats", {})
            usage_raw = {
                "prompt_tokens": stats.get("input_tokens", 0),
                "completion_tokens": stats.get("total_output_tokens", 0),
                "total_tokens": stats.get("input_tokens", 0) + stats.get("total_output_tokens", 0),
            }
        usage_metadata = {
            "prompt_tokens": usage_raw.get("prompt_tokens", 0),
            "completion_tokens": usage_raw.get("completion_tokens", 0),
            "total_tokens": usage_raw.get("total_tokens", 0),
        }
        if usage_metadata["total_tokens"] == 0:
            usage_metadata["total_tokens"] = (
                usage_metadata["prompt_tokens"] + usage_metadata["completion_tokens"]
            )
        ai_message = AIMessage(
            content=content,
            response_metadata={
                "model_name": self.model_name,
                "finish_reason": data.get("finish_reason", "stop"),
                "token_usage": usage_metadata,
            },
            usage_metadata={
                "input_tokens": usage_metadata["prompt_tokens"],
                "output_tokens": usage_metadata["completion_tokens"],
                "total_tokens": usage_metadata["total_tokens"],
                "prompt_tokens": usage_metadata["prompt_tokens"],
                "completion_tokens": usage_metadata["completion_tokens"],
            },
        )
        return ChatResult(generations=[ChatGeneration(message=ai_message)])


def get_decision_llm_native() -> LMStudioChat:
    return LMStudioChat(
        model_name=DECISION_MODEL_ID,
        temperature=DECISION_TEMPERATURE,
        max_tokens=DECISION_MAX_TOKENS,
        timeout=DECISION_TIMEOUT_S,
        base_url=LMSTUDIO_NATIVE_BASE_URL,
    )


if __name__ == "__main__":
    from advanced_logger import get_logger

    logger = get_logger("llm_config_test")
    logger.info("Test ChatOpenAI → %s", LMSTUDIO_OPENAI_BASE)
    try:
        llm = get_decision_llm()
        r = llm.invoke(
            [
                SystemMessage(content="Reply with exactly: OK"),
                HumanMessage(content="ping"),
            ]
        )
        logger.info("R: %s", r.content)
    except Exception as e:
        logger.error("%s", e)
