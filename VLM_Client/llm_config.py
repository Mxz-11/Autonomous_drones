"""
llm_config.py — Configuración centralizada de modelos LLM locales.

Utiliza ChatOpenAI de LangChain apuntando a LM Studio (compatible con OpenAI API).
Define múltiples modelos: uno para decisiones del dron y otro para resúmenes estratégicos.
"""

from langchain_openai import ChatOpenAI

# ================= CONFIGURACIÓN DE MODELOS =================
LMSTUDIO_BASE_URL = "http://localhost:1235/v1"

DECISION_MODEL_ID = "qwen/qwen3-vl-8b"
DECISION_TEMPERATURE = 0.3       
DECISION_MAX_TOKENS = 256        

SUMMARY_MODEL_ID = "qwen/qwen3-vl-8b" 
SUMMARY_TEMPERATURE = 0.5
SUMMARY_MAX_TOKENS = 512         


def get_decision_llm() -> ChatOpenAI:
    """
    Devuelve el LLM configurado para tomar decisiones de vuelo.
    Conecta a LM Studio en localhost.
    """
    return ChatOpenAI(
        base_url=LMSTUDIO_BASE_URL,
        api_key="lm-studio",           
        model=DECISION_MODEL_ID,
        temperature=DECISION_TEMPERATURE,
        max_tokens=DECISION_MAX_TOKENS,
    )

def get_summary_llm() -> ChatOpenAI:
    """
    Devuelve el LLM configurado para generar resúmenes estratégicos.
    Puede apuntar a un modelo diferente si se carga uno más ligero.
    """
    return ChatOpenAI(
        base_url=LMSTUDIO_BASE_URL,
        api_key="lm-studio",
        model=SUMMARY_MODEL_ID,
        temperature=SUMMARY_TEMPERATURE,
        max_tokens=SUMMARY_MAX_TOKENS,
    )

def get_custom_llm(model_id: str, temperature: float = 0.3,
                   max_tokens: int = 256, base_url: str | None = None) -> ChatOpenAI:
    """
    Devuelve un LLM personalizado. Útil si se quiere usar un tercer modelo
    o apuntar a otro servidor compatible con OpenAI API.

    Args:
        model_id:    Identificador del modelo en LM Studio.
        temperature: Temperatura de generación (0.0 - 1.0).
        max_tokens:  Máximo de tokens en la respuesta.
        base_url:    URL base del servidor (por defecto usa LMSTUDIO_BASE_URL).
    """
    return ChatOpenAI(
        base_url=base_url or LMSTUDIO_BASE_URL,
        api_key="lm-studio",
        model=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
    )

if __name__ == "__main__":
    print("=" * 50)
    print("Test de conexión a LM Studio")
    print("=" * 50)

    try:
        decision_llm = get_decision_llm()
        print(f"[OK] Modelo de decisión: {DECISION_MODEL_ID}")
        print(f"     URL: {LMSTUDIO_BASE_URL}")

        summary_llm = get_summary_llm()
        print(f"[OK] Modelo de resumen:  {SUMMARY_MODEL_ID}")

        print("\n[TEST] Enviando prompt de prueba al modelo de decisión...")
        response = decision_llm.invoke("Responde solo 'OK' si estás funcionando.")
        print(f"[RESPUESTA] {response.content}")
        print("\n[OK] Conexión a LM Studio verificada correctamente.")

    except Exception as e:
        print(f"\n[ERROR] No se pudo conectar a LM Studio: {e}")
        print("        Asegúrate de que LM Studio está corriendo en localhost:1235")
        print("        y que tienes un modelo cargado.")
