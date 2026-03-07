# VLM_Client — Agente LangChain para Control de Dron

Sistema de control inteligente para un dron simulado en **Webots**, usando un agente **LangChain** conectado a un LLM local servido desde **LM Studio**.

## Arquitectura

```
┌─────────────┐    socket     ┌────────────────────┐    OpenAI API    ┌──────────────┐
│   Webots     │◄────────────►│   vlm_client.py    │◄───────────────►│  LM Studio   │
│  (Crazyflie) │  frames/cmd  │   (agente ReAct)   │   decisiones    │  (Qwen 3)    │
└─────────────┘               └─────────┬──────────┘                 └──────────────┘
                                        │
                          ┌─────────────┼─────────────┐
                          ▼             ▼             ▼
                   MissionState   HybridMemory   AgentTools
                   (eventos)      (resumen+buf)  (herramientas)
```

## Estructura de archivos

```
VLM_Client/
├── __init__.py         # Exports del paquete
├── vlm_client.py       # Punto de entrada principal (agente + loop)
├── llm_config.py       # Configuración de modelos LLM locales
├── mission_state.py    # Estado global de misión (log de eventos)
├── hybrid_memory.py    # Memoria híbrida (resumen persistente + buffer)
├── agent_tools.py      # Herramientas invocables por el agente
├── requirements.txt    # Dependencias Python
└── README_AGENT.md     # Esta documentación
```

### Descripción de cada módulo

| Módulo | Responsabilidad |
|--------|-----------------|
| `llm_config.py` | Centraliza la configuración de LLMs locales (LM Studio). Soporta múltiples modelos. |
| `mission_state.py` | Mantiene un log estructurado de eventos `{actor, action, data, timestamp}`. Serializa como JSON. |
| `hybrid_memory.py` | Combina resumen estratégico persistente en disco + buffer de eventos recientes. Actualizable con LLM. |
| `agent_tools.py` | Define 5 herramientas LangChain: `register_event`, `update_memory`, `generate_decision`, `send_full_payload`, `get_mission_status`. |
| `vlm_client.py` | Loop principal: recibe frames de Webots, invoca al agente, envía comandos al dron. Dos modos: agente ReAct o LLM directo. |

---

## Requisitos previos

1. **Python 3.10+**
2. **LM Studio** instalado y configurado
3. **Webots** con el controlador del Crazyflie

---

## Instalación paso a paso

### 1. Crear entorno virtual

```bash
cd TFG
python -m venv venv
source venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install -r VLM_Client/requirements.txt
```

### 3. Configurar LM Studio

1. **Descargar LM Studio** desde [https://lmstudio.ai](https://lmstudio.ai).
2. **Descargar un modelo** compatible (ej: `Qwen/Qwen3-VL-8B`):
   - Abre LM Studio → Search → Busca `qwen3-vl-8b` → Download.
3. **Cargar el modelo**:
   - Ve a la pestaña "Local Server" (icono de `<->` en la barra lateral).
   - Selecciona el modelo descargado.
   - Click en "Start Server".
4. **Verificar el puerto**:
   - Por defecto LM Studio sirve en `http://localhost:1234/v1`.
   - Si usas otro puerto, edita `LMSTUDIO_BASE_URL` en `llm_config.py`.
   - En este proyecto usamos el puerto **1235** (`http://localhost:1235/v1`).

### 4. Verificar conexión al LLM

```bash
cd TFG/VLM_Client
python llm_config.py
```

Salida esperada:
```
[OK] Modelo de decisión: qwen/qwen3-vl-8b
[OK] Modelo de resumen:  qwen/qwen3-vl-8b
[TEST] Enviando prompt de prueba...
[RESPUESTA] OK
[OK] Conexión a LM Studio verificada correctamente.
```

---

## Ejecución

### 1. Iniciar Webots

Abre el mundo del Crazyflie en Webots. El controlador `crazyflie.c` debe estar escuchando en el puerto 9002.

### 2. Iniciar el cliente VLM

```bash
cd TFG/VLM_Client
python vlm_client.py
```

El sistema:
1. Inicializa los LLMs, MissionState, HybridMemory y las herramientas.
2. Conecta a Webots por socket (reintenta automáticamente).
3. Recibe frames de la cámara del dron.
4. Invoca al agente LangChain para decidir movimientos.
5. Envía comandos `vx vy vz yaw` al dron.

### Modos de operación

En `vlm_client.py`, la variable `USE_AGENT_MODE` controla el modo:

| Modo | Variable | Descripción |
|------|----------|-------------|
| **Agente ReAct** | `USE_AGENT_MODE = True` | Agente LangChain con herramientas y razonamiento multi-paso. Más inteligente. |
| **LLM Directo** | `USE_AGENT_MODE = False` | Invocación directa al LLM con imagen + contexto. Más rápido. |

---

## Tests de módulos individuales

Cada módulo tiene un bloque `if __name__ == "__main__"` con pruebas:

```bash
# Test de MissionState (no requiere LM Studio)
python mission_state.py

# Test de HybridMemory (no requiere LM Studio)
python hybrid_memory.py

# Test de AgentTools (no requiere LM Studio)
python agent_tools.py

# Test de conexión a LLM (REQUIERE LM Studio)
python llm_config.py
```

---

## Configuración avanzada

### Usar múltiples modelos

En `llm_config.py`, puedes configurar modelos diferentes para decisiones y resúmenes:

```python
DECISION_MODEL_ID = "qwen/qwen3-vl-8b"     # Modelo multimodal para visión
SUMMARY_MODEL_ID  = "lmstudio-community/Meta-Llama-3.1-8B"  # Modelo texto para resúmenes
```

### Crear un LLM personalizado

```python
from llm_config import get_custom_llm

# Conectar a otro servidor o modelo
my_llm = get_custom_llm(
    model_id="mistral/mistral-7b",
    temperature=0.1,
    max_tokens=128,
    base_url="http://localhost:8080/v1",  # Otro servidor
)
```

### Ajustar la memoria

En `vlm_client.py`:

```python
SUMMARY_UPDATE_INTERVAL = 20   # Cada cuántos eventos actualizar resumen
MAX_RECENT_EVENTS = 30         # Eventos en el buffer de memoria
```

---

## Fichero de resumen persistente

El resumen estratégico se guarda automáticamente en:

```
VLM_Client/mission_summary.json
```

Ejemplo de contenido:
```json
{
  "mission_name": "drone_figure8_mission",
  "summary": "El dron ha completado 3 bucles del ocho...",
  "last_update": "2026-02-24T13:45:00+00:00",
  "events_summarized": 60
}
```

Este fichero persiste entre ejecuciones, permitiendo al agente retomar el contexto.
