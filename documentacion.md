# Documentación del Proyecto

## Arquitectura del VLM Client (`vlm_client.py`)

**vlm_client.py** — Cliente VLM principal con agente LangChain.

Punto de entrada del sistema. Conecta a Webots por socket, recibe frames
de la cámara del dron, y usa un agente LangChain con herramientas para
decidir los movimientos basándose en el estado completo de la misión.

Arquitectura:
```text
  ┌─────────┐    socket     ┌──────────────┐    LangChain    ┌──────────┐
  │ Webots  │◄──────────────│ vlm_client   │────────────────►│ LM Studio│
  │ (dron)  │   cmd/frames  │ (agente)     │   decisiones    │ (LLM)    │
  └─────────┘               └──────┬───────┘                 └──────────┘
                                   │
                        ┌──────────┼──────────┐
                        ▼          ▼          ▼
                 MissionState  HybridMem  AgentTools
```

## Controlador del Dron en Webots (`crazyflie.c`)

El archivo `crazyflie.c` es el controlador escrito en C que se ejecuta dentro del simulador Webots para manejar el dron (Crazyflie). Actúa como la interfaz de bajo nivel que conecta los motores y sensores de Webots con el sistema autónomo de IA de alto nivel. 

Sus responsabilidades y componentes principales son:

1. **Comunicación por Sockets TCP (Cliente-Servidor)**
   - Crea un servidor TCP en el puerto `9002` que acepta conexiones entrantes (por ejemplo, desde el `vlm_client.py`).
   - Escucha instrucciones de velocidad objetivo (`vx, vy, vz, yaw_rate`). 
   - Detecta la palabra clave `FRAME` en el socket, lo cual indica que el cliente de Python desea capturar la cámara en el siguiente paso de simulación.

2. **Sensores y Actuadores**
   - **Sensores activos**: Inercial (IMU para ángulos roll/pitch/yaw), GPS (para coordinadas XYZ) y Giroscopio (para velocidades angulares).
   - **Motores**: Activa los 4 motores del quadcopter en modo velocidad.
   - **Cámara (`vlm_camera`)**: Es leída por el controlador en cada iteración del simulador. 

3. **Sistema PID y Estabilización Automática**
   - Utiliza la librería de control PID importada en `pid_controller.h`.
   - Lee el estado *real* (actual_state) a través del GPS e IMU, y lo compara con el estado *deseado* (desired_state) proveniente del TCP.
   - La función `pid_velocity_fixed_height_controller` calcula la potencia necesaria de los 4 rotores (`motor_power.m1` a `m4`) para mantener la altitud, rotar en su propio eje y navegar fluidamente, evitando que el dron pierda el control.

4. **Transmisión de Imágenes (*Pull-Based*)**
   - Cuando recibe la orden `FRAME`, el controlador recupera el *buffer* completo de la imagen generada por Webots.
   - Primero envía una pequeña cabecera con las dimensiones `[ancho, alto]` e inmediatamente después manda los fotogramas sin procesar (bytes) de forma bloqueante a través de la red TCP en fragmentos (chunks) hasta completar el tamaño de la imagen hacia el cliente.

## Ejecutar el proyecto

1. Abre Webots y carga la escena.

2. Abre una terminal y ejecuta el siguiente comando:

```bash
   cd /VLM_Client
   source venv/bin/activate
   python vlm_client.py
```

3. Abre LM Studio y carga el modelo.


   