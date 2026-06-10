## Misión 1

```bash
DRONE_TARGET_X=27 DRONE_TARGET_Y=0 VLM_CRUISE_Z=1.00 python vlm_client.py --prompt prompts/mision1.md
```

## Misión 2

```bash
python vlm_client.py --prompt prompts/mision2.md
```

## Misión 2 coordinada

```bash
DRONE_TARGET_X=4 DRONE_TARGET_Y=2 VLM_ARRIVAL_HOVER=1 VLM_CRUISE_Z=0.80 python vlm_client.py --prompt prompts/mision2_coor.md
```

## Misión 3

```bash
VLM_ARRIVAL_HOVER=1 VLM_CRUISE_Z=0.90 python vlm_client.py --prompt prompts/mision3.md
```

## Misión 3 coordinada

```bash
DRONE_TARGET_X=9 DRONE_TARGET_Y=-5 VLM_ARRIVAL_HOVER=1 VLM_CRUISE_Z=0.90 python vlm_client.py --prompt prompts/mision3_coor.md
```

## Misión 4

```bash
DRONE_TARGET_X=4 DRONE_TARGET_Y=0 VLM_ARRIVAL_HOVER=1 VLM_CRUISE_Z=0.65 VLM_OBSTACLE_AVOID=0 python vlm_client.py --prompt prompts/mision4.md
```

## Variables de entorno disponibles

| Variable | Descripción | Por defecto |
|---|---|---|
| `DRONE_TARGET_X` | Coordenada X del objetivo (activa guardrails GPS) | `None` (búsqueda visual) |
| `DRONE_TARGET_Y` | Coordenada Y del objetivo | `None` |
| `VLM_ARRIVAL_HOVER` | `1` = hover al llegar, `0` = aterrizar | `0` |
| `VLM_CRUISE_Z` | Altitud de crucero en metros | `0.60` |
| `VLM_OBSTACLE_AVOID` | `0` = desactivar obstacle avoidance CV | `1` |
| `VLM_ASYNC_MODE` | `1` = LLM asíncrono (más frames, respuesta vieja) | `0` |
| `VLM_USE_AGENT` | `1` = modo agente ReAct | `0` |
