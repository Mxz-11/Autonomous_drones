## Misión 1

```bash
VLM_CRUISE_Z=0.7 python vlm_client.py --prompt prompts/mision1.md --target 12,0
```

## Misión 2

```bash
VLM_ARRIVAL_HOVER=1 VLM_CRUISE_Z=0.80 python vlm_client.py --prompt prompts/mision2_coor.md --target 4,2
```

## Misión 3

```bash
python vlm_client.py --prompt prompts/mision3_coor.md --target 9,-5 --waypoint 6,5
```

## Misión 4

```bash
VLM_ARRIVAL_HOVER=1 VLM_CRUISE_Z=0.70 VLM_OBSTACLE_AVOID=0 python vlm_client.py --prompt prompts/mision4.md --target 12,0
```

> El arco está en X=4, Y=0, pero el target se pone en `6,0` a propósito:
> - `--target` (no `DRONE_TARGET_*`, que se ignoran) es lo único que genera la cabecera `Y-drift` que el prompt usa para centrarse respecto a Y=0.
> - El guardrail dispara `gps_landing`/hover dentro de `ARRIVAL_RADIUS=0.6` del target. Con X=4 frenaría en X≈3.4 (antes del arco); con X=6 solo para en X≈5.4, ya pasada la línea de éxito (X=4.5).
> - `VLM_OBSTACLE_AVOID=0`: el arco no es un obstáculo, hay que cruzarlo, no esquivarlo.

## Variables de entorno disponibles

> Las variables `DRONE_TARGET_X` / `DRONE_TARGET_Y` están **obsoletas y se ignoran**
> (solo imprimen un `[WARN]`). El objetivo se pasa siempre por CLI: `--target X,Y`.

| Variable | Descripción | Por defecto |
|---|---|---|
| `--target X,Y` (CLI) | Objetivo GPS: activa guardrails y la cabecera `Y-drift` | `None` (búsqueda visual) |
| `--waypoint X,Y` (CLI) | Waypoint intermedio (corredor Y hasta cruzar X) | `None` |
| `VLM_ARRIVAL_HOVER` | `1` = hover al llegar, `0` = aterrizar | `0` |
| `VLM_CRUISE_Z` | Altitud de crucero en metros | `0.60` |
| `VLM_OBSTACLE_AVOID` | `0` = desactivar obstacle avoidance CV | `1` |
| `VLM_ASYNC_MODE` | `1` = LLM asíncrono (más frames, respuesta vieja) | `0` |
| `VLM_USE_AGENT` | `1` = modo agente ReAct | `0` |
