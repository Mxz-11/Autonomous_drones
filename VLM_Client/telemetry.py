"""
telemetry.py — Observabilidad con OpenTelemetry.

Proporciona inicialización centralizada de trazas (traces) y métricas
para el sistema de control del dron. Usa exportadores de consola por
defecto; para producción, sustituir por OTLPSpanExporter / OTLPMetricExporter.

Uso:
    from telemetry import init_telemetry, get_tracer, get_meter

    init_telemetry("vlm_client")
    tracer = get_tracer("main_loop")
    meter  = get_meter("drone_metrics")
"""

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    PeriodicExportingMetricReader,
    ConsoleMetricExporter,
)
from opentelemetry.sdk.resources import Resource

# --------- Estado global ---------

_initialized = False


def init_telemetry(
    service_name: str = "vlm_client",
    enable_console: bool = False,
    metric_export_interval_ms: int = 30_000,
) -> None:
    """
    Inicializa OpenTelemetry: TracerProvider + MeterProvider.

    Args:
        service_name:              Nombre del servicio (aparece en trazas).
        enable_console:            Si True usa ConsoleExporter (stdout).
        metric_export_interval_ms: Intervalo de exportación de métricas.
    """
    global _initialized
    if _initialized:
        return

    resource = Resource.create({"service.name": service_name})

    # ---- Traces ----
    tracer_provider = TracerProvider(resource=resource)
    if enable_console:
        tracer_provider.add_span_processor(
            BatchSpanProcessor(ConsoleSpanExporter())
        )
    trace.set_tracer_provider(tracer_provider)

    # ---- Metrics ----
    readers = []
    if enable_console:
        readers.append(
            PeriodicExportingMetricReader(
                ConsoleMetricExporter(),
                export_interval_millis=metric_export_interval_ms,
            )
        )
    meter_provider = MeterProvider(resource=resource, metric_readers=readers)
    metrics.set_meter_provider(meter_provider)

    _initialized = True
    print(f"[OTEL] Telemetry initialized (service={service_name})")


def get_tracer(name: str = "vlm_client") -> trace.Tracer:
    """Devuelve un tracer de OpenTelemetry."""
    return trace.get_tracer(name)


def get_meter(name: str = "vlm_client") -> metrics.Meter:
    """Devuelve un meter de OpenTelemetry."""
    return metrics.get_meter(name)
