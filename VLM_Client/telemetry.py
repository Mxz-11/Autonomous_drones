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

_initialized = False


def init_telemetry(
    service_name: str = "vlm_client",
    enable_console: bool = False,
    metric_export_interval_ms: int = 30_000,
) -> None:
    global _initialized
    if _initialized:
        return

    resource = Resource.create({"service.name": service_name})

    tracer_provider = TracerProvider(resource=resource)
    if enable_console:
        tracer_provider.add_span_processor(
            BatchSpanProcessor(ConsoleSpanExporter())
        )
    trace.set_tracer_provider(tracer_provider)

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
    return trace.get_tracer(name)


def get_meter(name: str = "vlm_client") -> metrics.Meter:
    return metrics.get_meter(name)
