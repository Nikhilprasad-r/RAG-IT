"""Tracing setup: send ragkit's OpenTelemetry spans to Langfuse, Phoenix or any OTLP backend.

Configured under `observability` in ragkit.yaml. The CLI and server call
`configure_tracing` once at start-up; library code only uses the OpenTelemetry
API, so nothing is exported when the exporter is `none`.

If the process already has an OpenTelemetry SDK tracer provider (your own
setup, or tests), ragkit adds its exporter to that provider instead of
replacing it.
"""

import atexit
import base64
import os
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

from ragkit.core.config import ObservabilityConfig, ProjectConfig
from ragkit.core.errors import ConfigError
from ragkit.core.telemetry import set_capture_content

DEFAULT_OTLP_ENDPOINT = "http://localhost:4318/v1/traces"
DEFAULT_PHOENIX_ENDPOINT = "http://localhost:6006/v1/traces"

_configured: set[str] = set()


def configure_tracing(project: ProjectConfig) -> TracerProvider | None:
    """Install the configured exporter. Safe to call more than once per process."""
    config = project.observability
    set_capture_content(config.capture_content)
    if config.exporter == "none":
        return None

    exporter = build_exporter(config)
    key = f"{config.exporter}:{config.endpoint}"
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider(
            resource=Resource.create({"service.name": config.service_name or project.name}),
            sampler=ParentBased(TraceIdRatioBased(config.sample_ratio)),
        )
        trace.set_tracer_provider(provider)
        atexit.register(provider.shutdown)
    if key not in _configured:
        processor = (
            SimpleSpanProcessor(exporter)
            if config.exporter == "console"
            else BatchSpanProcessor(exporter)
        )
        provider.add_span_processor(processor)
        _configured.add(key)
    return provider


def build_exporter(config: ObservabilityConfig) -> SpanExporter:
    if config.exporter == "console":
        return ConsoleSpanExporter()
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    endpoint, headers = otlp_target(config)
    return OTLPSpanExporter(endpoint=endpoint, headers=headers)


def otlp_target(config: ObservabilityConfig) -> tuple[str, dict[str, str]]:
    """Endpoint and headers for the OTLP/HTTP exporters."""
    headers = dict(config.headers)
    if config.exporter == "langfuse":
        public = os.environ.get(config.langfuse.public_key_env)
        secret = os.environ.get(config.langfuse.secret_key_env)
        if not (public and secret):
            raise ConfigError(
                f"The langfuse exporter needs {config.langfuse.public_key_env} and "
                f"{config.langfuse.secret_key_env} set."
            )
        token = base64.b64encode(f"{public}:{secret}".encode()).decode()
        headers.setdefault("Authorization", f"Basic {token}")
        default = config.langfuse.host.rstrip("/") + "/api/public/otel/v1/traces"
    elif config.exporter == "phoenix":
        default = DEFAULT_PHOENIX_ENDPOINT
    else:
        default = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", DEFAULT_OTLP_ENDPOINT)
    return config.endpoint or default, headers


def instrument_app(app: Any, project: ProjectConfig) -> None:
    """Add HTTP server spans to a FastAPI app when tracing is on."""
    if project.observability.exporter == "none":
        return
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app, excluded_urls="health")


def flush() -> None:
    provider = trace.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        provider.force_flush()


__all__ = ["build_exporter", "configure_tracing", "flush", "instrument_app", "otlp_target"]
