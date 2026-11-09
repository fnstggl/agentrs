"""Metrics (Prometheus), structured JSON logs, OpenTelemetry tracing."""

from __future__ import annotations

import json
import logging
import os
import sys
import time

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest


class Metrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        self.actions = Counter("agentrs_actions_total", "Gateway decisions", ["provider", "operation", "status"],
                               registry=self.registry)
        self.agents_active = Gauge("agentrs_agents_active", "Active agents", registry=self.registry)
        self.jobs = Gauge("agentrs_jobs", "Durable jobs by kind/status", ["kind", "status"], registry=self.registry)
        self.followups_pending = Gauge("agentrs_followups_pending", "Pending delayed-outcome observations",
                                       registry=self.registry)
        self.inference_batches = Counter("agentrs_inference_batches_total", "Inference batches", registry=self.registry)
        self.task_seconds = Histogram("agentrs_task_seconds", "Wall time per task", registry=self.registry)
        self.egress = Counter("agentrs_egress_total", "Outbound requests", ["provider", "outcome"], registry=self.registry)

    def action(self, provider: str, operation: str, status: str) -> None:
        self.actions.labels(provider, operation, status).inc()

    def render(self) -> bytes:
        return generate_latest(self.registry)


class JsonFormatter(logging.Formatter):
    def format(self, r: logging.LogRecord) -> str:
        msg = r.getMessage()
        try:
            body = json.loads(msg)
        except ValueError:
            body = {"msg": msg}
        return json.dumps({"ts": round(time.time(), 3), "level": r.levelname, "logger": r.name, **body})


def setup_logging(level: str = "INFO") -> None:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    root = logging.getLogger("agentrs")
    root.handlers[:] = [h]
    root.setLevel(level)


def setup_tracing(service: str):
    """OpenTelemetry tracer. Exports over OTLP when OTEL_EXPORTER_OTLP_ENDPOINT is set and the exporter
    is installed; otherwise spans are kept in memory (tests / local)."""
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    provider = TracerProvider(resource=Resource.create({"service.name": service}))
    exporter = None
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            exporter = OTLPSpanExporter()
        except ImportError:
            exporter = None
    mem = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter or mem))
    return provider.get_tracer(service), mem
