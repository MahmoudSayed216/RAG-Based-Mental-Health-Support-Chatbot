"""
OpenTelemetry instrumentation for the Mental Health RAG API.

Metrics by category:
  ┌─────────────────┬───────────────────────────────────────────────┐
  │ Model / NLP     │ mental_health_intent_total (Counter)          │
  │                 │ llm_response_duration_seconds (Histogram)     │
  │                 │ llm_tokens_total (Counter)                    │
  │                 │ llm_refusals_total (Counter)                  │
  │                 │ rag_usage_total (Counter)                     │
  │                 │ emotion_total (Counter)                       │
  ├─────────────────┼───────────────────────────────────────────────┤
  │ Data            │ query_length_chars (Histogram)                │
  │                 │ feedback_votes_total (Counter)                │
  │                 │ sessions_total (Counter)                      │
  │                 │ messages_per_session (Histogram)              │
  ├─────────────────┼───────────────────────────────────────────────┤
  │ Server          │ http_requests_total (Counter)                 │
  │                 │ http_errors_total (Counter)                   │
  │                 │ process_uptime_seconds (ObservableGauge)      │
  │                 │ request_duration_seconds (Histogram)          │
  └─────────────────┴───────────────────────────────────────────────┘
"""

import os
import time
from dotenv import load_dotenv

from opentelemetry import metrics, trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.metrics import MeterProvider, Histogram, Counter, UpDownCounter
from opentelemetry.sdk.metrics.export import (
    PeriodicExportingMetricReader,
    AggregationTemporality,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

load_dotenv()

# ─── Module-level instrument handles (populated by setup_telemetry) ───
_intent_counter = None
_llm_duration_histogram = None
_query_length_histogram = None
_feedback_counter = None
_request_counter = None
_error_counter = None
_rag_usage_counter = None
_token_counter = None
_request_duration_histogram = None
_refusal_counter = None
_session_counter = None
_messages_per_session_histogram = None
_emotion_counter = None          # NEW: tracks detected emotions
_tracer = None
_setup_done = False
_language_counter = None
_start_time = time.time()

_REFUSAL_PHRASES = (
    "i can't help with that",
    "i cannot help with that",
    "i'm unable to help",
    "i am unable to assist",
    "i can't assist with that",
    "i cannot provide",
    "i'm not able to",
)


def setup_telemetry(service_name: str = None):
    """
    Initialise the full OTel pipeline:
        App  →  OTLP gRPC  →  OTel Collector  →  Axiom

    Must be called ONCE, after load_dotenv(), before the app starts.
    """
    global _intent_counter, _llm_duration_histogram, _query_length_histogram
    global _feedback_counter, _request_counter, _error_counter
    global _rag_usage_counter, _token_counter, _request_duration_histogram
    global _refusal_counter, _session_counter, _messages_per_session_histogram
    global _emotion_counter, _tracer, _setup_done, _language_counter

    if _setup_done:
        return
    _setup_done = True

    # ── Resource (identifies this service in Axiom) ──
    resource = Resource.create({
        "service.name": service_name or os.getenv("OTEL_SERVICE_NAME", "mental-health-rag-api"),
        "service.version": "1.0.0",
        "deployment.environment": os.getenv("DEPLOYMENT_ENV", "development"),
    })

    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

    # ── TRACING ──
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        )
    )
    trace.set_tracer_provider(tracer_provider)
    _tracer = trace.get_tracer("mental-health-rag-api", "1.0.0")

    # ── METRICS ──
    metric_reader = PeriodicExportingMetricReader(
        exporter=OTLPMetricExporter(
            endpoint=otlp_endpoint,
            insecure=True,
            preferred_temporality={
    Counter: AggregationTemporality.CUMULATIVE,
    Histogram: AggregationTemporality.DELTA,
},
        ),
        export_interval_millis=int(os.getenv("OTEL_METRIC_EXPORT_INTERVAL_MS", "15000")),
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    meter = metrics.get_meter("mental-health-rag-api", "1.0.0")

    # ──────────────────────────────────────────────
    # 1. MODEL / NLP METRICS
    # ──────────────────────────────────────────────

    _intent_counter = meter.create_counter(
        name="mental_health_intent_total",
        description="Count of detected user intents, broken down by intent type",
        unit="1",
    )

    _llm_duration_histogram = meter.create_histogram(
        name="llm_response_duration_seconds",
        description="Wall-clock duration of each pipeline step (LLM call, retrieval, language/emotion detection)",
        unit="s",
    )

    _rag_usage_counter = meter.create_counter(
        name="rag_usage_total",
        description="Count of requests by whether RAG retrieval was triggered",
        unit="1",
    )

    _token_counter = meter.create_counter(
        name="llm_tokens_total",
        description="Token usage by step and token type (input/output/total)",
        unit="1",
    )

    _refusal_counter = meter.create_counter(
        name="llm_refusals_total",
        description="Count of responses where the LLM refused to answer",
        unit="1",
    )

    _emotion_counter = meter.create_counter(
        name="emotion_total",
        description="Count of detected user emotions per request",
        unit="1",
    )

    _language_counter = meter.create_counter(
        name="language_total",
        description="Count of detected user query languages",
        unit="1",
    )

    # ──────────────────────────────────────────────
    # 2. DATA METRICS
    # ──────────────────────────────────────────────

    _query_length_histogram = meter.create_histogram(
        name="query_length_chars",
        description="Distribution of user query lengths in characters",
        unit="chars",
    )

    _feedback_counter = meter.create_counter(
        name="feedback_votes_total",
        description="Count of user feedback votes (positive / negative)",
        unit="1",
    )

    _session_counter = meter.create_counter(
        name="sessions_total",
        description="Count of new chat sessions created",
        unit="1",
    )

    _messages_per_session_histogram = meter.create_histogram(
        name="messages_per_session",
        description="Number of messages in a session at the time of each request",
        unit="1",
    )

    # ──────────────────────────────────────────────
    # 3. SERVER METRICS
    # ──────────────────────────────────────────────

    _request_counter = meter.create_counter(
        name="http_requests_total",
        description="Total HTTP requests by method, endpoint, and status code",
        unit="1",
    )

    _error_counter = meter.create_counter(
        name="http_errors_total",
        description="Total HTTP errors (4xx / 5xx) by method, endpoint, status code, and error_type (client/server)",
        unit="1",
    )

    _request_duration_histogram = meter.create_histogram(
        name="request_duration_seconds",
        description="End-to-end duration of an API request",
        unit="s",
    )

    # Uptime is an observable gauge – automatically sampled every export interval
    def _observe_uptime(options):
        yield metrics.Observation(time.time() - _start_time, {})

    meter.create_observable_gauge(
        name="process_uptime_seconds",
        description="Seconds since the application process started",
        unit="s",
        callbacks=[_observe_uptime],
    )

    print(f"✅ OpenTelemetry initialised → endpoint={otlp_endpoint}")


# ════════════════════════════════════════════════
#  PUBLIC HELPER FUNCTIONS
#  (safe to call even if setup_telemetry was never
#   invoked — they silently no-op.)
# ════════════════════════════════════════════════
def record_language(language: str):
    """MODEL — record the detected language label for a request."""
    if _language_counter is not None:
        _language_counter.add(1, {"language": language})

def record_intent(intent: str):
    """MODEL — increment the intent counter for the given intent label."""
    if _intent_counter is not None:
        _intent_counter.add(1, {"intent": intent})


def record_llm_duration(identifier: str, duration_s: float):
    """MODEL — record how long a pipeline step took."""
    if _llm_duration_histogram is not None:
        _llm_duration_histogram.record(duration_s, {"llm_identifier": identifier})


def record_rag_usage(used: bool):
    """MODEL — record whether a request triggered RAG retrieval."""
    if _rag_usage_counter is not None:
        _rag_usage_counter.add(1, {"used": str(used).lower()})


def record_tokens(step: str, input_tokens: int, output_tokens: int, total_tokens: int):
    """MODEL — record token usage for a given pipeline step."""
    if _token_counter is None:
        return
    _token_counter.add(input_tokens,  {"step": step, "token_type": "input"})
    _token_counter.add(output_tokens, {"step": step, "token_type": "output"})
    _token_counter.add(total_tokens,  {"step": step, "token_type": "total"})


def check_and_record_refusal(response_text: str, intent: str) -> bool:
    """MODEL — detect and record an LLM refusal. Returns True if detected."""
    if not response_text:
        return False
    is_refusal = any(p in response_text.lower() for p in _REFUSAL_PHRASES)
    if is_refusal and _refusal_counter is not None:
        _refusal_counter.add(1, {"intent": intent})
    return is_refusal


def record_emotion(emotion: str):
    """MODEL — record the detected emotion label for a request."""
    if _emotion_counter is not None:
        _emotion_counter.add(1, {"emotion": emotion})


def record_query_length(length: int, language: str = "unknown"):
    """DATA — record the character length of a user query."""
    if _query_length_histogram is not None:
        _query_length_histogram.record(length, {"detected_language": language})


def record_feedback(vote: str):
    """DATA — record a user feedback vote ('positive' or 'negative')."""
    if _feedback_counter is not None:
        _feedback_counter.add(1, {"vote_type": vote})


def record_session_created():
    """DATA — record creation of a new chat session."""
    if _session_counter is not None:
        _session_counter.add(1, {})


def record_messages_per_session(count: int):
    """DATA — record the number of messages in a session at request time."""
    if _messages_per_session_histogram is not None:
        _messages_per_session_histogram.record(count, {})


def record_http_request(method: str, endpoint: str, status_code: int):
    """SERVER — count an HTTP request; also count as error (client/server) if status >= 400."""
    if _request_counter is not None:
        _request_counter.add(1, {
            "http.method": method,
            "http.endpoint": endpoint,
            "http.status_code": str(status_code),
        })
    if status_code >= 400 and _error_counter is not None:
        error_type = "server" if status_code >= 500 else "client"
        _error_counter.add(1, {
            "http.method": method,
            "http.endpoint": endpoint,
            "http.status_code": str(status_code),
            "error_type": error_type,
        })


def record_request_duration(endpoint: str, duration_s: float):
    """SERVER — record end-to-end request duration."""
    if _request_duration_histogram is not None:
        _request_duration_histogram.record(duration_s, {"endpoint": endpoint})


def get_tracer():
    """Return the OTel tracer (or None if telemetry is not set up)."""
    return _tracer