# Rex Trading System - End-to-End Observability

## Overview

The Rex Trading System implements comprehensive end-to-end observability with:

- **🔍 Distributed Tracing** - Full request tracing across A2A agents using OpenTelemetry
- **📊 Structured Logging** - JSON-formatted logs with trace correlation
- **❌ Error Tracking** - Automatic error aggregation and alerting
- **📈 Metrics Collection** - Performance and business metrics
- **🎯 Context Propagation** - Trace context across agent boundaries

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

```bash
# Optional: Configure external trace collection
export OTEL_EXPORTER_OTLP_ENDPOINT="https://your-otel-collector:4317"
export OTEL_EXPORTER_OTLP_HEADERS="api-key=your-api-key"

# Or for local development with Jaeger
export JAEGER_ENDPOINT="http://localhost:14268/api/traces"
export JAEGER_HOST="localhost"
export JAEGER_PORT="6831"

# Logging configuration
export ENVIRONMENT="production"  # or "development"
```

### 3. Start the Application

```bash
python app.py
```

Visit the observability dashboard at: **http://localhost:5000/observability/dashboard.html**

## Core Components

### 1. Distributed Tracing (`tracer.py`)

Provides OpenTelemetry-based tracing:

```python
from src.rex.observability import get_tracer, trace_context

tracer = get_tracer("my-service")

# Context manager
with trace_context("operation_name") as span:
    span.set_attribute("key", "value")
    # Your code here

# Decorator
@tracer.trace_function("custom_name")
def my_function():
    return "result"
```

**Features:**
- Automatic HTTP instrumentation
- Cross-service trace propagation
- OTLP and Jaeger export support
- Span attributes and status tracking

### 2. Structured Logging (`logger.py`)

JSON-formatted logging with trace correlation:

```python
from src.rex.observability import get_logger

logger = get_logger("my-component")

logger.info("Operation completed", extra={
    'user_id': 123,
    'operation_type': 'data_load'
})

logger.error("Operation failed", error=exception, extra={
    'context': 'additional_info'
})
```

**Features:**
- Automatic trace ID injection
- Structured JSON output
- Error with stack traces
- Multiple output handlers

### 3. Error Tracking (`error_tracker.py`)

Automatic error aggregation and alerting:

```python
from src.rex.observability import track_error, ErrorSeverity, ErrorCategory

# Manual tracking
error_id = track_error(
    exception,
    severity=ErrorSeverity.HIGH,
    category=ErrorCategory.DATABASE_ERROR
)

# Decorator for automatic tracking
@track_errors(severity=ErrorSeverity.MEDIUM)
def risky_function():
    # Function code here
    pass
```

**Features:**
- Error fingerprinting for deduplication
- Severity and category classification
- Alert thresholds and callbacks
- Error trend analysis

### 4. Metrics Collection (`metrics.py`)

Performance and business metrics:

```python
from src.rex.observability import get_metrics, get_business_metrics

metrics = get_metrics()
business_metrics = get_business_metrics()

# Custom metrics
metrics.counter("api.requests", 1.0, {"endpoint": "/api/data"})
metrics.gauge("system.memory.usage", 85.5)
metrics.histogram("request.duration", 250.0)

# Business metrics
business_metrics.track_trade_execution("BTC-USD", "buy", 0.1, 50000, True)
business_metrics.track_api_request("/api/market", "GET", 200, 150)
```

**Features:**
- Counter, gauge, histogram, and timer metrics
- Tag-based dimensions
- Prometheus export format
- Automatic percentile calculations

### 5. A2A Context Propagation (`context.py`)

Trace context across agent boundaries:

```python
from src.rex.observability import (
    create_trace_context, A2AContextEnhancer, 
    with_trace_context, ObservableWorkflow
)

# Create workflow context
workflow = ObservableWorkflow("bulk_data_load", user_id=123)

# Execute steps with tracing
result = await workflow.execute_step(
    step_name="load_data",
    agent_id="historical-loader-001", 
    operation=load_function,
    symbol="BTC-USD"
)

# A2A message enhancement
message = {"type": "DATA_REQUEST", "payload": {...}}
enhanced_message = A2AContextEnhancer.enhance_message(message)
```

## A2A Agent Integration

### Observable Agent Pattern

```python
from src.rex.observability.integration import (
    ObservableA2AAgent, observable_agent_method
)

class MyAgent(ObservableA2AAgent):
    def __init__(self):
        super().__init__(agent_id="my-agent-001")
    
    @observable_agent_method("my-agent-001", "process_data")
    async def process_data(self, data):
        # Automatically traced and logged
        return {"result": "processed"}
    
    def _process_message_impl(self, message):
        # Handle A2A message with observability
        return {"status": "processed"}
```

### Workflow Orchestration

```python
async def data_processing_workflow():
    workflow = ObservableWorkflow("data_pipeline", user_id=123)
    
    # Step 1: Load data
    data = await workflow.execute_step(
        "load_historical_data",
        "historical-agent-001",
        historical_agent.load_data,
        symbol="BTC-USD"
    )
    
    # Step 2: Process data  
    result = await workflow.execute_step(
        "process_data",
        "processing-agent-001", 
        processing_agent.analyze,
        data=data
    )
    
    return workflow.get_summary()
```

## API Endpoints

### Health Check
```
GET /observability/health
```

### Metrics
```
GET /observability/metrics?hours=24&format=json
GET /observability/metrics?format=prometheus
GET /observability/metrics/<metric_name>?hours=1
```

### Error Tracking
```
GET /observability/errors/summary?hours=24
GET /observability/errors/<error_id>
GET /observability/errors/fingerprint/<fingerprint>?limit=10
```

### Dashboard
```
GET /observability/dashboard?hours=24
GET /observability/dashboard.html
```

## External Integration

### OTLP Export (Production)

Configure OTLP exporter for production monitoring:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="https://api.datadoghq.com/api/v1/otel/traces"
export OTEL_EXPORTER_OTLP_HEADERS="DD-API-KEY=your-datadog-key"
```

### Jaeger (Development)

Run Jaeger locally:

```bash
docker run -d --name jaeger \
  -p 16686:16686 \
  -p 14268:14268 \
  -p 6831:6831/udp \
  jaegertracing/all-in-one:latest

export JAEGER_ENDPOINT="http://localhost:14268/api/traces"
```

Visit Jaeger UI: http://localhost:16686

## Example Usage

See `observability_example.py` for a complete demonstration:

```bash
python observability_example.py
```

This example shows:
- A2A workflow with distributed tracing
- Error tracking and aggregation
- Metrics collection
- Dashboard integration

## Best Practices

### 1. Trace Naming
- Use clear, hierarchical names: `agent.historical-loader.load_data`
- Include operation type: `api.get_market_data`, `workflow.bulk_load`

### 2. Structured Logging
- Use consistent field names: `user_id`, `agent_id`, `operation_type`
- Include context: trace IDs, request IDs, workflow IDs
- Log at appropriate levels: DEBUG for detailed info, ERROR for failures

### 3. Error Handling
- Categorize errors appropriately
- Set severity based on business impact
- Include context in error tracking

### 4. Metrics Strategy
- Use counters for events: `trades.executed`, `errors.total`
- Use gauges for current values: `active_connections`, `queue_size`
- Use histograms for distributions: `request_duration`, `trade_volume`

### 5. Performance Considerations
- Sampling for high-volume traces
- Async logging for performance
- Metric aggregation intervals
- Regular cleanup of old data

## Troubleshooting

### No Traces Appearing
1. Check OTEL configuration: `GET /observability/health`
2. Verify exporter endpoint connectivity
3. Check trace sampling configuration

### Missing Logs
1. Ensure log directory permissions: `mkdir logs && chmod 755 logs`
2. Check log level configuration
3. Verify structured logger setup

### High Memory Usage
1. Adjust retention periods in error tracker
2. Configure metric point limits
3. Enable log rotation

## Configuration Reference

### Environment Variables

```bash
# Service identification
SERVICE_VERSION=1.0.0
ENVIRONMENT=production
HOSTNAME=trading-server-01

# OpenTelemetry
OTEL_EXPORTER_OTLP_ENDPOINT=https://your-collector:4317
OTEL_EXPORTER_OTLP_HEADERS=api-key=secret

# Jaeger (alternative)
JAEGER_ENDPOINT=http://localhost:14268/api/traces
JAEGER_HOST=localhost
JAEGER_PORT=6831

# Metrics export
METRICS_ENDPOINT=https://your-metrics-system/api/metrics

# Development
DISABLE_SSL_VERIFY=true  # Development only
```

### File Configuration

Create `logs/` directory for log files:
```bash
mkdir -p logs
chmod 755 logs
```

## Monitoring Alerts

Example alert conditions:

1. **High Error Rate**: >5% error rate in 5 minutes
2. **Critical Errors**: Any CRITICAL severity error
3. **Slow Operations**: 95th percentile >1000ms
4. **Failed Workflows**: Any workflow failure

Set up alerts using your monitoring system based on the metrics and logs exported by the observability system.