---
tags: [mlops, observability, advanced]
status: growing
created: 2026-06-27
---

# Observability

## 1. Why This Matters

A model is running in production. Is it working correctly? How do you know?

Observability is the practice of making a system's internal state inferable from its external outputs. In ML, this means knowing not just "is the model responding?" but "is it responding correctly? Is it degrading slowly? Is it serving the right users with the right latency?"

Without observability, you are flying blind. With it, you can detect, diagnose, and fix problems before users notice.

---

## 2. The Three Pillars

### 2.1 Logging

Structured, searchable records of events.

**What to log**:
```json
{
  "timestamp": "2026-06-27T14:30:00Z",
  "request_id": "req_abc123",
  "model_version": "v2.3.1",
  "latency_ms": 145,
  "prediction": 0.87,
  "confidence": 0.92,
  "features_hash": "a1b2c3d4",
  "error": null
}
```

**Best practices**:
- **Structured JSON** (not plain text) — queryable with tools like jq, Loki, Elasticsearch
- **Correlation ID** — trace a request across all services
- **Log levels**: DEBUG (dev), INFO (normal), WARN (potential issue), ERROR (failure)
- **Never log PII** — mask or omit personal data
- **Link logs to code versions** — use [[Git]] commit hashes in log metadata

### 2.2 Metrics

Numerical measurements collected over time.

| Metric Type | Examples | What It Tells You |
|---|---|---|
| **Counters** | Total requests, total errors | Volume (increases monotonically) |
| **Gauges** | Current memory use, queue depth | Current state (up/down) |
| **Histograms** | Latency (p50, p95, p99), prediction values | Distribution over time |
| **Rates** | Requests/second, error rate, throughput | Velocity (derivative of counter) |

**The USE method**: for every resource, monitor:
- **U**tilization: how busy is it?
- **S**aturation: how much backlog?
- **E**rrors: how many failures?

**The RED method**: for every request, monitor:
- **R**ate: requests per second
- **E**rrors: failed requests per second
- **D**uration: latency distribution

### 2.3 Tracing

Follow a single request across multiple services.

```
Gateway → Auth → Feature Store → Model Server → Response
  2ms      5ms       12ms            45ms         3ms
```

**Why tracing matters for ML**: a slow prediction may not be the model's fault — it could be the feature store, the embedding service, or the cache miss. Tracing tells you where the time went.

**OpenTelemetry**: the standard for distributed tracing. Instrument once, export to any backend.

---

## 3. ML-Specific Observability

Combine with [[Experiment Tracking]] to correlate prediction behavior with specific training runs.

### 3.1 Prediction Monitoring

Beyond system health, monitor the **quality and behavior** of predictions:

| What to Monitor | How |
|---|---|
| **Prediction distribution** | Histogram of scores over time |
| **Confidence scores** | Are predictions becoming less certain? |
| **Feature values** | Distribution of each input feature |
| **Model version** | Which version served each prediction |
| **Error cases** | What inputs cause the model to fail |

### 3.2 Data Pipeline Monitoring

| What | Alert When |
|---|---|
| **Data freshness** | Last successful pipeline run > 2 hours ago |
| **Row counts** | Significantly different from expected |
| **Null rates** | Sudden spike in missing values |
| **Schema violations** | Unexpected column types or names |

---

## 4. Alerting Strategy

### 4.1 Alert Design

A good alert:
- **Actionable**: someone can do something about it
- **Timely**: early enough to prevent impact
- **Specific**: tells you what is wrong
- **Noiseless**: does not fire unnecessarily

Set up alerting for [[A-B Testing]] experiments to catch regressions early.

### 4.2 Severity Levels

| Level | Response | Example |
|---|---|---|
| **Page (P0)** | Immediate (5 min) | Model returning 50% errors |
| **Ticket (P1)** | Within 1 hour | Latency P95 exceeds threshold |
| **Slack (P2)** | Same day | Gradual accuracy decline |
| **Dashboard (P3)** | Review weekly | Minor feature drift |

---

## 5. Tools

| Domain | Tool |
|---|---|
| **Metrics** | Prometheus + Grafana |
| **Logging** | ELK (Elasticsearch, Logstash, Kibana), Loki |
| **Tracing** | Jaeger, OpenTelemetry |
| **ML-specific** | WhyLabs, Evidently, Arize |

---

## 6. Common Mistakes

1. **Logging everything**: too much data is as bad as too little. Log what is actionable. Archive old logs.

2. **No correlation IDs**: without a request ID, you cannot connect logs, metrics, and traces for the same event.

3. **Alert fatigue**: too many alerts → alerts are ignored. Every alert should require a specific action.

4. **Monitoring code but not data**: the code runs fine, but the data is wrong. Monitor data quality separately from system health.

5. **No dashboards**: a Grafana dashboard that is never looked at is wasted effort. Build dashboards for specific audiences (on-call, team lead, business).

---

## 7. Check Your Understanding

1. A user reports a slow response. How do you diagnose it? (Find the request ID → trace through services → identify the bottleneck.)

2. What is the difference between a counter and a gauge? (Counter: monotonically increasing (total requests). Gauge: fluctuating (current memory).)

3. Why log in structured JSON instead of plain text? (Queryable — you can search for specific fields without parsing.)

4. Your alert "P99 latency > 1s" fires at 3 AM. The on-call engineer checks and finds a one-time spike caused by a batch job. What is wrong? (The alert is too sensitive — add a duration window or exclude batch times.)

5. A model's accuracy drops but all system metrics (latency, error rate) are normal. What is missing? (ML-specific monitoring — prediction quality is not captured by system metrics.)

---

## 8. Summary

Observability makes production ML systems understandable. Logging records events, metrics measure trends, and traces follow requests. ML-specific observability adds prediction and data quality monitoring. Alerts should be actionable, timely, and specific. The three pillars together let you detect, diagnose, and fix problems quickly.

---

## 9. Where to Go Next

- [[Model Monitoring]] — ML-specific monitoring in depth
- [[Data & Concept Drift]] — Data quality monitoring
- [[ML Pipelines]] — Observability for pipeline failures
