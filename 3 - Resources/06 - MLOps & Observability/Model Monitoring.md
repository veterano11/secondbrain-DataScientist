---
tags: [mlops, monitoring, advanced]
status: growing
created: 2026-06-27
---

# Model Monitoring

## 1. Why This Matters

A model that works perfectly on day 1 can fail on day 30. Data distributions shift. User behavior changes. External events (holidays, economic changes, competitor actions) make training data obsolete.

Model monitoring detects these changes before they cause business impact. It is the difference between finding out on Monday morning that your fraud detection model has been missing attacks all weekend, and catching it in real-time.

---

## 2. The Three Pillars of ML Monitoring

### 2.1 Data Drift

Input distribution $P(X)$ changes → the model sees unfamiliar patterns.

**Example**: a model trained on pre-pandemic shopping behavior sees entirely different purchase patterns during a lockdown.

See [[Data & Concept Drift]] for full details.

### 2.2 Concept Drift

The relationship $P(y|X)$ changes → the model's mapping is stale.

**Example**: a spam filter trained in 2023 may not recognize 2024's sophisticated phishing emails — the features that once indicated spam no longer do.

### 2.3 System Degradation

Operational problems: latency spikes, memory leaks, service outages.

**Example**: a feature extraction service slows down, increasing prediction latency from 50ms to 5 seconds. Users time out.

---

## 3. What to Monitor

### 3.1 Prediction Quality

Requires ground truth (may be delayed):

| Metric | How to Measure |
|---|---|
| **Accuracy / RMSE** | Compare predictions against actual outcomes |
| **Residual analysis** | Plot prediction errors over time |
| **Calibration** | For probabilistic models: do 90% confidence predictions match reality? |
| **Model confidence** | Are predictions becoming less certain? |

For deeper evaluation techniques, see [[Model Evaluation]].

### 3.2 Data Quality

| Check | Method |
|---|---|
| **Missing rate** | % of nulls per feature over time |
| **Range violations** | Values outside expected bounds |
| **Type violations** | Unexpected data types |
| **Cardinality changes** | New categories in categorical features |

### 3.3 Prediction Distribution

| Check | What It Detects |
|---|---|
| **Mean prediction** | Shift in overall model output |
| **Class proportions** | For classification: are predictions balanced? |
| **Score distribution** | Are probabilities concentrated or spread out? |
| **Prediction velocity** | Rate of change of predictions over time |

These checks ensure [[Feature Engineering]] produces consistent inputs for the model.

### 3.4 System Health

| Metric | Alert When |
|---|---|
| **p50/p95/p99 latency** | P99 exceeds threshold |
| **Error rate** | >X% of predictions return errors |
| **Throughput** | Drops below expected volume |
| **Memory/CPU** | Approaches resource limits |
| **Data freshness** | Last successful pipeline run |



---

## 4. Detection Methods

### 4.1 Statistical Tests

| Test | What It Compares | Good For |
|---|---|---|
| **PSI** | Distribution of binned reference vs production | General drift detection |
| **KS test** | Distribution of two continuous samples | Continuous features |
| **Jensen-Shannon** | Symmetric KL divergence | Comparing two distributions |
| **Z-score** | Current metric vs historical mean | Simple threshold-based |

### 4.2 Window-Based Monitoring

Compare a **reference window** (e.g., training data or last 30 days) with a **current window** (e.g., last hour or last day):

```python
def detect_drift(reference: np.array, current: np.array, threshold: float = 0.1):
    psi = compute_psi(reference, current)
    return psi > threshold  # True if drift detected
```

**Window sizes**:
- Small window (1 hour): sensitive to recent changes, noisy
- Large window (7 days): smoother, slower to react
- Multi-window: compare against multiple windows (1h, 24h, 7d) for robust detection

### 4.3 Adaptive Thresholds

Static thresholds become stale as data evolves. Adaptive methods:

- **Rolling statistics**: mean ± 3σ over a sliding window
- **EWMA**: exponentially weighted moving average (reacts faster to recent changes)
- **Seasonal decomposition**: account for weekly/daily patterns before detecting drift

---

## 5. Alerting and Response

### 5.1 Severity Levels

| Level | Response | Example |
|---|---|---|
| **P0 (Critical)** | Immediate (within 5 min) | Model returning errors, data pipeline down |
| **P1 (High)** | Within 30 min | Prediction latency > 5s, accuracy drop > 5% |
| **P2 (Medium)** | Within 4 hours | Gradual drift detected, data freshness delay |
| **P3 (Low)** | Next business day | Minor distribution shift, single metric fluctuation |

### 5.2 Runbooks

Every alert should have a documented response:

```
Alert: Model accuracy dropped by 10%
1. Check if ground truth data is complete (maybe labels are delayed?)
2. Compare current predictions vs last week's distribution
3. Check recent pipeline runs for errors
4. If drift confirmed: trigger retraining pipeline
5. If retraining takes > 4 hours: roll back to previous model version
6. Validate the fix with [[A-B Testing]] before re-promoting
```

---

## 6. Tools

| Tool | Focus |
|---|---|
| **WhyLabs / Whylogs** | Data and ML monitoring |
| **Evidently** | Drift detection and model evaluation |
| **Arize AI** | Production monitoring and observability |
| **Grafana + Prometheus** | System metrics and alerting |
| **MLflow** | [[Experiment Tracking]] + model registry |

---

## 7. Common Mistakes

1. **Monitoring accuracy without latency**: a model that takes 10 seconds is useless regardless of accuracy. Monitor both.

2. **Alerts without runbooks**: "Model accuracy dropped" with no documented response leads to panic. Write runbooks in advance.

3. **Not accounting for delays in ground truth**: accuracy monitoring is only as timely as the label feedback loop. For delayed labels, use proxy metrics (prediction distribution, confidence).

4. **Threshold fishing**: adjusting thresholds reactively to reduce alerts creates fragile monitoring. Set thresholds based on historical analysis.

5. **Not monitoring data quality**: "garbage in, garbage out" — if input data quality degrades, the model will too. Monitor data quality at the pipeline stage, not just model output.

---

## 8. Check Your Understanding

1. Your model's accuracy dropped from 92% to 85% overnight. What do you check first? (Ground truth data completeness, data drift, pipeline health.)

2. Why monitor prediction distribution even before ground truth is available? (Distribution changes are an early warning signal before labels arrive.)

3. A P1 alert fires at 3 AM. The on-call engineer has 30 minutes to respond. What should be in the runbook? (Step-by-step diagnosis instructions.)

4. Your KS test detects drift on a feature that was not important to the model. Should you alert? (Probably not — focus drift detection on important features.)

5. What is the difference between PSI and KS test? (PSI discretizes data into bins first, KS test works on continuous CDFs. Both measure distribution shift.)

---

## 9. Summary

Model monitoring detects when production models degrade. The three pillars are data drift, concept drift, and system health. Monitor prediction quality (requires ground truth), prediction distribution (early warning), data quality (prevent garbage-in), and system health (latency, errors). Use statistical tests with adaptive thresholds. Alert with severity levels and documented runbooks.

---

## 10. Where to Go Next

- [[Data & Concept Drift]] — Detailed drift detection methods
- [[ML Pipelines]] — Building pipelines that support monitoring
- [[Observability]] — Logging, metrics, and tracing for ML
