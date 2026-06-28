---
tags: [mlops, pipelines, core]
status: growing
created: 2026-06-27
---

# ML Pipelines

## 1. Why This Matters

A model in a Jupyter notebook is not a deployed system. The gap between "works on my machine" and "works reliably in production" is filled by **ML pipelines**: automated, repeatable, and monitored workflows that transform raw data into predictions.

Pipelines ensure that every step — from data ingestion to model deployment — is automated, tested, and auditable. Without them, ML projects die in production.

---

## 2. Pipeline Stages

```
Data Ingestion → Validation → Transformation → Training → Evaluation → Deploy
```

### 2.1 Data Ingestion

Getting data from source systems into the pipeline.

| Pattern | Latency | Tools |
|---|---|---|
| **Batch** | Hourly/Daily | Airflow, cron, scheduled jobs |
| **Streaming** | Real-time (seconds) | Kafka, Kinesis, Flink |
| **Trigger-based** | On data arrival | Cloud functions, webhooks |

**Best practices**:
- Store raw data immutably (never modify the original)
- Log data source, timestamp, and version
- Monitor for missing data or delayed arrivals

### 2.2 Data Validation

Check data quality before it enters the pipeline.

```
✓ Schema matches expected columns and types
✓ Value ranges are within expected bounds
✓ Missing rate is below threshold
✓ No duplicate primary keys
```

**Tools**: Great Expectations, Pandera, TensorFlow Data Validation. Monitor for [[Data & Concept Drift]] after deployment.

**What happens when validation fails**:
- **Warning**: log and continue (minor issues)
- **Block**: stop the pipeline (critical issues)
- **Alert**: notify the team

### 2.3 Transformation (Feature Engineering)

Convert raw data into model-ready features.

Always build a **single pipeline** for both training and serving to prevent train/serve skew:

```python
# BAD: separate code for train and serve
train_features = scale(X_train)  # different code path
serve_features = scale(X_serve)  # could diverge over time

# GOOD: shared pipeline
pipeline = StandardScaler()
pipeline.fit(X_train)              # fit on train
train_features = pipeline.transform(X_train)
serve_features = pipeline.transform(X_serve)  # same logic
```

### 2.4 Training

- Track every experiment ([[Experiment Tracking]])
- Log hyperparameters, metrics, and artifacts
- Version the training data and code
- Tune and log hyperparameters ([[Hyperparameter Tuning]])
- Reproduce any previous result

### 2.5 Evaluation

- Compare candidate vs champion (current production model)
- Evaluate on multiple metrics (not just accuracy)
- Test on held-out data and data slices (see [[Model Evaluation]])
- Automatically promote if candidate beats champion

### 2.6 Deployment

| Strategy | Description | When to Use |
|---|---|---|
| **Shadow** | New model runs in parallel with production, no user impact | Testing reliability |
| **Canary** | Route small % of traffic to new model | Gradual rollout |
| **Blue/Green** | Instant switch between old and new | Low-risk deployments |
| **Rolling** | Gradually replace instances of old model | Zero-downtime updates |

Validate model changes with [[A-B Testing]] before full rollout.

---

## 3. Feature Stores

A **feature store** (Feast, Tecton) solves a common problem: the same feature computed differently in training vs serving.

**What it provides**:
- **Single definition**: feature logic defined once, used everywhere
- **Online serving**: low-latency feature retrieval (Redis, DynamoDB)
- **Offline serving**: batch feature computation for training (S3, BigQuery)
- **Point-in-time correctness**: features are computed as they were at the prediction time (preventing data leakage)

```python
# Feast example
features = feature_store.get_online_features(
    features=["user:age", "user:total_purchases", "item:category"],
    entity_rows=[{"user_id": 123, "item_id": 456}]
).to_dict()
```

---

## 4. Orchestration

Orchestration tools schedule, monitor, and retry pipeline steps.

| Tool | Key Features |
|---|---|
| **Airflow** | DAG-based, mature, large ecosystem |
| **Prefect** | Python-native, better error handling |
| **Dagster** | Data-aware, asset-focused |
| **Kubeflow** | Kubernetes-native, ML-specific |
| **Flyte** | Type-safe, ML-focused |

All of them model pipelines as **DAGs** (directed acyclic graphs) — steps with dependencies that can run in parallel where possible.

---

## 5. Common Mistakes

1. **Train/serve skew**: different feature engineering code in training and serving. Always use a single pipeline.

2. **Not versioning data**: you cannot reproduce a model without knowing which data version it was trained on. Use DVC or similar.

3. **Manual deployment steps**: "someone runs a script" is not a deployment strategy. Automate everything.

4. **No monitoring between pipeline runs**: a silent failure (data stopped arriving) can go undetected for days. Monitor data freshness.

5. **Ignoring dependencies**: feature transformations often depend on reference data (lookup tables). Version these too.

---

## 6. Check Your Understanding

1. Why should training and serving use the same feature engineering code? (Prevents train/serve skew — differences that degrade serving performance.)

2. A pipeline step fails at 3 AM. What should happen? (Alert the team, retry if transient, block the pipeline if critical.)

3. What is the difference between a feature store and a regular database? (Feature store handles point-in-time correctness, online + offline serving, and feature sharing across teams.)

4. Your model was trained on data with a "total_spent" feature. In production, this feature is computed differently. What problem do you expect? (Train/serve skew — the model may see different distributions than expected.)

5. You deploy a new model via canary deployment. What fraction of traffic do you start with? (Typically 1-5%, then gradually increase while monitoring metrics.)

---

## 7. Summary

ML pipelines automate the end-to-end ML workflow: ingest → validate → transform → train → evaluate → deploy. The key principles are: share feature code between training and serving, version everything (data, code, model), automate all deployment steps, and monitor for failures. A well-built pipeline makes model updates safe, fast, and auditable.

---

## 8. Where to Go Next

- [[Experiment Tracking]] — Logging experiments within pipelines
- [[Model Monitoring]] — Monitoring deployed models
- [[Feature Engineering]] — Features that flow through the pipeline
