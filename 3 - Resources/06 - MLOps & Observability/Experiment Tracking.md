---
tags: [mlops, experiments, core]
status: growing
created: 2026-06-27
---

# Experiment Tracking

## 1. Why This Matters

"How did we get that 92% accuracy result? What hyperparameters did we use?" If you have ever asked this question, you need experiment tracking.

Experiments in ML are not like experiments in other software. You cannot just look at the code — the same code with different random seeds, training data versions, or hyperparameters produces different results. Experiment tracking logs **everything** needed to reproduce any result.

---

## 2. What to Track

### 2.1 The Four Categories

| Category | Examples |
|---|---|
| **Code** | Git commit hash, branch, diff from main |
| **Data** | Dataset path, version, hash, schema |
| **Hyperparameters** | Learning rate, batch size, model architecture, optimizer (see [[Hyperparameter Tuning]]) |
| **Metrics** | Loss, accuracy, F1, AUC (training + validation) (see [[Model Evaluation]]) |
| **Artifacts** | Model weights, predictions, plots, confusion matrices |
| **Environment** | Python version, GPU type, CUDA version, installed packages |

### 2.2 The Golden Rule

If you cannot reproduce it, you did not track it.

---

## 3. MLflow

### 3.1 Core Concepts

- **Experiment**: a logical grouping of runs (e.g., "bert-fine-tuning")
- **Run**: a single execution (e.g., "lr=3e-5, batch=32, epoch=3")
- **Parameters**: key-value pairs (hyperparameters)
- **Metrics**: key-value pairs that can be updated over time (loss per epoch)
- **Artifacts**: files (model weights, plots, data samples)

### 3.2 Basic Usage

```python
import mlflow

# Set experiment
mlflow.set_experiment("churn-prediction")

# Start a run
with mlflow.start_run(run_name="random-forest-v3"):
    # Log parameters
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("learning_rate", 0.01)

    # Train model
    model = RandomForestClassifier(n_estimators=200, max_depth=10)
    model.fit(X_train, y_train)

    # Log metrics
    accuracy = model.score(X_val, y_val)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1", f1_score(y_val, model.predict(X_val)))

    # Log model
    mlflow.sklearn.log_model(model, "model")

    # Log artifacts
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact("feature_importances.png")
```

### 3.3 Viewing Results

```
mlflow ui  # starts the tracking UI at http://localhost:5000
```

The UI lets you:
- Compare runs side-by-side (parameters, metrics)
- Filter by parameters or metrics
- Download artifacts
- Register best models

---

## 4. Weights & Biases

W&B is a cloud-based (or self-hosted) alternative with richer dashboards:

```python
import wandb

wandb.init(project="churn-prediction", config={
    "n_estimators": 200,
    "max_depth": 10,
})

model = RandomForestClassifier(**wandb.config)
model.fit(X_train, y_train)

wandb.log({"accuracy": accuracy, "f1": f1})
wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(y_val, preds)})
```

**Advantages**: collaborative dashboards, automatic hardware monitoring, report creation.

**Disadvantages**: cloud dependency (self-hosted available), can be expensive at scale.

---

## 5. Model Registry

### 5.1 Purpose

Track which model is in production and manage model versions.

```
Development → Staging → Production → Archived
```

Each stage transition requires:
- Validation criteria (accuracy threshold, fairness check)
- Approval (manual or automated)
- Audit log (who promoted, when, why)

### 5.2 MLflow Model Registry

```bash
# Register a model
mlflow.register_model("runs:/<run_id>/model", "churn-classifier")

# Promote to staging
mlflow.models.transition_model_version_stage(
    name="churn-classifier",
    version=2,
    stage="Staging"
)
```

---

## 6. Best Practices

1. **Log every run**: even "failed" runs contain valuable information (what hyperparameters did not work).

2. **Log baselines**: always a simple baseline (mean prediction, linear model) for comparison. Combine with [[A-B Testing]] to validate improvements online.

3. **Tag experiments**: maintain hypotheses as tags so you can search by intent. Log [[Data & Concept Drift]] signals alongside experiment data.

4. **Reproduce from tracked data**: a run should be fully reproducible using only the tracked parameters.

5. **Automate logging**: use callbacks or decorators so logging happens automatically, not manually.

---

## 7. Common Mistakes

1. **Not logging the code version**: a run with no git commit hash is irreproducible.

2. **Overwriting previous results**: always log a new run, never update an existing one.

3. **Logging too many metrics for each run**: human review is hard when each run has 50+ metrics. Track the most important ones.

4. **Not logging failed runs**: failed runs tell you what does not work. Log them with a "status" tag.

5. **Manual logging**: "I'll track the results manually in a spreadsheet" — you will not, and the spreadsheet will be lost.

---

## 8. Check Your Understanding

1. A coworker asks "how did you get that 92% accuracy?" What information do you need to answer? (Git commit, dataset version, hyperparameters, random seed.)

2. Why log baselines? (To know if your complex model is actually better than a simple one.)

3. MLflow tracks parameters and metrics. What is the difference? (Parameters are set before training — they do not change. Metrics are computed during/after training.)

4. You have 100 runs logged. How do you find the best one? (Filter by validation metric, compare parameters, check artifacts for overfitting signs.)

5. A model is in production. A newer version fails validation. What happens? (It stays in staging — only validated models are promoted.)

---

## 9. Summary

Experiment tracking ensures reproducibility. Log everything: code version, data version, hyperparameters, metrics, and artifacts. Use MLflow (self-hosted, open source) or W&B (cloud, richer UI). Maintain a model registry with stage transitions. The rule: if it is not logged, it did not happen.

---

## 10. Where to Go Next

- [[ML Pipelines]] — Integrating tracking into automated pipelines
- [[Model Monitoring]] — Monitoring deployed models
- [[Git]] — Version control for code and data
