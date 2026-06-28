---
tags: [mlops, drift, advanced]
status: growing
created: 2026-06-27
---

# Data & Concept Drift

## 1. Why This Matters

No data distribution stays static. Customer behavior evolves. Markets shift. Seasons change. A model trained on last year's data will inevitably face different patterns today. Understanding drift — when and why it happens — is essential for keeping models reliable in production.

---

## 2. Data Drift (Covariate Shift)

### 2.1 Definition

The input distribution changes: $P_{\text{train}}(X) \neq P_{\text{prod}}(X)$, but $P(y|X)$ stays the same.

**Example**: a fraud detection model trained on transaction data from 2023 sees transactions from 2024. Average transaction amounts increase by 15% due to inflation. The model has never seen this distribution of amounts.

### 2.2 Common Causes

| Cause | Example |
|---|---|
| **Seasonality** | Holiday shopping patterns, weather-dependent behavior |
| **User evolution** | Users become more sophisticated over time |
| **External events** | COVID, economic recession, competitor changes |
| **Data pipeline changes** | New sensor, new logging format, upstream data changes |
| **Sampling bias** | Training data was collected differently than production data |

### 2.3 Detection

**Population Stability Index (PSI)**:

$$\text{PSI} = \sum_i (p_i - q_i) \cdot \ln\left(\frac{p_i}{q_i}\right)$$

- $p_i$: proportion in production bin $i$
- $q_i$: proportion in reference (training) bin $i$
- PSI < 0.1: no significant change
- PSI 0.1-0.2: moderate change (investigate)
- PSI > 0.2: significant drift (take action)

**KS Test**: compare the cumulative distribution of reference and production data for continuous features.

**Wasserstein Distance**: measures the "work" needed to transform one distribution into another. More sensitive than PSI for some patterns.

### 2.4 Mitigation

- **Retrain**: periodically retrain on fresh data
- **Adaptive model**: use online learning to update continuously
- **Robust features**: engineer features that are stable across time (see [[Feature Engineering]])
- **Monitor**: detect drift early and alert the team

---

## 3. Concept Drift

### 3.1 Definition

The relationship between inputs and target changes: $P_{\text{train}}(y|X) \neq P_{\text{prod}}(y|X)$.

**Example**: a credit risk model trained when interest rates were low. Now rates are high. The same income and credit score profile that was "low risk" before is now "medium risk" because higher rates increase default probability.

### 3.2 Types of Concept Drift

```
Sudden:    ▁▁▁▁▁▁▁███   (fraud rule change overnight)
Gradual:   ▁▁▁▁▂▃▄▅▆▇█   (user preferences evolve slowly)
Recurring: ▁▁▃▁▁▃▁▁▃▁▁▃  (seasonal patterns repeat)
```

### 3.3 Detection

Concept drift is harder to detect than data drift because it requires ground truth.

**Methods**:
- **Monitor prediction errors**: when errors increase, concept drift is likely
- **Monitor residual distribution**: if errors become biased (e.g., always overpredicting)
- **Monitor model confidence**: if the model becomes less certain on predictions
- **Monitor feature importance**: if important features change over time (SHAP-based detection)

---

## 4. Local vs Global Drift

**Global drift**: the entire population shifts. Easy to detect (large sample size).

**Local drift**: only a segment of the population shifts. Harder to detect (smaller sample size, but potentially catastrophic for that segment).

**Example**: a medical diagnosis model performs well overall but starts failing for elderly patients (a small segment). Global metrics look fine, but the model is harmful for a specific group.

**Detection**: monitor performance by segments — age groups, regions, customer tiers. Use [[Model Evaluation]] to assess each segment independently.

---

## 5. Retraining Strategies

| Strategy | How It Works | Pros | Cons |
|---|---|---|---|
| **Scheduled** | Retrain every N days/weeks | Simple, predictable | May be too slow for sudden drift |
| **Performance-triggered** | Retrain when accuracy drops | Reacts to real degradation | Requires ground truth |
| **Drift-triggered** | Retrain when drift is detected | Proactive | Does not guarantee performance improvement |
| **Online learning** | Update model incrementally | Always current | Complex, risk of instability |

In practice, most teams use a combination: scheduled retraining as a baseline with drift-triggered retraining for rapid response. Track retraining runs with [[Experiment Tracking]] to compare performance across versions.

---

## 6. Common Mistakes

1. **Confusing data drift with concept drift**: they require different responses. Data drift → retrain on new data. Concept drift → may need a fundamentally different model.

2. **Setting thresholds too tight**: every minor fluctuation triggers alerts. Set thresholds based on business impact, not statistical significance.

3. **Ignoring label delay**: if ground truth takes 30 days to arrive, accuracy-based drift detection is always 30 days behind. Use proxy metrics (prediction distribution) for early warning.

4. **Not monitoring local drift**: global metrics can look fine while a critical segment degrades. Segment your monitoring.

5. **Retraining without validation**: automatically retraining on drifted data can amplify the drift if the most recent data is noisy. Always validate on a clean holdout set.

---

## 7. Check Your Understanding

1. A model trained on summer data predicts ice cream sales in winter. What kind of drift is this? (Data drift — the input features (temperature, daylight hours) are different.)

2. A spam filter trained in 2023 misses 2024 spam because spammers changed tactics. What kind of drift? (Concept drift — the relationship between email features and spam changed.)

3. Your model's accuracy dropped by 10% but prediction distribution is unchanged. What happened? (Concept drift — input patterns are the same but the label relationship changed.)

4. You monitor PSI daily and get alerts every Monday because weekend patterns differ from weekdays. What do you do? (Account for seasonality — compare Monday to previous Mondays, not to weekday average.)

5. Ground truth labels take 30 days to arrive. How do you detect drift in the meantime? (Monitor prediction distribution, feature distributions, and model confidence as proxy metrics.)

---

When concept drift is suspected, [[Hyperparameter Tuning]] or a full model architecture change may be needed — not just retraining on fresh data.

## 8. Summary

Drift is inevitable. Data drift (input distribution changes) is easier to detect with PSI/KS tests. Concept drift (label relationship changes) is harder — it requires ground truth or proxy metrics. Monitor both globally and by segment. Choose a retraining strategy based on your drift speed and label availability. The key is detecting drift before business impact, not after.

---

## 9. Where to Go Next

- [[Model Monitoring]] — Operational drift monitoring
- [[ML Pipelines]] — Building pipelines that handle retraining
- [[Observability]] — Infrastructure for drift detection
