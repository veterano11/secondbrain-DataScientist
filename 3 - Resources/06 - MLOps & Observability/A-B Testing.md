---
tags: [mlops, experimentation, advanced]
status: growing
created: 2026-06-27
---

# A/B Testing

## 1. Why This Matters

You built a new model. It gets 1% higher accuracy than the current production model. Should you deploy it? The answer is not obvious — the difference could be noise, or the new model might have hidden failure modes.

A/B testing is how you **reliably compare two models (or two strategies) in production** with real users. It is the standard for making data-driven deployment decisions.

---

## 2. Design of A/B Tests

### 2.1 Basic Setup

- **Control (A)**: current production model
- **Treatment (B)**: new model to evaluate
- **Metric**: what you measure (conversion rate, accuracy, revenue)
- **Randomization unit**: who gets which model (user, session, event)

Users are randomly assigned to A or B. Their experience differs only in which model serves them.

Always validate experiment designs with [[Experiment Tracking]] to log test parameters and outcomes.

### 2.2 Sample Size

$$n = \frac{(Z_{\alpha/2} + Z_\beta)^2 \cdot 2\sigma^2}{\delta^2}$$

- $\delta$: minimum effect you want to detect (MDE)
- $\alpha$: significance level (default 0.05)
- $\beta$: Type II error rate (default 0.2, power = 0.8)
- $\sigma^2$: variance of the metric

**Concrete example** — detecting a 1% conversion rate improvement:
- Baseline CTR: 5% → $\sigma^2 \approx 0.05 \times 0.95 \approx 0.048$
- MDE: 1% absolute (5% → 6%)
- $\alpha = 0.05$, $\beta = 0.2$
- $n \approx 3,500$ per group

If you have 10,000 users/day, the test takes ~1 day. If you have 500 users/day, it takes ~2 weeks.

### 2.3 Duration

**Minimum duration**: at least one full business cycle (1 week minimum to capture weekly patterns).

**Stopping rule**: do NOT stop early when results look significant — early stopping inflates false positives.

**Peeking problem**: checking results every day and stopping when p < 0.05 makes the true Type I error rate much higher than 5%.

---

## 3. Metrics

| Metric Type | Examples | Notes |
|---|---|---|
| **Primary** | Conversion rate, revenue per user | The one you optimize for |
| **Secondary** | Click-through rate, session duration | Additional context |
| **Guardrail** | Latency, error rate, support tickets | Must not degrade |
| **Segment** | Metrics by user type, region | Detect local effects |

Always define **guardrail metrics** — things that must not get worse. A model might improve conversion but increase latency to 10 seconds (destroying user experience).

---

## 4. Statistical Tests

| Data Type | Test | Example |
|---|---|---|
| **Binary** (clicked?, converted?) | Z-test for proportions | CTR comparison |
| **Continuous** (revenue, time) | t-test | Average order value |
| **Count** (purchases per user) | Poisson / negative binomial | Purchase frequency |
| **Rank** (ratings, preferences) | Mann-Whitney U | User satisfaction scores |

### 4.1 Multiple Comparisons

If you test 10 metrics with $\alpha = 0.05$, you have a ~40% chance of at least one false positive.

**Corrections**:
- **Bonferroni**: divide $\alpha$ by number of tests (conservative)
- **FDR** (False Discovery Rate): less conservative, controls expected proportion of false positives

---

## 5. Common Pitfalls

| Pitfall | Problem | Solution |
|---|---|---|
| **Peeking** | Stopping early inflates false positives | Pre-register duration |
| **Lack of observability** | Blind to root cause of metric change | Use [[Observability]] to diagnose |
| **Novelty effect** | Users interact more with anything new | Run test long enough for novelty to wear off |
| **Network effects** | Treatment affects control (e.g., social platform) | Cluster randomization |
| **Selection bias** | Non-random assignment | Proper randomization |
| **Simpson's paradox** | Overall effect reverses within segments | Pre-register segment analysis |

---

## 6. ML-Specific A/B Testing

### 6.1 Challenges

- **Delayed feedback**: loan default prediction takes months to validate
- **Non-stationarity**: seasonality affects both groups
- **Interference**: model A's predictions affect model B's data (e.g., recommendation systems)

### 6.2 Interleaving

For ranking/recommendation models, instead of A/B (each user sees one model), **interleave** the results:

```
User sees: Model A result 1, Model B result 1, Model A result 2, Model B result 2...
Click on: Model B result 1 → preference for Model B
```

More sensitive than A/B for ranking tasks (detects smaller differences).

### 6.3 Multi-Armed Bandits

Instead of a fixed 50/50 split, dynamically allocate more traffic to the better-performing model. Integrate bandit logic into [[ML Pipelines]] for automated model rollouts:

```
Day 1: A=50%, B=50%  (exploration)
Day 2: A=45%, B=55%  (B is slightly better)
Day 3: A=30%, B=70%  (B is clearly better)
Day 7: A=5%,  B=95%  (B is deployed)
```

Minimizes opportunity cost during testing. More complex to analyze.

---

## 7. Common Mistakes

1. **Stopping too early**: "the p-value is 0.04 after 2 days" — keep running. Early stopping invalidates the test.

2. **Ignoring practical significance**: a 0.1% improvement may be statistically significant (p < 0.05) but not worth the engineering cost of deployment.

3. **Not accounting for delayed feedback**: a loan model seems better for 6 months, then defaults start appearing.

4. **Comparing more than 2 variants without correction**: comparing 5 models requires correcting for multiple comparisons.

5. **Running too long without checking**: if a test runs for months, users may be exposed to a worse model unnecessarily. Use bandits for long-running tests.

---

## 8. Check Your Understanding

1. Your A/B test has been running for 3 days. The p-value is 0.03. Should you stop and declare B the winner? (No — early stopping inflates false positives. Wait until the pre-registered duration.)

2. Model A increases conversion by 2% but increases latency from 50ms to 2s. Is A a winner? (No — guardrail metric was violated.)

3. You test 20 metrics and 1 shows p < 0.05. Is this evidence of a real effect? (Maybe not — with 20 tests, 1 significant result is expected by chance under the null.)

4. Why does interleaving detect smaller differences for ranking models? (Each user sees both models, controlling for user-level variance.)

5. A multi-armed bandit allocates 80% of traffic to model B after 1 week. Can you trust the result? (Be cautious — early concentration can be based on noisy estimates. Use a minimum exploration rate.)

---

## 9. Summary

A/B testing is how you reliably compare models in production. Design the test before running it: define the metric, calculate sample size, set duration. Use guardrail metrics to catch regressions. Avoid peeking (checking results early). For ranking tasks, consider interleaving. For minimizing opportunity cost, use multi-armed bandits. The key principle: pre-register the analysis plan and do not deviate.

---

## 10. Where to Go Next

- [[Model Evaluation]] — Offline evaluation to decide what to test online
- [[Statistics]] — Hypothesis testing fundamentals
- [[Model Monitoring]] — Monitoring deployed models during and after A/B tests
- [[Feature Engineering]] — Feature design for experiment consistency
