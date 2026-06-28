---
tags: [mathematics, statistics, foundational]
status: growing
created: 2026-06-27
---

# Statistics

## 1. Why This Matters

If probability is the mathematics of uncertainty, statistics is the practice of **making decisions with data despite uncertainty**.

You trained a model and it got 92% accuracy. Is that good? How do you know it is not just luck? You tried two different models and one performed 1% better. Is that a real improvement? You deployed a model and next month accuracy dropped to 88%. Is that a real degradation or just random noise?

Statistics gives you the tools to answer these questions. It separates signal from noise. Without it, you are guessing.

---

## 2. Descriptive Statistics — Summarizing Data

Before making inferences, you need to **describe** what you see.

### 2.1 Measures of Central Tendency

**Mean** (average): $\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$
- Sensitive to outliers. A single billionaire in a room of 100 people raises the mean income drastically.

**Median**: the middle value when data is sorted.
- Robust to outliers. The median income barely changes if you add a billionaire.

**Mode**: the most frequent value.
- Useful for categorical data. "The most common product category is Electronics."

**When to use which?**
- Normal distribution → mean (most efficient estimator)
- Skewed data (income, house prices) → median
- Categorical data → mode

### 2.2 Measures of Dispersion

**Variance**: $\sigma^2 = \frac{1}{n}\sum (x_i - \bar{x})^2$
- Average squared distance from the mean.

**Standard Deviation**: $\sigma = \sqrt{\sigma^2}$
- In the same units as the data. If height has mean 170cm and std 10cm, most people are between 160-180cm.

**Interquartile Range (IQR)**: Q3 - Q1 (75th percentile - 25th percentile)
- Robust to outliers. Contains the middle 50% of data.

**Why dispersion matters**: two datasets can have the same mean but very different spreads. A model trained on high-variance data may need more regularization.

### 2.3 Shape

- **Skewness**: asymmetry. Positive skew → long tail to the right (like income). Negative skew → long tail to the left.
- **Kurtosis**: tail heaviness. High kurtosis → more outliers. Financial returns have high kurtosis (fat tails).

---

## 3. Inferential Statistics — Drawing Conclusions from Samples

You never have all the data (the population). You have a **sample**. Statistics tells you what you can infer about the population from that sample.

### 3.1 Estimation

**Point estimate**: a single best guess for a population parameter.
- Example: the sample mean $\bar{x}$ is a point estimate of the population mean $\mu$.

**Confidence interval**: a range that likely contains the true value.

$$CI = \hat{\theta} \pm z_{\alpha/2} \cdot SE$$

- $\hat{\theta}$: point estimate
- $SE$: standard error (standard deviation of the sampling distribution)
- $z_{\alpha/2}$: critical value (1.96 for 95% confidence)

**Interpretation**: "If we repeated this experiment many times, 95% of the confidence intervals would contain the true population parameter."

This does NOT mean "there is a 95% chance the true value is in this interval." The true value either is or is not in the interval. The 95% refers to the **procedure**, not the specific interval.

### 3.2 Hypothesis Testing

**The framework**:

1. **$H_0$ (null hypothesis)**: the default assumption (no effect, no difference)
2. **$H_1$ (alternative hypothesis)**: what you want to prove (there is an effect)

**Example**:
- $H_0$: the new model has the same accuracy as the old model
- $H_1$: the new model has higher accuracy

3. **Choose significance level $\alpha$** (typically 0.05)
4. **Compute a test statistic** and its **p-value**
5. **Decision**: if $p < \alpha$, reject $H_0$ (you have evidence for $H_1$)

### 3.3 The p-value

**Definition**: the probability of observing data as extreme as yours (or more extreme) **assuming $H_0$ is true**.

**What p < 0.05 means**: "If there were truly no effect, we would see data this extreme less than 5% of the time."

**Common misinterpretations**:
- ❌ "There is a 95% chance the effect is real"
- ❌ "P values tell you the size of the effect"
- ✅ "P values tell you how surprising your data would be under $H_0$"

### 3.4 Type I and Type II Errors

| Decision | $H_0$ is true | $H_0$ is false |
|---|---|---|
| **Reject $H_0$** | Type I error (false positive) | Correct! |
| **Fail to reject $H_0$** | Correct! | Type II error (false negative) |

- **Type I rate** = $\alpha$ (you control this — usually 0.05)
- **Type II rate** = $\beta$
- **Power** = $1 - \beta$ (probability of detecting a real effect)

In ML terms:
- False positive: deploying a model that does not actually improve
- False negative: NOT deploying a model that WOULD have improved
- Power: how likely your A/B test is to detect a real improvement

### 3.5 Common Statistical Tests

| Test | What it compares | When to use |
|---|---|---|
| **t-test** | Means of two groups | A/B test, normal-ish data |
| **ANOVA** | Means of 3+ groups | Multiple model comparison |
| **Chi-squared** | Categorical distributions | Feature independence test |
| **Mann-Whitney U** | Medians of two groups | Non-normal data, small samples |
| **KS test** | Full distributions | Detecting data drift |

---

## 4. The Bias-Variance Tradeoff

This is the **central concept** connecting statistics to [[Supervised Learning]].

### 4.1 Definitions

- **Bias**: error from assuming the model form is simpler than reality. A linear model on non-linear data has high bias.
- **Variance**: error from sensitivity to fluctuations in training data. A deep decision tree has high variance.

$$E[(y - \hat{f}(x))^2] = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

### 4.2 The Tradeoff

- **Simple model** (linear regression): high bias, low variance
- **Complex model** (deep tree): low bias, high variance
- **The goal**: find the sweet spot where total error is minimized

**Visual description**: imagine shooting arrows at a target.
- High bias: all arrows clustered in the wrong place (off-center)
- High variance: all arrows scattered randomly (some close, some far)
- Ideal: all arrows clustered at the center (low bias, low variance)

### 4.3 Consequences in Practice

- **Underfitting** (high bias): model is too simple. Training error is high. Fix: more features, more complex model.
- **Overfitting** (high variance): model memorized training data. Training error is low but validation error is high. Fix: regularization, more data, simpler model.

The learning curve tells you which problem you have:
- If training and validation curves converge but both are high → high bias (need a more expressive model)
- If there is a large gap between training and validation → high variance (need regularization or more data)

---

## 5. Correlation and Causation

### 5.1 Correlation

$$\rho_{X,Y} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

- Ranges from -1 to 1
- 0 means no linear relationship
- ±1 means perfect linear relationship

### 5.2 Correlation ≠ Causation

Classic example: ice cream sales and drowning incidents are correlated. Does ice cream cause drowning? No. Both are caused by hot weather (a **confounder**).

In ML: a model might learn that "umbrella sales" predicts "rain", but umbrella sales do not cause rain. This matters when you deploy the model in a new environment where the correlation might break.

### 5.3 Simpson's Paradox

A trend appears in several groups but disappears or reverses when the groups are combined.

**Example**: UC Berkeley gender bias case. In 1973, UC Berkeley was sued for gender bias: overall, men were admitted at a higher rate than women. However, when examining individual departments, most departments had **equal or higher** admission rates for women. The paradox was caused by women applying to more competitive departments (with lower admission rates overall).

**Takeaway**: always check for confounding variables. Your ML model's performance may hide biases that only appear when segmenting the data.

---

## 6. Statistical Thinking in ML

| ML Concept | Statistical Parallel |
|---|---|
| Training loss | Sample error (how well you fit the data) |
| Validation loss | Out-of-sample error (how well you generalize) |
| L2 regularization | Bayesian prior (weights ~ Normal(0, 1/λ)) |
| L1 regularization | Bayesian prior (weights ~ Laplace(0, 1/λ)) |
| Cross-validation | Repeated sampling to estimate generalization error |
| Ensemble methods | Reducing variance by averaging multiple estimators |
| Gradient descent | Optimization for MLE/MAP estimation |

---

## 7. Common Mistakes

1. **p-hacking**: running many tests and reporting only the significant ones. If you test 20 features for significance with $\alpha=0.05$, one will appear significant by chance alone.

2. **Ignoring multiple comparisons**: if you compare 10 models and pick the best, the probability that the best is an overestimate is much higher than 5%. Correct with Bonferroni or FDR.

3. **Confusing practical and statistical significance**: a result can be statistically significant (p < 0.05) but practically irrelevant (0.001% improvement).

4. **Not checking assumptions**: t-tests assume normality. Linear regression assumes homoscedasticity. Violating assumptions can invalidate conclusions.

5. **Survivorship bias**: analyzing only successful cases. In ML: evaluating models only on data that reached production ignores the failures, which contain valuable information.

---

## 8. Check Your Understanding

1. A/B test: model A has 94.2% accuracy, model B has 94.5% accuracy, p = 0.04. What can you conclude?
2. Your model has high training accuracy but low test accuracy. Is this bias or variance? What do you do?
3. Why does cross-validation give a better estimate of generalization error than a single train/test split?
4. You find a correlation of -0.9 between years of experience and errors. Can you conclude that more experience causes fewer errors?
5. A 95% confidence interval for the click-through rate is [0.032, 0.038]. Does this mean there is a 95% chance the true CTR is between 3.2% and 3.8%?

---

## 9. Summary

Statistics is the bridge between data and decisions. Descriptive statistics summarizes what you see. Inferential statistics tells you what you can conclude beyond your data. Hypothesis testing helps you separate signal from noise. The bias-variance tradeoff is the unifying concept that connects statistics to ML — it explains overfitting, underfitting, and why model complexity must be carefully controlled.

---

## 10. Where to Go Next

- [[Probability]] — The mathematical foundation for statistics
- [[Model Evaluation]] — Cross-validation, metrics, and how to measure performance
- [[A/B Testing]] — Applying hypothesis testing to compare models in production
- [[Feature Engineering]] — Why standardization, encoding, and selection matter
- [[Bayesian Inference]] — Bayesian vs. frequentist approaches to statistics
- [[Linear Algebra]] — Covariance, PCA, and the math behind statistical methods
