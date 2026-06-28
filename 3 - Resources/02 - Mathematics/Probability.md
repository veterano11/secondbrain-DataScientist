---
tags: [mathematics, probability, foundational]
status: growing
created: 2026-06-27
---

# Probability

## 1. Why This Matters

Machine learning is fundamentally about making predictions under uncertainty. Will this customer churn? Is this email spam? What is the next word in this sentence? None of these questions have deterministic answers — they are all **probabilistic**.

Every ML model outputs probabilities, whether explicitly (logistic regression gives $P(y=1|x)$) or implicitly (a classifier's decision threshold implies a probability cutoff). Understanding probability lets you:
- Interpret model outputs correctly
- Design loss functions (cross-entropy is derived from likelihood)
- Understand [[Bayesian Inference|Bayesian methods]] (which treat model weights as distributions)
- Detect when your model is uncertain vs confident but wrong

---

## 2. Foundations

### 2.1 The Three Axioms (Kolmogorov)

1. $P(A) \geq 0$ — probabilities are non-negative
2. $P(\Omega) = 1$ — the probability of "something happening" is 1
3. $P(A \cup B) = P(A) + P(B)$ if $A$ and $B$ are disjoint (mutually exclusive)

From these three simple rules, all of probability theory follows.

### 2.2 Visualizing Probability

Think of a **Venn diagram**. The sample space $\Omega$ is the entire rectangle. An event $A$ is a region inside. $P(A)$ is the area of $A$ divided by the total area.

- $P(A \cup B)$: area covered by $A$ or $B$ (or both)
- $P(A \cap B)$: area where $A$ and $B$ overlap
- If $A \cap B = \emptyset$ (no overlap), they are mutually exclusive

### 2.3 The Addition Rule

$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

We subtract the intersection because it was counted twice.

---

## 3. Conditional Probability and Bayes

### 3.1 Conditional Probability

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

**Read as**: "probability of $A$ given that $B$ happened."

**Intuition**: if you know $B$ is true, the only part of the Venn diagram that matters is the $B$ region. $P(A|B)$ is the fraction of $B$ that overlaps with $A$.

**Concrete example**: In a medical test:
- $P(\text{disease}) = 0.01$ (1% of population has the disease)
- $P(\text{positive}|\text{disease}) = 0.99$ (test catches 99% of cases)
- $P(\text{positive}|\text{no disease}) = 0.05$ (5% false positive rate)

If you test positive, what is $P(\text{disease}|\text{positive})$?

Most people guess 99%. The correct answer is much lower because the disease is rare. We will compute it with Bayes.

### 3.2 Bayes Theorem

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

**Let us compute the medical test example**:

$$P(\text{disease}|\text{positive}) = \frac{0.99 \times 0.01}{0.99 \times 0.01 + 0.05 \times 0.99}$$

$$= \frac{0.0099}{0.0099 + 0.0495} = \frac{0.0099}{0.0594} \approx 0.167$$

**Only 16.7%!** Even with a positive test, there is an 83.3% chance you do NOT have the disease. This is because the disease is rare and the test has a 5% false positive rate.

This is why understanding Bayes is essential for interpreting ML model outputs — especially in imbalanced classification.

### 3.3 The Bayesian Framework for ML

Bayes theorem provides a framework for learning from data:

$$P(\text{model}|\text{data}) = \frac{P(\text{data}|\text{model}) \cdot P(\text{model})}{P(\text{data})}$$

- **Prior** $P(\text{model})$: what we believe before seeing data
- **Likelihood** $P(\text{data}|\text{model})$: how well the model explains the data
- **Posterior** $P(\text{model}|\text{data})$: what we believe after seeing data
- **Evidence** $P(\text{data})$: how probable the data is under all models (normalization)

**Maximum Likelihood Estimation (MLE)**: find the model that maximizes $P(\text{data}|\text{model})$ — equivalent to minimizing cross-entropy loss.

**Maximum A Posteriori (MAP)**: find the model that maximizes $P(\text{model}|\text{data})$ — equivalent to MLE with regularization.

---

## 4. Random Variables

### 4.1 Intuition

A random variable is not a variable that "varies randomly." It is a **function that maps outcomes to numbers**.

- **Discrete**: countable outcomes (coin flip → $\{0, 1\}$, dice → $\{1, 2, 3, 4, 5, 6\}$)
- **Continuous**: outcomes in a range (temperature, height, price)

### 4.2 Probability Distributions

A distribution tells you how likely each value is.

**For discrete variables**: Probability Mass Function (PMF)

$$P(X = k) = ...$$

**For continuous variables**: Probability Density Function (PDF)

$$P(a \leq X \leq b) = \int_a^b f(x) dx$$

Note: $P(X = \text{exact value}) = 0$ for continuous variables. You can only talk about ranges.

### 4.3 Expectation and Variance

**Expected value** (the "average" you would see over infinite samples):

$$\text{Discrete: } E[X] = \sum x \cdot P(X=x)$$
$$\text{Continuous: } E[X] = \int x \cdot f(x) dx$$

**Variance** (how spread out the distribution is):

$$\text{Var}[X] = E[(X - E[X])^2] = E[X^2] - E[X]^2$$

**Standard deviation**: $\text{Std}[X] = \sqrt{\text{Var}[X]}$

### 4.4 Linearity of Expectation

$$E[aX + bY] = aE[X] + bE[Y]$$

This holds **always**, even if $X$ and $Y$ are not independent. This property is incredibly useful for deriving results in ML.

---

## 5. Key Distributions

### 5.1 Bernoulli

$$P(X=1) = p, \quad P(X=0) = 1-p$$

- **Use**: binary outcome (click / no click, spam / not spam)
- $E[X] = p$, $\text{Var}[X] = p(1-p)$

### 5.2 Binomial

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

- Number of successes in $n$ independent Bernoulli trials
- $E[X] = np$, $\text{Var}[X] = np(1-p)$

### 5.3 Normal (Gaussian)

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}$$

This is the **most important distribution** in statistics and ML.

**Why?**
- **Central Limit Theorem**: the sum of many independent random variables is approximately normal, regardless of their original distribution
- Many natural phenomena follow a normal distribution (heights, measurement errors)
- It is the assumption behind linear regression, Gaussian Processes, and VAEs

**Properties**:
- Symmetric around $\mu$
- 68% of data within $\mu \pm \sigma$
- 95% within $\mu \pm 2\sigma$
- 99.7% within $\mu \pm 3\sigma$

### 5.4 Other Important Distributions

| Distribution | Use Case | Parameters |
|---|---|---|
| **Poisson** | Count of events in fixed time (e.g., website visits per minute) | $\lambda$ (rate) |
| **Exponential** | Time between events (e.g., time between customer arrivals) | $\lambda$ |
| **Uniform** | Every value equally likely (e.g., random initialization) | $a, b$ |
| **Beta** | Prior for probabilities (conjugate prior for Bernoulli) | $\alpha, \beta$ |

### 5.5 Central Limit Theorem (CLT)

**Statement**: The distribution of the sample mean $\bar{X} = \frac{1}{n}\sum X_i$ approaches a normal distribution as $n \to \infty$, regardless of the original distribution of $X$.

**Why this matters**: it justifies using the normal distribution for confidence intervals, hypothesis tests, and many ML methods, even when the underlying data is not normal.

---

## 6. Joint, Marginal, and Conditional

### 6.1 Joint Distribution

$P(X=x, Y=y)$ — probability both events happen simultaneously.

### 6.2 Marginal Distribution

$$P(X=x) = \sum_y P(X=x, Y=y)$$

"Summing out" the other variable. This is how you get the distribution of one variable from a joint distribution.

### 6.3 Independence

$$P(X, Y) = P(X) P(Y)$$

If $X$ and $Y$ are independent, knowing $X$ tells you nothing about $Y$.

In ML, we often **assume** independence when it is not true (Naive Bayes, i.i.d. assumption). The art is knowing when this assumption is good enough.

### 6.4 Law of Total Probability

$$P(A) = \sum_i P(A|B_i) P(B_i)$$

A way to compute $P(A)$ by considering all possible scenarios $B_i$.

---

## 7. Common Mistakes

1. **The Prosecutor's Fallacy**: confusing $P(\text{evidence}|\text{innocent})$ with $P(\text{innocent}|\text{evidence})$. These are very different (Bayes!).

2. **Ignoring the base rate**: as the medical test example shows, rare events remain rare even after positive evidence.

3. **Assuming independence**: "the probability of 10 heads in a row is $0.5^{10} \approx 0.001$" — but only if the coin is fair AND each flip is independent.

4. **Confusing $P(A \cap B)$ with $P(A|B)$**: they are related ($P(A|B) = P(A \cap B) / P(B)$) but very different.

5. **The Gambler's Fallacy**: "it landed heads 5 times in a row, so tails is due." Each flip is independent — the probability remains 50%.

---

## 8. Check Your Understanding

1. A model predicts $P(\text{spam}|\text{email}) = 0.9$. If 2% of all emails are spam and the model catches 95% of spam, what is the false positive rate?
2. Why is $P(X = 5)$ zero for a continuous distribution?
3. The sample mean of 100 uniform(0,1) variables is approximately normal. What theorem guarantees this?
4. You flip a coin 10 times and get 10 heads. What is the MLE estimate of $p$ (probability of heads)?
5. Why does the sigmoid function $\frac{1}{1+e^{-x}}$ output values between 0 and 1? How is it related to probability?

---

## 9. Summary

Probability is the mathematics of uncertainty. Bayes theorem tells us how to update beliefs with evidence. Distributions (especially the normal) describe how random variables behave. Every ML model is implicitly doing probabilistic inference — understanding probability lets you understand what your model is really saying.

---

## 10. Where to Go Next

- [[Statistics]] — Hypothesis testing, confidence intervals, and inference
- [[Linear Algebra]] — Random vectors, covariance matrices
- [[Supervised Learning]] — Logistic regression outputs probabilities
- [[Model Evaluation]] — AUC, calibration, and probabilistic metrics
- [[Probabilistic Programming]] — Bayesian computation with PyMC, Stan, etc.
- [[Neural Networks]] — Softmax outputs are probability distributions
