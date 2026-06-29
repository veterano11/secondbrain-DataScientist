---
tags: [machine-learning, evaluation, core]
status: growing
created: 2026-06-27
---

# Model Evaluation

## 1. Escenario de aprendizaje

Has entrenado un modelo para diagnosticar cáncer de mama a partir de imágenes de resonancia. En tus pruebas obtuviste 98% de precisión. Pero al desplegarlo en un hospital real, el modelo falla estrepitosamente: clasifica tumores malignos como benignos. ¿Qué pasó? Hiciste overfitting — tu modelo memorizó los datos de entrenamiento y no generaliza a pacientes nuevos. La evaluación rigurosa es cómo detectas esto. Sin una evaluación adecuada, no sabes si tu modelo es bueno o solo tuvo suerte.

Evaluation responde tres preguntas:
1. **¿Qué tan bueno es este modelo?** (métrica)
2. **¿Puedo confiar en ese número?** (varianza de la estimación)
3. **¿Funcionará con datos nuevos?** (generalización)

---

## 2. The Train / Validation / Test Framework

### 2.1 The Three Sets

```python
from sklearn.model_selection import train_test_split

# Step 1: Split off test set immediately (never touch it until the end)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2)

# Step 2: Split remaining into train and validation
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25)
# Final split: 60% train, 20% val, 20% test
```

| Set | Purpose | How often used |
|---|---|---|
| **Training** | Learn parameters (weights) | Every iteration |
| **Validation** | Tune hyperparameters, detect overfitting | Every experiment |
| **Test** | Final, unbiased evaluation | Exactly once |

**The golden rule**: the test set must never influence any decision — not feature selection, not hyperparameter tuning, not preprocessing. Once you use it for a decision, it is no longer an unbiased estimate.

### 2.2 Cross-Validation

Instead of a single validation set, use k-fold cross-validation for more reliable estimates:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
```

**How k-fold works**:
1. Split training data into $k$ equal folds
2. For each fold: train on $k-1$ folds, evaluate on the held-out fold
3. Average the $k$ scores

**Choosing $k$**:
- $k=5$ or $k=10$: standard choices (good bias-variance tradeoff)
- $k=n$ (LOO — Leave One Out): high variance, expensive — rarely worth it
- **Stratified**: preserves class proportions in each fold (for classification)

Cross-validation gives you two things: a more reliable estimate of performance (averaged across folds) and an estimate of the variance (how much performance depends on which data points are in training). See [[Statistics]] for more on estimation theory.

---

## 3. Metrics

### 3.1 Classification Metrics

**The Confusion Matrix** — everything derives from this:

```
              Predicted: 0    Predicted: 1
Actual: 0         TN            FP
Actual: 1         FN            TP
```

**Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
- **Works for**: balanced classes
- **Fails for**: imbalanced classes (99% accuracy on 99:1 data is trivial)

**Precision**: $\frac{TP}{TP + FP}$
- "When the model predicts positive, how often is it right?"
- **Optimize for**: minimizing false positives (spam detection, fraud alerts)

**Recall** (Sensitivity): $\frac{TP}{TP + FN}$
- "What fraction of actual positives did the model catch?"
- **Optimize for**: minimizing false negatives (disease screening, safety systems)

**F1 Score**: $2 \cdot \frac{P \cdot R}{P + R}$
- Harmonic mean of precision and recall
- **Use when**: both false positives and false negatives matter

**ROC-AUC**: area under the ROC curve (TPR vs FPR at all thresholds)
- **Interprets as**: "probability that a random positive is scored higher than a random negative" (a core [[Probability]] concept)
- **Use for**: comparing classifiers, imbalanced data
- **Range**: 0.5 (random) to 1.0 (perfect)

**Precision-Recall AUC**: better than ROC-AUC for highly imbalanced data.

### 3.2 Regression Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| **MSE** | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Penalizes large errors heavily |
| **RMSE** | $\sqrt{MSE}$ | Same units as target |
| **MAE** | $\frac{1}{n}\sum| y_i - \hat{y}_i|$ | Less sensitive to outliers |
| **R²** | $1 - \frac{SS_{res}}{SS_{tot}}$ | Proportion of variance explained |

**RMSE vs MAE**: if errors are normally distributed, RMSE ≈ 1.25 × MAE. If RMSE >> MAE, large outliers exist.

**R²**: 1.0 = perfect prediction. 0.0 = predicting the mean. Negative = worse than the mean.

### 3.3 Clustering Metrics

| Metric | What it measures |
|---|---|
| **Silhouette** | Cohesion vs separation (-1 to 1) |
| **Davies-Bouldin** | Average similarity of each cluster to its most similar one |
| **Inertia** | Sum of squared distances to centroid (K-Means) |

Silhouette is the most interpretable: 0.7+ = well-separated clusters, 0.3-0.5 = overlapping, <0.2 = essentially no structure.

---

## 4. Overfitting and Underfitting

### 4.1 Diagnosis with Learning Curves

```
# High Bias (underfitting)
Train accuracy:  0.82
Val accuracy:    0.80
Train = Val ≈ low → model is too simple for the data

# High Variance (overfitting)
Train accuracy:  0.99
Val accuracy:    0.85
Big gap → model is memorizing training data
```

### 4.2 What to Do

| Problem | Symptoms | Fixes |
|---|---|---|
| **High bias** | Both train and val error high | More features, more complex model, less regularization |
| **High variance** | Low train error, high val error | More data, regularization, simpler model, feature selection |
| **Both** | High train error, slight gap | More data + more complex model + regularization |

---

## 5. The Bias-Variance Tradeoff (Deep Dive)

$$E[(y - \hat{f}(x))^2] = \text{Bias}[\hat{f}(x)]^2 + \text{Var}[\hat{f}(x)] + \sigma^2$$

**Intuition**:
- **Bias**: how far is the model's average prediction from the truth?
- **Variance**: how much does the prediction change if I use different training data?
- **Irreducible error**: noise inherent in the data (cannot be reduced)

**The tradeoff**: more complex models have lower bias but higher variance. Simpler models have higher bias but lower variance. The optimal complexity minimizes total error.

This is not just theory — it determines whether adding more features will help or hurt, whether you need more data, and which algorithm to choose. [[Regularization]] directly controls this tradeoff.

---

## 6. Common Mistakes

1. **Looking at the test set too often**: every time you evaluate on test and then change a hyperparameter, the test estimate becomes biased. The test set is for the final report, not for model development.

2. **Data leakage in cross-validation**: scaling before splitting, using future information, or including the target in features. CV folds must be completely independent.

3. **Ignoring class imbalance**: 99% accuracy on 99:1 data is meaningless. Use precision, recall, F1, or PR-AUC.

4. **Comparing models on a single metric**: accuracy can hide poor performance on minority classes. Always check the full picture (confusion matrix, multiple metrics).

5. **Not checking learning curves**: if you only look at final metrics, you miss whether you are underfitting or overfitting. Learning curves tell you what to do next.

---

## 7. Check Your Understanding

1. You get 99% accuracy on a dataset where 99% of samples are class A and 1% are class B. Is this good? What metric should you use?
2. Why does cross-validation give a more reliable estimate than a single train/val split?
3. A model has RMSE = 5.0 and MAE = 2.0. What does this tell you about the error distribution?
4. Training accuracy = 0.99, validation accuracy = 0.88. Are you overfitting or underfitting? How would you fix it?
5. The test set must be used exactly once. What happens if you use it multiple times?

---

## 8. Resumen

Proper evaluation is what separates real ML results from accidental overfitting. The train/val/test framework prevents data leakage. Metrics must match the problem (accuracy for balanced, F1/PR-AUC for imbalanced). Learning curves diagnose bias vs variance. Cross-validation gives reliable estimates. The golden rule: the test set is sacred — touch it only once, at the very end.

---

## 9. Where to Go Next

- [[Supervised Learning]] — Models that need evaluation
- [[Feature Engineering]] — Features that affect bias and variance
- [[A/B Testing]] — Evaluating models in production
