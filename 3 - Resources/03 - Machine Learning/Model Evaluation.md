---
tags: [machine-learning, evaluation, core]
status: growing
created: 2026-06-27
---

# Evaluación de Modelos

## 1. Escenario de aprendizaje

Has entrenado un modelo para diagnosticar cáncer de mama a partir de imágenes de resonancia. En tus pruebas obtuviste 98% de precisión. Pero al desplegarlo en un hospital real, el modelo falla estrepitosamente: clasifica tumores malignos como benignos. ¿Qué pasó? Hiciste overfitting — tu modelo memorizó los datos de entrenamiento y no generaliza a pacientes nuevos. La evaluación rigurosa es cómo detectas esto. Sin una evaluación adecuada, no sabes si tu modelo es bueno o solo tuvo suerte.

La evaluación responde tres preguntas:
1. **¿Qué tan bueno es este modelo?** (métrica)
2. **¿Puedo confiar en ese número?** (varianza de la estimación)
3. **¿Funcionará con datos nuevos?** (generalización)

---

## 2. El Framework de Entrenamiento / Validación / Prueba

### 2.1 Los Tres Conjuntos

```python
from sklearn.model_selection import train_test_split

# Paso 1: Separar conjunto de prueba inmediatamente (nuncar tocarlo hasta el final)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2)

# Paso 2: Dividir lo restante en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25)
# División final: 60% entrenamiento, 20% validación, 20% prueba
```

| Conjunto | Propósito | Frecuencia de uso |
|---|---|---|
| **Entrenamiento** | Aprender parámetros (pesos) | Cada iteración |
| **Validación** | Ajustar hiperparámetros, detectar overfitting | Cada experimento |
| **Prueba** | Evaluación final, imparcial | Exactamente una vez |

**La regla de oro**: el conjunto de prueba nunca debe influir en ninguna decisión — ni selección de features, ni ajuste de hiperparámetros, ni preprocesamiento. Una vez que lo usas para una decisión, ya no es una estimación imparcial.

### 2.2 Validación Cruzada

En lugar de un solo conjunto de validación, usar validación cruzada k-fold para estimaciones más confiables:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"Precisión: {scores.mean():.3f} ± {scores.std():.3f}")
```

**Cómo funciona k-fold**:
1. Dividir datos de entrenamiento en $k$ pliegues iguales
2. Para cada pliegue: entrenar en $k-1$ pliegues, evaluar en el pliegue apartado
3. Promediar los $k$ scores

**Elección de $k$**:
- $k=5$ o $k=10$: opciones estándar (buen tradeoff sesgo-varianza)
- $k=n$ (LOO — Leave One Out): alta varianza, costoso — rara vez vale la pena
- **Estratificado**: preserva proporciones de clases en cada pliegue (para clasificación)

La validación cruzada te da dos cosas: una estimación más confiable del rendimiento (promediada entre pliegues) y una estimación de la varianza (cuánto depende el rendimiento de qué puntos de datos están en entrenamiento). Ver [[Statistics]] para más sobre teoría de estimación.

---

## 3. Métricas

### 3.1 Métricas de Clasificación

**La Matriz de Confusión** — todo se deriva de esta:

```
              Predicho: 0    Predicho: 1
Real: 0            VN            FP
Real: 1            FN            VP
```

**Exactitud (Accuracy)**: $\frac{VP + VN}{VP + VN + FP + FN}$
- **Funciona para**: clases balanceadas
- **Fall para**: clases imbalanceadas (99% de exactitud en datos 99:1 es trivial)

**Precisión**: $\frac{VP}{VP + FP}$
- "¿Cuándo el modelo predice positivo, con qué frecuencia tiene razón?"
- **Optimizar para**: minimizar falsos positivos (detección de spam, alertas de fraude)

**Sensibilidad (Recall)**: $\frac{VP}{VP + FN}$
- "¿Qué fracción de positivos reales atrapó el modelo?"
- **Optimizar para**: minimizar falsos negativos (detección de enfermedades, sistemas de seguridad)

**Score F1**: $2 \cdot \frac{P \cdot R}{P + R}$
- Media armónica de precisión y sensibilidad
- **Usar cuando**: tanto falsos positivos como falsos negativos importan

**ROC-AUC**: área bajo la curva ROC (TPR vs FPR en todos los umbrales)
- **Interpretación**: "probabilidad de que un positivo aleatorio tenga puntuación más alta que un negativo aleatorio" (concepto clave de [[Probability]])
- **Usar para**: comparar clasificadores, datos imbalanceados
- **Rango**: 0.5 (aleatorio) a 1.0 (perfecto)

**Precisión-Sensibilidad AUC**: mejor que ROC-AUC para datos altamente imbalanceados.

### 3.2 Métricas de Regresión

| Métrica | Fórmula | Interpretación |
|---|---|---|
| **MSE** | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Penaliza errores grandes fuertemente |
| **RMSE** | $\sqrt{MSE}$ | Mismas unidades que el objetivo |
| **MAE** | $\frac{1}{n}\sum| y_i - \hat{y}_i|$ | Menos sensible a outliers |
| **R²** | $1 - \frac{SS_{res}}{SS_{tot}}$ | Proporción de varianza explicada |

**RMSE vs MAE**: si los errores están distribuidos normalmente, RMSE ≈ 1.25 × MAE. Si RMSE >> MAE, existen outliers grandes.

**R²**: 1.0 = predicción perfecta. 0.0 = predecir la media. Negativo = peor que la media.

### 3.3 Métricas de Clustering

| Métrica | Qué mide |
|---|---|
| **Silueta** | Cohesión vs separación (-1 a 1) |
| **Davies-Bouldin** | Similitud promedio de cada cluster con el más similar |
| **Inercia** | Suma de distancias cuadradas al centroide (K-Means) |

La silueta es la más interpretable: 0.7+ = clusters bien separados, 0.3-0.5 = superpuestos, <0.2 = esencialmente sin estructura.

---

## 4. Overfitting y Underfitting

### 4.1 Diagnóstico con Curvas de Aprendizaje

```
# Alto Sesgo (underfitting)
Exactitud entrenamiento:  0.82
Exactitud validación:     0.80
Entrenamiento = Validación ≈ bajo → modelo demasiado simple para los datos

# Alta Varianza (overfitting)
Exactitud entrenamiento:  0.99
Exactitud validación:     0.85
Gran brecha → modelo está memorizando datos de entrenamiento
```

### 4.2 Qué Hacer

| Problema | Síntomas | Soluciones |
|---|---|---|
| **Alto sesgo** | Error alto en entrenamiento y validación | Más features, modelo más complejo, menos regularización |
| **Alta varianza** | Error bajo en entrenamiento, alto en validación | Más datos, regularización, modelo más simple, selección de features |
| **Ambos** | Error alto en entrenamiento, brecha leve | Más datos + modelo más complejo + regularización |

---

## 5. El Tradeoff Sesgo-Varianza (Profundización)

$$E[(y - \hat{f}(x))^2] = \text{Sesgo}[\hat{f}(x)]^2 + \text{Var}[\hat{f}(x)] + \sigma^2$$

**Intuición**:
- **Sesgo**: ¿qué tan lejos está la predicción promedio del modelo de la verdad?
- **Varianza**: ¿cuánto cambia la predicción si uso diferentes datos de entrenamiento?
- **Error irreducible**: ruido inherente en los datos (no se puede reducir)

**El tradeoff**: modelos más complejos tienen menor sesgo pero mayor varianza. Modelos más simples tienen mayor sesgo pero menor varianza. La complejidad óptima minimiza el error total.

Esto no es solo teoría — determina si agregar más features ayudará o perjudicará, si necesitas más datos, y qué algoritmo elegir. [[Regularization]] controla directamente este tradeoff.

---

## 6. Errores Comunes

1. **Mirar el conjunto de prueba demasiado seguido**: cada vez que evalúas en prueba y luego cambias un hiperparámetro, la estimación de prueba se sesga. El conjunto de prueba es para el reporte final, no para el desarrollo del modelo.

2. **Data leakage en validación cruzada**: escalar antes de dividir, usar información futura, o incluir el objetivo en las features. Los pliegues de CV deben ser completamente independientes.

3. **Ignorar el desbalance de clases**: 99% de exactitud en datos 99:1 es insignificante. Usar precisión, sensibilidad, F1, o PR-AUC.

4. **Comparar modelos en una sola métrica**: la exactitud puede ocultar mal rendimiento en clases minoritarias. Siempre verificar el panorama completo (matriz de confusión, múltiples métricas).

5. **No verificar curvas de aprendizaje**: si solo miras métricas finales, pierdes si estás haciendo underfitting o overfitting. Las curvas de aprendizaje te dicen qué hacer después.

---

## 7. Comprueba tu Conocimiento

1. Obtienes 99% de exactitud en un dataset donde 99% de muestras son clase A y 1% son clase B. ¿Es bueno? ¿Qué métrica deberías usar?
2. ¿Por qué la validación cruzada da una estimación más confiable que una sola división entrenamiento/validación?
3. Un modelo tiene RMSE = 5.0 y MAE = 2.0. ¿Qué te dice esto sobre la distribución de errores?
4. Exactitud de entrenamiento = 0.99, exactitud de validación = 0.88. ¿Estás haciendo overfitting o underfitting? ¿Cómo lo arreglarías?
5. El conjunto de prueba debe usarse exactamente una vez. ¿Qué pasa si lo usas múltiples veces?

---

## 8. Resumen

La evaluación adecuada es lo que separa resultados reales de ML de un overfitting accidental. El framework entrenamiento/validación/prueba previene data leakage. Las métricas deben coincidir con el problema (exactitud para balanceados, F1/PR-AUC para imbalanceados). Las curvas de aprendizaje diagnostican sesgo vs varianza. La validación cruzada da estimaciones confiables. La regla de oro: el conjunto de prueba es sagrado — tócalo solo una vez, al final.

---

## 9. ¿Dónde ir Siguente?

- [[Supervised Learning]] — Modelos que necesitan evaluación
- [[Feature Engineering]] — Features que afectan sesgo y varianza
- [[A/B Testing]] — Evaluar modelos en producción
