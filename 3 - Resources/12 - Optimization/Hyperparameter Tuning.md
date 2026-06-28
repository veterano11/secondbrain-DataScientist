---
tags: [optimization, hyperparameters, tuning, automl]
status: growing
created: 2026-06-27
---

# Hyperparameter Tuning

## 1. Escenario de aprendizaje

Entrenaste un XGBoost para clasificación, usaste los defaults, y obtuviste AUC=0.82. Tu colega usó el mismo modelo pero tuneó hiperparámetros y llegó a AUC=0.87. Sin cambiar una línea de features, solo ajustando parámetros.

La diferencia entre un modelo promedio y uno excelente suele estar en el tuning. Esta nota te da las herramientas para encontrarlos.

## 2. ¿Qué son los hiperparámetros?

No se aprenden de los datos. Los setea el humano antes del entrenamiento.

| Parámetro | Hiperparámetro |
|-----------|---------------|
| Pesos $w$ de una red | Learning rate |
| Coeficientes de regresión | $C$ (regularización en SVM) |
| Bias de una neurona | Número de capas |
| Splits del decision tree | Profundidad máxima |

## 3. Grid Search

Probás todas las combinaciones de una grilla predefinida.

```python
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth':    [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
}

grid = GridSearchCV(XGBClassifier(), param_grid, cv=5, scoring='roc_auc')
grid.fit(X_train, y_train)

print(f"Mejores params: {grid.best_params_}")
print(f"Mejor score: {grid.best_score_:.4f}")
```

**Problema**: $3 \times 3 \times 3 = 27$ combinaciones. Cada una entrena 5 folds = 135 entrenamientos. Si agregás 2 parámetros más con 3 valores cada uno: $3^5 = 243$ combinaciones × 5 folds = 1215 entrenamientos.

**La maldición de la dimensionalidad**: el número de combinaciones crece exponencialmente con el número de parámetros.

## 4. Random Search

En vez de probar todas las combinaciones, muestreás al azar del espacio de búsqueda.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

param_dist = {
    'n_estimators': randint(100, 1000),
    'max_depth':    randint(3, 15),
    'learning_rate': uniform(0.001, 0.3),
    'subsample':     uniform(0.5, 0.5),
    'colsample_bytree': uniform(0.5, 0.5),
}

random_search = RandomizedSearchCV(
    XGBClassifier(), param_dist, n_iter=50, cv=5, scoring='roc_auc'
)
random_search.fit(X_train, y_train)
```

**¿Por qué random search es mejor que grid?** (Bergstra & Bengio 2012):

Imaginá que solo 2 de 5 parámetros importan realmente. Grid Search desperdicia la mayoría de las evaluaciones variando parámetros irrelevantes. Random Search cubre más valores de los parámetros importantes con la misma cantidad de evaluaciones.

```text
Grid Search con 5 params y 5 valores c/u: 3125 evaluaciones
  → cada valor de learning_rate se prueba solo 625 veces
Random Search con 100 evaluaciones:
  → cada parámetro se muestrea con 100 valores distintos
  → más cobertura de los parámetros que importan
```

## 5. Bayesian Optimization

Grid y random tratan cada evaluación como independiente. Bayesian Optimization construye un **modelo sustituto** (surrogate) de la función objetivo y lo usa para elegir la próxima configuración.

### 5.1 Cómo funciona

1. Evaluá unas configuraciones al azar (etapa de warmup)
2. Ajustá un Gaussian Process a los resultados
3. Elegí la próxima configuración maximizando una **función de adquisición** (Expected Improvement, UCB)
4. Evaluá la configuración elegida
5. Actualizá el modelo
6. Volvé al paso 3

```python
from skopt import gp_minimize

def objective(params):
    n_est, max_depth, lr = params
    model = XGBClassifier(
        n_estimators=int(n_est),
        max_depth=int(max_depth),
        learning_rate=lr,
    )
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc').mean()
    return -score  # minimizamos (score negativo)

res = gp_minimize(
    objective,
    dimensions=[
        (100, 1000),        # n_estimators
        (3, 15),            # max_depth
        (1e-3, 1e-0, 'log-uniform'),  # learning_rate
    ],
    n_calls=50,
    acq_func='EI',  # Expected Improvement
    random_state=42,
)

print(f"Mejor score: {-res.fun:.4f}")
print(f"Mejores params: n_est={int(res.x[0])}, depth={int(res.x[1])}, lr={res.x[2]:.4f}")
```

**Resultado típico**: Bayesian Optimization encuentra configuraciones comparables a grid search con 5-10× menos evaluaciones.

## 6. Optuna — el estándar moderno

Optuna mejora Bayesian Optimization con **pruning**: si una configuración se ve mal después de pocas epochs, la mata temprano.

```python
import optuna

def objective(trial):
    n_est = trial.suggest_int('n_estimators', 100, 1000)
    lr = trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True)
    max_depth = trial.suggest_int('max_depth', 3, 15)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)

    model = XGBClassifier(
        n_estimators=n_est,
        learning_rate=lr,
        max_depth=max_depth,
        subsample=subsample,
        early_stopping_rounds=10,
    )

    for epoch in range(100):
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  verbose=False)
        score = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

        # Reportar y permitir pruning
        trial.report(score, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return score

study = optuna.create_study(
    direction='maximize',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10)
)
study.optimize(objective, n_trials=100)

print(f"Best trial: {study.best_trial.params}")
print(f"Best score: {study.best_trial.value:.4f}")
```

**Optuna vs skopt vs hyperopt**:

| Característica | Optuna | skopt | Hyperopt |
|---------------|--------|-------|----------|
| Pruning | Sí | No | No |
| API Pythonic | Sí | Sí | Parcial |
| Visualización | Dashboard | Básica | Básica |
| Distribuido | Sí | Limitado | Sí |

## 7. PBT (Population-Based Training)

Usado en DeepMind para AlphaStar y Google para LLMs. En lugar de optimizar una configuración, mantenés una **población** de modelos entrenándose en paralelo. Periódicamente, los modelos peores copian pesos y parámetros de los mejores, y mutan sus hiperparámetros.

```
Población inicial: 50 modelos con HPs aleatorios
Cada 100 steps:
  1. Evaluar todos los modelos
  2. El 20% peor → copia pesos del 20% mejor
  3. Mutar HPs (lr *= 1.2 o /= 1.2)
```

**¿Cuándo usarlo?** Cuando el HP óptimo cambia durante el entrenamiento (learning rate que debería decaer, batch size que debería crecer). PBT se adapta dinámicamente.

## 8. Rangos típicos por modelo

### XGBoost / LightGBM
| HP | Rango | Escala | Nota |
|----|-------|--------|------|
| learning_rate | [0.001, 0.3] | Log | Más bajo requiere más n_estimators |
| n_estimators | [100, 2000] | Log | Early stopping lo determina |
| max_depth | [3, 15] | Entero | Árboles profundos overfitean |
| subsample | [0.5, 1.0] | Uniforme | Muestreo de filas |
| colsample_bytree | [0.3, 1.0] | Uniforme | Muestreo de columnas |
| reg_lambda | [1e-3, 10] | Log | Regularización L2 |

### [[Neural Networks|Redes neuronales]]
| HP | Rango | Escala |
|----|-------|--------|
| learning_rate | [1e-5, 1e-1] | Log |
| batch_size | [16, 512] | Potencia de 2 |
| dropout | [0.0, 0.5] | Uniforme |
| weight_decay | [1e-6, 1e-2] | Log |
| hidden_dim | [64, 1024] | Entero (potencia de 2) |

## 9. Errores comunes

1. **Over-tuning la validación**: si hacés 500 trials con random seed distinto, eventualmente encontrás una combinación que funciona increíble en validación por pura suerte. Validación externa final intocable.

2. **No fijar la semilla aleatoria**: el ruido de entrenamiento puede hacer que A parezca mejor que B cuando en realidad es al revés. Ejecutá los mejores trials con 3-5 semillas.

3. **Tunear parámetros que no importan**: `n_jobs` y `verbose` no afectan el score pero consumen trials en grid search. Excluilos del espacio de búsqueda.

4. **Rangos de búsqueda mal elegidos**: `learning_rate` entre [0.0, 1.0] en escala lineal es inútil — se pasa todo el tiempo entre 0.4 y 0.6 donde nunca funciona. Usá escala logarítmica.

## 10. Check Your Understanding

1. Tenés 10 hiperparámetros, cada uno puede tomar 5 valores. Grid Search = 10^5 combinaciones. ¿Cuántas evaluaciones necesitarías aprox con Bayesian Optimization para encontrar algo bueno? (Tip: típicamente 10-20× el número de parámetros)
2. ¿Por qué random search es más eficiente que grid search cuando solo 3 de 10 parámetros importan?
3. Optuna cancela trials que van mal con pruning. ¿Cómo sabe si un trial "va mal" temprano?
4. En PBT, los modelos malos copian pesos de los buenos. ¿Qué pasa con la diversidad de la población?

**Respuestas rápidas:**
1. ~100-200 evaluaciones con Bayesian Optimization, vs 10^5 con Grid Search.
2. Grid Search explora cada combinación en una cuadrícula fija. Random Search cubre más valores distintos de los parámetros importantes.
3. Compara la performance parcial del trial contra la mediana de trials previos al mismo paso.
4. Se pierde diversidad — pero la mutación de HPs reintroduce variedad.

## 11. Summary

El tuning de hiperparámetros separa modelos promedio de modelos competitivos. Combinado con buenas [[Training Techniques]], es la diferencia entre un proyecto de investigación y uno productivo. Grid Search no escala, Random Search es mejor, Bayesian Optimization es mejor aún (construye un modelo sustituto para elegir la próxima configuración). Optuna es el estándar moderno con pruning. Siempre fijar semilla, elegir rangos en escala correcta, y no tocar el test set durante el tuning.

## 12. Where to Go Next

- [[Gradient-Based Optimization]] — los parámetros que los HPs controlan
- [[Gaussian Processes]] — surrogate model detrás de Bayesian Optimization
- [[Experiment Tracking]] — registrar y comparar trials
- [[Model Evaluation]] — validación cruzada y test set en el contexto de tuning
- [[Gradient-Based Optimization]] — learning rate schedulers, momentum
