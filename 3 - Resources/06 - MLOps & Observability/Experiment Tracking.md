---
tags: [mlops, experiments, core]
status: growing
created: 2026-06-27
---

# Seguimiento de Experimentos

## 1. Escenario de aprendizaje

Obtuviste una precisión del 92% entrenando un modelo, pero dos semanas después nadie recuerda qué hiperparámetros usaste, con qué versión de datos o incluso qué semilla aleatoria. Tu jefe te pide que reproduzcas el resultado para producción y no puedes porque no quedó registro de nada. El seguimiento de experimentos evita exactamente esto: cada entrenamiento queda documentado automáticamente para que cualquier resultado sea reproducible.

"¿Cómo obtuvimos ese resultado de 92% de precisión? ¿Qué hiperparámetros usamos?" Si alguna vez te has hecho esta pregunta, necesitas seguimiento de experimentos.

Los experimentos en ML no son como los experimentos en otro software. No puedes simplemente mirar el código — el mismo código con diferentes semillas aleatorias, versiones de datos de entrenamiento o hiperparámetros produce resultados diferentes. El seguimiento de experimentos registra **todo** lo necesario para reproducir cualquier resultado.

---

## 2. Qué Registrar

### 2.1 Las Cuatro Categorías

| Categoría | Ejemplos |
|---|---|
| **Código** | Hash de commit de Git, rama, diff desde main |
| **Datos** | Ruta del dataset, versión, hash, esquema |
| **Hiperparámetros** | Tasa de aprendizaje, tamaño de lote, arquitectura del modelo, optimizador (ver [[Hyperparameter Tuning]]) |
| **Métricas** | Pérdida, precisión, F1, AUC (entrenamiento + validación) (ver [[Model Evaluation]]) |
| **Artefactos** | Pesos del modelo, predicciones, gráficos, matrices de confusión |
| **Entorno** | Versión de Python, tipo de GPU, versión de CUDA, paquetes instalados |

### 2.2 La Regla de Oro

Si no puedes reproducirlo, no lo registraste.

---

## 3. MLflow

### 3.1 Conceptos Clave

- **Experimento**: un agrupamiento lógico de ejecuciones (ej., "bert-fine-tuning")
- **Ejecución**: una sola corrida (ej., "lr=3e-5, batch=32, epoch=3")
- **Parámetros**: pares clave-valor (hiperparámetros)
- **Métricas**: pares clave-valor que pueden actualizarse en el tiempo (pérdida por época)
- **Artefactos**: archivos (pesos del modelo, gráficos, muestras de datos)

### 3.2 Uso Básico

```python
import mlflow

# Establecer experimento
mlflow.set_experiment("prediccion-rotacion")

# Iniciar una ejecución
with mlflow.start_run(run_name="random-forest-v3"):
    # Registrar parámetros
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("learning_rate", 0.01)

    # Entrenar modelo
    model = RandomForestClassifier(n_estimators=200, max_depth=10)
    model.fit(X_train, y_train)

    # Registrar métricas
    accuracy = model.score(X_val, y_val)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1", f1_score(y_val, model.predict(X_val)))

    # Registrar modelo
    mlflow.sklearn.log_model(model, "model")

    # Registrar artefactos
    mlflow.log_artifact("matriz_confusion.png")
    mlflow.log_artifact("importancias_caracteristicas.png")
```

### 3.3 Visualización de Resultados

```
mlflow ui  # inicia la UI de seguimiento en http://localhost:5000
```

La UI te permite:
- Comparar ejecuciones lado a lado (parámetros, métricas)
- Filtrar por parámetros o métricas
- Descargar artefactos
- Registrar los mejores modelos

---

## 4. Weights & Biases

W&B es una alternativa basada en la nube (o auto-alojada) con paneles más ricos:

```python
import wandb

wandb.init(project="prediccion-rotacion", config={
    "n_estimators": 200,
    "max_depth": 10,
})

model = RandomForestClassifier(**wandb.config)
model.fit(X_train, y_train)

wandb.log({"accuracy": accuracy, "f1": f1})
wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(y_val, preds)})
```

**Ventajas**: paneles colaborativos, monitoreo automático de hardware, creación de informes.

**Desventajas**: dependencia de la nube (disponible auto-alojado), puede ser costoso a escala.

---

## 5. Registro de Modelos

### 5.1 Propósito

Rastrear qué modelo está en producción y gestionar versiones de modelos.

```
Desarrollo → Staging → Producción → Archivado
```

Cada transición de etapa requiere:
- Criterios de validación (umbral de precisión, verificación de equidad)
- Aprobación (manual o automatizada)
- Registro de auditoría (quién promovió, cuándo, por qué)

### 5.2 Registro de Modelos en MLflow

```bash
# Registrar un modelo
mlflow.register_model("runs:/<run_id>/model", "clasificador-rotacion")

# Promover a staging
mlflow.models.transition_model_version_stage(
    name="clasificador-rotacion",
    version=2,
    stage="Staging"
)
```

---

## 6. Mejores Prácticas

1. **Registrar cada ejecución**: incluso las ejecuciones "fallidas" contienen información valiosa (qué hiperparámetros no funcionaron).

2. **Registrar líneas base**: siempre una línea base simple (predicción media, modelo lineal) para comparación. Combina con [[A-B Testing]] para validar mejoras online.

3. **Etiquetar experimentos**: mantener hipótesis como etiquetas para que puedas buscar por intención. Registrar señales de [[Data & Concept Drift]] junto con los datos del experimento.

4. **Reproducir a partir de datos registrados**: una ejecución debería ser totalmente reproducible usando solo los parámetros registrados.

5. **Automatizar el registro**: usar callbacks o decoradores para que el registro ocurra automáticamente, no manualmente.

---

## 7. Common Mistakes

1. **No registrar la versión del código**: una ejecución sin hash de commit de Git es irreproducible.

2. **Sobrescribir resultados anteriores**: siempre registrar una nueva ejecución, nunca actualizar una existente.

3. **Registrar demasiadas métricas por ejecución**: la revisión humana es difícil cuando cada ejecución tiene 50+ métricas. Registra las más importantes.

4. **No registrar ejecuciones fallidas**: las ejecuciones fallidas te dicen qué no funciona. Regístralas con una etiqueta de "estado".

5. **Registro manual**: "Registraré los resultados manualmente en una hoja de cálculo" — no lo harás, y la hoja de cálculo se perderá.

---

## 8. Check Your Understanding

1. Un colega pregunta "¿cómo obtuviste esa precisión del 92%?" ¿Qué información necesitas para responder? (Commit de Git, versión del dataset, hiperparámetros, semilla aleatoria.)

2. ¿Por qué registrar líneas base? (Para saber si tu modelo complejo es realmente mejor que uno simple.)

3. MLflow registra parámetros y métricas. ¿Cuál es la diferencia? (Los parámetros se establecen antes del entrenamiento — no cambian. Las métricas se computan durante/después del entrenamiento.)

4. Tienes 100 ejecuciones registradas. ¿Cómo encuentras la mejor? (Filtrar por métrica de validación, comparar parámetros, verificar artefactos en busca de señales de sobreajuste.)

5. Un modelo está en producción. Una versión más nueva falla la validación. ¿Qué sucede? (Se queda en staging — solo los modelos validados son promovidos.)

---

## 9. Summary

Experiment tracking ensures reproducibility. Log everything: code version, data version, hyperparameters, metrics, and artifacts. Use MLflow (self-hosted, open source) or W&B (cloud, richer UI). Maintain a model registry with stage transitions. The rule: if it is not logged, it did not happen.

---

## 10. Where to Go Next

- [[ML Pipelines]] — Integrando seguimiento en pipelines automatizados
- [[Model Monitoring]] — Monitoreando modelos desplegados
- [[Git]] — Control de versiones para código y datos
