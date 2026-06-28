---
tags: [mlops, pipelines, core]
status: growing
created: 2026-06-27
---

# Pipelines de ML

## 1. Escenario de aprendizaje

Tu equipo entrenó un modelo de detección de fraudes en un Jupyter Notebook con precisión del 95%. Pero cuando intentan ponerlo en producción, los datos llegan con formatos diferentes, las transformaciones se aplican distinto y nadie sabe cómo reproducir los resultados del entrenamiento. Pasar de "funciona en mi máquina" a "funciona de manera confiable en producción" requiere automatizar cada paso del flujo: desde que los datos llegan hasta que el modelo genera predicciones.

Un modelo en un Jupyter Notebook no es un sistema desplegado. La brecha entre "funciona en mi máquina" y "funciona de manera confiable en producción" se llena con **pipelines de ML**: flujos de trabajo automatizados, repetibles y monitoreados que transforman datos crudos en predicciones.

Los pipelines aseguran que cada paso — desde la ingesta de datos hasta el despliegue del modelo — esté automatizado, probado y sea auditable. Sin ellos, los proyectos de ML mueren en producción.

---

## 2. Etapas del Pipeline

```
Ingesta de Datos → Validación → Transformación → Entrenamiento → Evaluación → Despliegue
```

### 2.1 Ingesta de Datos

Obtener datos desde los sistemas fuente hasta el pipeline.

| Patrón | Latencia | Herramientas |
|---|---|---|
| **Batch** | Por hora/día | Airflow, cron, tareas programadas |
| **Streaming** | Tiempo real (segundos) | Kafka, Kinesis, Flink |
| **Basado en disparadores** | Al llegar datos | Cloud functions, webhooks |

**Mejores prácticas**:
- Almacenar datos crudos de forma inmutable (nunca modificar el original)
- Registrar fuente, marca de tiempo y versión de los datos
- Monitorear datos faltantes o llegadas retrasadas

### 2.2 Validación de Datos

Verificar la calidad de los datos antes de que entren al pipeline.

```
✓ El esquema coincide con columnas y tipos esperados
✓ Los rangos de valores están dentro de los límites esperados
✓ La tasa de valores faltantes está por debajo del umbral
✓ No hay claves primarias duplicadas
```

**Herramientas**: Great Expectations, Pandera, TensorFlow Data Validation. Monitorear [[Data & Concept Drift]] después del despliegue.

**Qué sucede cuando la validación falla**:
- **Advertencia**: registrar y continuar (problemas menores)
- **Bloquear**: detener el pipeline (problemas críticos)
- **Alertar**: notificar al equipo

### 2.3 Transformación (Ingeniería de Características)

Convertir datos crudos en características listas para el modelo.

Construye siempre un **solo pipeline** tanto para entrenamiento como para servicio para evitar desviación entrenamiento/servicio:

```python
# MALO: código separado para entrenar y servir
train_features = escalar(X_train)  # ruta de código diferente
serve_features = escalar(X_serve)  # podría divergir con el tiempo

# BUENO: pipeline compartido
pipeline = StandardScaler()
pipeline.fit(X_train)              # ajustar en entrenamiento
train_features = pipeline.transform(X_train)
serve_features = pipeline.transform(X_serve)  # misma lógica
```

### 2.4 Entrenamiento

- Registrar cada experimento ([[Experiment Tracking]])
- Registrar hiperparámetros, métricas y artefactos
- Versionar los datos de entrenamiento y el código
- Ajustar y registrar hiperparámetros ([[Hyperparameter Tuning]])
- Reproducir cualquier resultado anterior

### 2.5 Evaluación

- Comparar candidato vs campeón (modelo actual en producción)
- Evaluar en múltiples métricas (no solo precisión)
- Probar en datos reservados y segmentos de datos (ver [[Model Evaluation]])
- Promover automáticamente si el candidato supera al campeón

### 2.6 Despliegue

| Estrategia | Descripción | Cuándo Usar |
|---|---|---|
| **Sombra** | Nuevo modelo corre en paralelo con producción, sin impacto al usuario | Probar fiabilidad |
| **Canary** | Enrutar pequeño % de tráfico al nuevo modelo | Despliegue gradual |
| **Blue/Green** | Cambio instantáneo entre modelo antiguo y nuevo | Despliegues de bajo riesgo |
| **Rolling** | Reemplazar gradualmente instancias del modelo antiguo | Actualizaciones sin tiempo de inactividad |

Validar cambios de modelo con [[A-B Testing]] antes del despliegue completo.

---

## 3. Almacenes de Características (Feature Stores)

Un **feature store** (Feast, Tecton) resuelve un problema común: la misma característica computada de manera diferente en entrenamiento vs servicio.

**Lo que proporciona**:
- **Definición única**: lógica de características definida una vez, usada en todas partes
- **Servicio online**: recuperación de características de baja latencia (Redis, DynamoDB)
- **Servicio offline**: cómputo batch de características para entrenamiento (S3, BigQuery)
- **Corrección puntual**: las características se computan como estaban en el momento de la predicción (previniendo fuga de datos)

```python
# Ejemplo con Feast
features = feature_store.get_online_features(
    features=["usuario:edad", "usuario:compras_totales", "item:categoria"],
    entity_rows=[{"user_id": 123, "item_id": 456}]
).to_dict()
```

---

## 4. Orquestación

Las herramientas de orquestación programan, monitorean y reintentan pasos del pipeline.

| Herramienta | Características Clave |
|---|---|
| **Airflow** | Basado en DAG, maduro, ecosistema grande |
| **Prefect** | Nativo en Python, mejor manejo de errores |
| **Dagster** | Consciente de datos, enfocado en activos |
| **Kubeflow** | Nativo de Kubernetes, específico para ML |
| **Flyte** | Type-safe, enfocado en ML |

Todas modelan los pipelines como **DAGs** (grafos acíclicos dirigidos) — pasos con dependencias que pueden ejecutarse en paralelo cuando sea posible.

---

## 5. Common Mistakes

1. **Desviación entrenamiento/servicio**: código de ingeniería de características diferente en entrenamiento y servicio. Usa siempre un solo pipeline.

2. **No versionar datos**: no puedes reproducir un modelo sin saber qué versión de datos se usó para entrenarlo. Usa DVC o similar.

3. **Pasos de despliegue manuales**: "alguien ejecuta un script" no es una estrategia de despliegue. Automatiza todo.

4. **Sin monitoreo entre ejecuciones del pipeline**: una falla silenciosa (los datos dejaron de llegar) puede pasar desapercibida por días. Monitorea la frescura de los datos.

5. **Ignorar dependencias**: las transformaciones de características a menudo dependen de datos de referencia (tablas de búsqueda). Versiona estos también.

---

## 6. Check Your Understanding

1. ¿Por qué el entrenamiento y el servicio deberían usar el mismo código de ingeniería de características? (Previene la desviación entrenamiento/servicio — diferencias que degradan el rendimiento en servicio.)

2. Un paso del pipeline falla a las 3 AM. ¿Qué debería suceder? (Alertar al equipo, reintentar si es transitorio, bloquear el pipeline si es crítico.)

3. ¿Cuál es la diferencia entre un feature store y una base de datos regular? (El feature store maneja corrección puntual, servicio online + offline y compartición de características entre equipos.)

4. Tu modelo fue entrenado con datos que tenían una característica "gasto_total". En producción, esta característica se computa de manera diferente. ¿Qué problema esperas? (Desviación entrenamiento/servicio — el modelo puede ver distribuciones diferentes a las esperadas.)

5. Despliegas un nuevo modelo mediante despliegue canary. ¿Con qué fracción de tráfico comienzas? (Típicamente 1-5%, luego aumentas gradualmente mientras monitoreas métricas.)

---

## 7. Summary

ML pipelines automate the end-to-end ML workflow: ingest → validate → transform → train → evaluate → deploy. The key principles are: share feature code between training and serving, version everything (data, code, model), automate all deployment steps, and monitor for failures. A well-built pipeline makes model updates safe, fast, and auditable.

---

## 8. Where to Go Next

- [[Experiment Tracking]] — Registrando experimentos dentro de pipelines
- [[Model Monitoring]] — Monitoreando modelos desplegados
- [[Feature Engineering]] — Características que fluyen a través del pipeline
