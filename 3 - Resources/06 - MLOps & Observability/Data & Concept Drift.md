---
tags: [mlops, drift, advanced]
status: growing
created: 2026-06-27
---

# Deriva de Datos y Concepto

## 1. Escenario de aprendizaje

Tu modelo de predicción de riesgo crediticio se entrenó cuando las tasas de interés estaban en 3%. Ahora las tasas subieron al 8%. Un perfil de cliente que antes era "bajo riesgo" de repente es "riesgo medio" — los mismos ingresos y puntaje crediticio ahora tienen mayor probabilidad de default porque pagar la deuda es más caro. La relación entre los datos y lo que predices cambió. Eso es deriva de concepto. Si no la detectas, tu modelo está tomando decisiones incorrectas sin que nadie lo sepa.

Ninguna distribución de datos permanece estática. El comportamiento del cliente evoluciona. Los mercados cambian. Las estaciones varían. Un modelo entrenado con datos del año pasado inevitablemente enfrentará patrones diferentes hoy. Entender la deriva — cuándo y por qué sucede — es esencial para mantener modelos fiables en producción.

---

## 2. Deriva de Datos (Covariate Shift)

### 2.1 Definición

La distribución de entrada cambia: $P_{\text{entrenamiento}}(X) \neq P_{\text{producción}}(X)$, pero $P(y|X)$ permanece igual.

**Ejemplo**: un modelo de detección de fraudes entrenado con datos de transacciones de 2023 ve transacciones de 2024. Los montos promedio de transacciones aumentan un 15% debido a la inflación. El modelo nunca ha visto esta distribución de montos.

### 2.2 Causas Comunes

| Causa | Ejemplo |
|---|---|
| **Estacionalidad** | Patrones de compras navideñas, comportamiento dependiente del clima |
| **Evolución del usuario** | Los usuarios se vuelven más sofisticados con el tiempo |
| **Eventos externos** | COVID, recesión económica, cambios de competidores |
| **Cambios en el pipeline de datos** | Nuevo sensor, nuevo formato de registro, cambios en datos upstream |
| **Sesgo de muestreo** | Los datos de entrenamiento se recolectaron de manera diferente a los datos de producción |

### 2.3 Detección

**Population Stability Index (PSI)**:

$$\text{PSI} = \sum_i (p_i - q_i) \cdot \ln\left(\frac{p_i}{q_i}\right)$$

- $p_i$: proporción en el intervalo de producción $i$
- $q_i$: proporción en el intervalo de referencia (entrenamiento) $i$
- PSI < 0.1: sin cambio significativo
- PSI 0.1-0.2: cambio moderado (investigar)
- PSI > 0.2: deriva significativa (tomar acción)

**KS Test**: compara la distribución acumulada de los datos de referencia y producción para características continuas.

**Distancia de Wasserstein**: mide el "trabajo" necesario para transformar una distribución en otra. Más sensible que PSI para algunos patrones.

### 2.4 Mitigación

- **Reentrenar**: reentrenar periódicamente con datos frescos
- **Modelo adaptativo**: usar aprendizaje online para actualizarse continuamente
- **Características robustas**: diseñar características que sean estables a lo largo del tiempo (ver [[Feature Engineering]])
- **Monitorear**: detectar deriva temprano y alertar al equipo

---

## 3. Deriva de Concepto

### 3.1 Definición

La relación entre entradas y objetivo cambia: $P_{\text{entrenamiento}}(y|X) \neq P_{\text{producción}}(y|X)$.

**Ejemplo**: un modelo de riesgo crediticio entrenado cuando las tasas de interés eran bajas. Ahora las tasas son altas. El mismo perfil de ingresos y puntaje crediticio que era "bajo riesgo" antes ahora es "riesgo medio" porque las tasas más altas aumentan la probabilidad de default.

### 3.2 Tipos de Deriva de Concepto

```
Súbita:    ▁▁▁▁▁▁▁███   (cambio de regla de fraude de la noche a la mañana)
Gradual:   ▁▁▁▁▂▃▄▅▆▇█   (las preferencias del usuario evolucionan lentamente)
Recurrente: ▁▁▃▁▁▃▁▁▃▁▁▃  (los patrones estacionales se repiten)
```

### 3.3 Detección

La deriva de concepto es más difícil de detectar que la deriva de datos porque requiere datos reales.

**Métodos**:
- **Monitorear errores de predicción**: cuando los errores aumentan, es probable que haya deriva de concepto
- **Monitorear distribución de residuos**: si los errores se vuelven sesgados (ej., siempre sobreprediciendo)
- **Monitorear confianza del modelo**: si el modelo se vuelve menos seguro en sus predicciones
- **Monitorear importancia de características**: si las características importantes cambian con el tiempo (detección basada en SHAP)

---

## 4. Deriva Local vs Global

**Deriva global**: toda la población cambia. Fácil de detectar (tamaño de muestra grande).

**Deriva local**: solo un segmento de la población cambia. Más difícil de detectar (tamaño de muestra más pequeño, pero potencialmente catastrófico para ese segmento).

**Ejemplo**: un modelo de diagnóstico médico funciona bien en general pero empieza a fallar en pacientes ancianos (un segmento pequeño). Las métricas globales se ven bien, pero el modelo es perjudicial para un grupo específico.

**Detección**: monitorear el rendimiento por segmentos — grupos de edad, regiones, niveles de cliente. Usar [[Model Evaluation]] para evaluar cada segmento independientemente.

---

## 5. Estrategias de Reentrenamiento

| Estrategia | Cómo Funciona | Pros | Contras |
|---|---|---|---|
| **Programada** | Reentrenar cada N días/semanas | Simple, predecible | Puede ser muy lenta para deriva súbita |
| **Disparada por rendimiento** | Reentrenar cuando cae la precisión | Reacciona a degradación real | Requiere datos reales |
| **Disparada por deriva** | Reentrenar cuando se detecta deriva | Proactivo | No garantiza mejora en rendimiento |
| **Aprendizaje online** | Actualizar el modelo incrementalmente | Siempre actualizado | Complejo, riesgo de inestabilidad |

En la práctica, la mayoría de los equipos usan una combinación: reentrenamiento programado como base con reentrenamiento disparado por deriva para respuesta rápida. Registra las ejecuciones de reentrenamiento con [[Experiment Tracking]] para comparar rendimiento entre versiones.

---

## 6. Common Mistakes

1. **Confundir deriva de datos con deriva de concepto**: requieren respuestas diferentes. Deriva de datos → reentrenar con nuevos datos. Deriva de concepto → puede necesitar un modelo fundamentalmente diferente.

2. **Establecer umbrales demasiado ajustados**: cada fluctuación menor dispara alertas. Establece umbrales basados en impacto al negocio, no en significancia estadística.

3. **Ignorar el retraso de etiquetas**: si los datos reales tardan 30 días en llegar, la detección de deriva basada en precisión siempre está 30 días atrasada. Usa métricas proxy (distribución de predicciones) para alerta temprana.

4. **No monitorear deriva local**: las métricas globales pueden verse bien mientras un segmento crítico se degrada. Segmenta tu monitoreo.

5. **Reentrenar sin validación**: reentrenar automáticamente con datos con deriva puede amplificar la deriva si los datos más recientes son ruidosos. Siempre valida en un conjunto de validación limpio.

---

## 7. Check Your Understanding

1. Un modelo entrenado con datos de verano predice ventas de helados en invierno. ¿Qué tipo de deriva es? (Deriva de datos — las características de entrada (temperatura, horas de luz) son diferentes.)

2. Un filtro de spam entrenado en 2023 no detecta spam de 2024 porque los spammers cambiaron sus tácticas. ¿Qué tipo de deriva? (Deriva de concepto — la relación entre las características del correo y el spam cambió.)

3. La precisión de tu modelo cayó un 10% pero la distribución de predicciones no cambió. ¿Qué sucedió? (Deriva de concepto — los patrones de entrada son los mismos pero la relación con la etiqueta cambió.)

4. Monitoreas PSI diariamente y recibes alertas cada lunes porque los patrones de fin de semana difieren de los días laborables. ¿Qué haces? (Considerar la estacionalidad — comparar lunes con lunes anteriores, no con el promedio de días laborables.)

5. Las etiquetas reales tardan 30 días en llegar. ¿Cómo detectas deriva mientras tanto? (Monitorear distribución de predicciones, distribuciones de características y confianza del modelo como métricas proxy.)

---

Cuando se sospecha deriva de concepto, puede ser necesario [[Hyperparameter Tuning]] o un cambio completo en la arquitectura del modelo — no solo reentrenar con datos frescos.

## 8. Summary

Drift is inevitable. Data drift (input distribution changes) is easier to detect with PSI/KS tests. Concept drift (label relationship changes) is harder — it requires ground truth or proxy metrics. Monitor both globally and by segment. Choose a retraining strategy based on your drift speed and label availability. The key is detecting drift before business impact, not after.

---

## 9. Where to Go Next

- [[Model Monitoring]] — Monitoreo operacional de deriva
- [[ML Pipelines]] — Construyendo pipelines que manejen reentrenamiento
- [[Observability]] — Infraestructura para detección de deriva
