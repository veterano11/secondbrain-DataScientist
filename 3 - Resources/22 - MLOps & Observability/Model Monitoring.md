---
tags: [mlops, monitoring, advanced]
status: growing
created: 2026-06-27
---

# Monitoreo de Modelos

## 1. Escenario de aprendizaje

Desplegaste un modelo de detección de transacciones fraudulentas y funcionó perfectamente el primer mes. Pero una mañana de lunes descubres que el fin de semana no detectó una nueva modalidad de fraude. Las distribuciones de datos cambiaron — los montos de transacción subieron, los patrones de compra son diferentes — y tu modelo no se enteró. El monitoreo de modelos te avisa de estos cambios en tiempo real, no cuando ya es tarde.

Un modelo que funciona perfectamente el día 1 puede fallar el día 30. Las distribuciones de datos cambian. El comportamiento del usuario evoluciona. Eventos externos (festivos, cambios económicos, acciones de competidores) vuelven obsoletos los datos de entrenamiento.

El monitoreo de modelos detecta estos cambios antes de que causen impacto en el negocio. Es la diferencia entre enterarte el lunes por la mañana que tu modelo de detección de fraudes ha estado fallando todo el fin de semana, y detectarlo en tiempo real.

---

## 2. Los Tres Pilares del Monitoreo de ML

### Diagrama Visual de los Tres Pilares

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRES PILARES DEL MONITOREO                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│                    ┌─────────────────────┐                         │
│                    │   MONITOREO DE ML   │                         │
│                    └─────────────────────┘                         │
│                              │                                      │
│          ┌───────────────────┼───────────────────┐                  │
│          │                   │                   │                  │
│          ▼                   ▼                   ▼                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │
│  │   DERIVA    │    │   DERIVA    │    │ DEGRADACIÓN │            │
│  │   DATOS     │    │  CONCEPTO   │    │  SISTEMA    │            │
│  └─────────────┘    └─────────────┘    └─────────────┘            │
│          │                   │                   │                  │
│          ▼                   ▼                   ▼                  │
│  P(X) cambia         P(y|X) cambia      Problemas ops             │
│  • Nuevos patrones   • Relación desact.  • Latencia ↑             │
│  • Distribuciones    • Modelo obsoleto   • Fugas memoria          │
│  • Valores atípicos  • Spammers nuevos   • Cortes servicio        │
│                                                                     │
│  EJEMPLO:               EJEMPLO:               EJEMPLO:            │
│  Compras prepandemia    Spam 2023 vs 2024     Servicio lento       │
│  vs durante pandemia    Phishing sofisticado  50ms → 5s            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.1 Deriva de Datos

La distribución de entrada $P(X)$ cambia → el modelo ve patrones desconocidos.

**Ejemplo**: un modelo entrenado con comportamiento de compras prepandemia ve patrones de compra completamente diferentes durante un confinamiento.

Ver [[Data & Concept Drift]] para detalles completos.

### 2.2 Deriva de Concepto

La relación $P(y|X)$ cambia → el mapeo del modelo está desactualizado.

**Ejemplo**: un filtro de spam entrenado en 2023 puede no reconocer los sofisticados correos de phishing de 2024 — las características que antes indicaban spam ya no lo hacen.

### 2.3 Degradación del Sistema

Problemas operacionales: picos de latencia, fugas de memoria, cortes de servicio.

**Ejemplo**: un servicio de extracción de características se ralentiza, aumentando la latencia de predicción de 50ms a 5 segundos. Los usuarios agotan el tiempo de espera.

---

## 3. Qué Monitorear

### 3.1 Calidad de Predicción

Requiere datos reales (pueden estar retrasados):

| Métrica | Cómo Medir |
|---|---|
| **Precisión / RMSE** | Comparar predicciones contra resultados reales |
| **Análisis de residuos** | Graficar errores de predicción a lo largo del tiempo |
| **Calibración** | Para modelos probabilísticos: ¿las predicciones con 90% de confianza coinciden con la realidad? |
| **Confianza del modelo** | ¿Las predicciones se están volviendo menos seguras? |

Para técnicas de evaluación más profundas, ver [[Model Evaluation]].

### 3.2 Calidad de Datos

| Verificación | Método |
|---|---|
| **Tasa de valores faltantes** | % de nulos por característica a lo largo del tiempo |
| **Violaciones de rango** | Valores fuera de los límites esperados |
| **Violaciones de tipo** | Tipos de datos inesperados |
| **Cambios de cardinalidad** | Nuevas categorías en características categóricas |

### 3.3 Distribución de Predicciones

| Verificación | Qué Detecta |
|---|---|
| **Predicción media** | Cambio en la salida general del modelo |
| **Proporciones de clase** | Para clasificación: ¿las predicciones están balanceadas? |
| **Distribución de puntuaciones** | ¿Las probabilidades están concentradas o dispersas? |
| **Velocidad de predicción** | Tasa de cambio de predicciones a lo largo del tiempo |

Estas verificaciones aseguran que [[Feature Engineering]] produce entradas consistentes para el modelo.

### 3.4 Salud del Sistema

| Métrica | Alertar Cuando |
|---|---|
| **Latencia p50/p95/p99** | P99 excede el umbral |
| **Tasa de error** | >X% de predicciones devuelven errores |
| **Rendimiento** | Cae por debajo del volumen esperado |
| **Memoria/CPU** | Se acerca a los límites de recursos |
| **Frescura de datos** | Última ejecución exitosa del pipeline |

---

## 4. Métodos de Detección

### 4.1 Pruebas Estadísticas

| Prueba | Qué Compara | Buena Para |
|---|---|---|
| **PSI** | Distribución de referencia agrupada vs producción | Detección general de deriva |
| **KS test** | Distribución de dos muestras continuas | Características continuas |
| **Jensen-Shannon** | Divergencia KL simétrica | Comparar dos distribuciones |
| **Z-score** | Métrica actual vs media histórica | Simple, basada en umbrales |

### 4.2 Monitoreo Basado en Ventanas

Comparar una **ventana de referencia** (ej., datos de entrenamiento o últimos 30 días) con una **ventana actual** (ej., última hora o último día):

```python
def detectar_deriva(referencia: np.array, actual: np.array, umbral: float = 0.1):
    psi = computar_psi(referencia, actual)
    return psi > umbral  # True si se detecta deriva
```

**Tamaños de ventana**:
- Ventana pequeña (1 hora): sensible a cambios recientes, ruidosa
- Ventana grande (7 días): más suave, más lenta en reaccionar
- Multi-ventana: comparar contra múltiples ventanas (1h, 24h, 7d) para detección robusta

### 4.3 Umbrales Adaptativos

Los umbrales estáticos se vuelven obsoletos a medida que los datos evolucionan. Métodos adaptativos:

- **Estadísticas móviles**: media ± 3σ sobre una ventana deslizante
- **EWMA**: media móvil con ponderación exponencial (reacciona más rápido a cambios recientes)
- **Descomposición estacional**: tener en cuenta patrones semanales/diarios antes de detectar deriva

---

## 5. Alertas y Respuesta

### 5.1 Niveles de Severidad

| Nivel | Respuesta | Ejemplo |
|---|---|---|
| **P0 (Crítico)** | Inmediata (5 min) | Modelo devolviendo errores, pipeline de datos caído |
| **P1 (Alto)** | Dentro de 30 min | Latencia de predicción > 5s, caída de precisión > 5% |
| **P2 (Medio)** | Dentro de 4 horas | Deriva gradual detectada, retraso en frescura de datos |
| **P3 (Bajo)** | Siguiente día hábil | Cambio menor en distribución, fluctuación de una métrica |

### 5.2 Runbooks

Toda alerta debe tener una respuesta documentada:

```
Alerta: La precisión del modelo cayó un 10%
1. Verificar si los datos reales están completos (¿quizás las etiquetas están retrasadas?)
2. Comparar predicciones actuales vs distribución de la semana pasada
3. Verificar ejecuciones recientes del pipeline por errores
4. Si se confirma deriva: disparar pipeline de reentrenamiento
5. Si el reentrenamiento tarda > 4 horas: revertir a versión anterior del modelo
6. Validar la corrección con [[A-B Testing]] antes de re-promover
```

---

## 6. Herramientas

| Herramienta | Enfoque |
|---|---|
| **WhyLabs / Whylogs** | Monitoreo de datos y ML |
| **Evidently** | Detección de deriva y evaluación de modelos |
| **Arize AI** | Monitoreo y observabilidad en producción |
| **Grafana + Prometheus** | Métricas del sistema y alertas |
| **MLflow** | [[Experiment Tracking]] + registro de modelos |

---

## 7. Common Mistakes

1. **Monitorear precisión sin latencia**: un modelo que tarda 10 segundos es inútil independientemente de la precisión. Monitorea ambos.

2. **Alertas sin runbooks**: "La precisión del modelo cayó" sin una respuesta documentada lleva al pánico. Escribe runbooks por adelantado.

3. **No considerar retrasos en los datos reales**: el monitoreo de precisión solo es tan oportuno como el bucle de retroalimentación de etiquetas. Para etiquetas retrasadas, usa métricas proxy (distribución de predicciones, confianza).

4. **Ajuste reactivo de umbrales**: ajustar umbrales reactivamente para reducir alertas crea un monitoreo frágil. Establece umbrales basados en análisis histórico.

5. **No monitorear la calidad de los datos**: "basura entra, basura sale" — si la calidad de los datos de entrada se degrada, el modelo también lo hará. Monitorea la calidad de los datos en la etapa del pipeline, no solo la salida del modelo.

---

## 8. Check Your Understanding

1. La precisión de tu modelo cayó de 92% a 85% de la noche a la mañana. ¿Qué verificas primero? (Integridad de los datos reales, deriva de datos, salud del pipeline.)

2. ¿Por qué monitorear la distribución de predicciones incluso antes de tener los datos reales? (Los cambios en la distribución son una señal de alerta temprana antes de que lleguen las etiquetas.)

3. Una alerta P1 se dispara a las 3 AM. El ingeniero de guardia tiene 30 minutos para responder. ¿Qué debería estar en el runbook? (Instrucciones de diagnóstico paso a paso.)

4. Tu prueba KS detecta deriva en una característica que no era importante para el modelo. ¿Deberías alertar? (Probablemente no — enfoca la detección de deriva en características importantes.)

5. ¿Cuál es la diferencia entre PSI y KS test? (PSI discretiza los datos en intervalos primero, KS test trabaja con CDFs continuas. Ambos miden cambio en la distribución.)

---

## 9. Resumen

Model monitoring detects when production models degrade. The three pillars are data drift, concept drift, and system health. Monitor prediction quality (requires ground truth), prediction distribution (early warning), data quality (prevent garbage-in), and system health (latency, errors). Use statistical tests with adaptive thresholds. Alert with severity levels and documented runbooks.

---

## 10. Where to Go Next

- [[Data & Concept Drift]] — Métodos detallados de detección de deriva
- [[ML Pipelines]] — Construyendo pipelines que soporten monitoreo
- [[Observability]] — Registro, métricas y trazado para ML
