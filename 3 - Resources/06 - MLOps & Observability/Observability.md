---
tags: [mlops, observability, advanced]
status: growing
created: 2026-06-27
---

# Observabilidad

## 1. Escenario de aprendizaje

Tu modelo está en producción respondiendo miles de predicciones por segundo. Un día, los usuarios empiezan a recibir respuestas lentas. ¿Es el modelo? ¿Es la base de datos? ¿Es la red? Sin observabilidad, no tienes idea de dónde está el problema. Con un sistema de logging, métricas y trazado bien configurado, puedes seguir una petición desde que llega hasta que se responde y ver exactamente dónde se está tardando.

Un modelo está ejecutándose en producción. ¿Está funcionando correctamente? ¿Cómo lo sabes?

La observabilidad es la práctica de hacer que el estado interno de un sistema sea inferible a partir de sus salidas externas. En ML, esto significa saber no solo "¿está respondiendo el modelo?" sino "¿está respondiendo correctamente? ¿Se está degradando lentamente? ¿Está sirviendo a los usuarios correctos con la latencia adecuada?"

Sin observabilidad, estás volando a ciegas. Con ella, puedes detectar, diagnosticar y solucionar problemas antes de que los usuarios los noten.

---

## 2. Los Tres Pilares

### 2.1 Logging

Registros estructurados y consultables de eventos.

**Qué registrar**:
```json
{
  "timestamp": "2026-06-27T14:30:00Z",
  "request_id": "req_abc123",
  "model_version": "v2.3.1",
  "latency_ms": 145,
  "prediction": 0.87,
  "confidence": 0.92,
  "features_hash": "a1b2c3d4",
  "error": null
}
```

**Mejores prácticas**:
- **JSON estructurado** (no texto plano) — consultable con herramientas como jq, Loki, Elasticsearch
- **ID de correlación** — rastrear una solicitud a través de todos los servicios
- **Niveles de log**: DEBUG (desarrollo), INFO (normal), WARN (problema potencial), ERROR (fallo)
- **Nunca registrar PII** — enmascarar u omitir datos personales
- **Vincular logs a versiones de código** — usar hashes de commit de [[Git]] en metadatos de log

### 2.2 Métricas

Mediciones numéricas recolectadas a lo largo del tiempo.

| Tipo de Métrica | Ejemplos | Qué Te Dice |
|---|---|---|
| **Contadores** | Total de solicitudes, total de errores | Volumen (aumenta monótonamente) |
| **Medidores** | Uso de memoria actual, profundidad de cola | Estado actual (sube/baja) |
| **Histogramas** | Latencia (p50, p95, p99), valores de predicción | Distribución en el tiempo |
| **Tasas** | Solicitudes/segundo, tasa de error, rendimiento | Velocidad (derivada del contador) |

**El método USE**: para cada recurso, monitorear:
- **U**tilización: ¿qué tan ocupado está?
- **S**aturación: ¿cuánto trabajo pendiente?
- **E**rrores: ¿cuántos fallos?

**El método RED**: para cada solicitud, monitorear:
- **R**ate: solicitudes por segundo
- **E**rrores: solicitudes fallidas por segundo
- **D**uración: distribución de latencia

### 2.3 Trazado

Seguir una sola solicitud a través de múltiples servicios.

```
Gateway → Auth → Feature Store → Model Server → Respuesta
  2ms      5ms       12ms            45ms         3ms
```

**Por qué el trazado importa para ML**: una predicción lenta puede no ser culpa del modelo — podría ser el feature store, el servicio de embeddings o la caché. El trazado te dice dónde se fue el tiempo.

**OpenTelemetry**: el estándar para trazado distribuido. Instrumenta una vez, exporta a cualquier backend.

---

## 3. Observabilidad Específica de ML

Combinar con [[Experiment Tracking]] para correlacionar el comportamiento de predicción con ejecuciones de entrenamiento específicas.

### 3.1 Monitoreo de Predicciones

Más allá de la salud del sistema, monitorear la **calidad y el comportamiento** de las predicciones:

| Qué Monitorear | Cómo |
|---|---|
| **Distribución de predicciones** | Histograma de puntuaciones a lo largo del tiempo |
| **Puntuaciones de confianza** | ¿Las predicciones se están volviendo menos seguras? |
| **Valores de características** | Distribución de cada característica de entrada |
| **Versión del modelo** | Qué versión sirvió cada predicción |
| **Casos de error** | Qué entradas hacen que el modelo falle |

### 3.2 Monitoreo del Pipeline de Datos

| Qué | Alertar Cuando |
|---|---|
| **Frescura de datos** | Última ejecución exitosa del pipeline > 2 horas |
| **Conteo de filas** | Significativamente diferente de lo esperado |
| **Tasas de nulos** | Pico repentino en valores faltantes |
| **Violaciones de esquema** | Tipos o nombres de columna inesperados |

---

## 4. Estrategia de Alertas

### 4.1 Diseño de Alertas

Una buena alerta:
- **Accionable**: alguien puede hacer algo al respecto
- **Oportuna**: suficientemente temprano para prevenir el impacto
- **Específica**: te dice qué está mal
- **Sin ruido**: no se dispara innecesariamente

Configura alertas para experimentos de [[A-B Testing]] para detectar regresiones temprano.

### 4.2 Niveles de Severidad

| Nivel | Respuesta | Ejemplo |
|---|---|---|
| **Página (P0)** | Inmediata (5 min) | Modelo devolviendo 50% de errores |
| **Ticket (P1)** | Dentro de 1 hora | Latencia P95 excede el umbral |
| **Slack (P2)** | Mismo día | Declive gradual de precisión |
| **Dashboard (P3)** | Revisión semanal | Deriva menor de características |

---

## 5. Herramientas

| Dominio | Herramienta |
|---|---|
| **Métricas** | Prometheus + Grafana |
| **Logging** | ELK (Elasticsearch, Logstash, Kibana), Loki |
| **Trazado** | Jaeger, OpenTelemetry |
| **Específico de ML** | WhyLabs, Evidently, Arize |

---

## 6. Common Mistakes

1. **Registrar todo**: demasiados datos es tan malo como muy pocos. Registra lo que sea accionable. Archiva logs antiguos.

2. **Sin IDs de correlación**: sin un ID de solicitud, no puedes conectar logs, métricas y trazas del mismo evento.

3. **Fatiga de alertas**: demasiadas alertas → las alertas se ignoran. Cada alerta debería requerir una acción específica.

4. **Monitorear código pero no datos**: el código funciona bien, pero los datos son incorrectos. Monitorea la calidad de los datos por separado de la salud del sistema.

5. **Sin paneles**: un panel de Grafana que nunca se mira es esfuerzo desperdiciado. Construye paneles para audiencias específicas (guardia, líder de equipo, negocio).

---

## 7. Check Your Understanding

1. Un usuario reporta una respuesta lenta. ¿Cómo lo diagnosticas? (Encontrar el ID de solicitud → trazar a través de servicios → identificar el cuello de botella.)

2. ¿Cuál es la diferencia entre un contador y un medidor? (Contador: aumenta monótonamente (total de solicitudes). Medidor: fluctúa (memoria actual).)

3. ¿Por qué registrar en JSON estructurado en lugar de texto plano? (Consultable — puedes buscar campos específicos sin parsear.)

4. Tu alerta "Latencia P99 > 1s" se dispara a las 3 AM. El ingeniero de guardia verifica y encuentra un pico único causado por un trabajo batch. ¿Qué está mal? (La alerta es demasiado sensible — añade una ventana de duración o excluye horarios de batch.)

5. La precisión de un modelo cae pero todas las métricas del sistema (latencia, tasa de error) son normales. ¿Qué falta? (Monitoreo específico de ML — la calidad de predicción no es capturada por las métricas del sistema.)

---

## 8. Resumen

Observability makes production ML systems understandable. Logging records events, metrics measure trends, and traces follow requests. ML-specific observability adds prediction and data quality monitoring. Alerts should be actionable, timely, and specific. The three pillars together let you detect, diagnose, and fix problems quickly.

---

## 9. Where to Go Next

- [[Model Monitoring]] — Monitoreo específico de ML en profundidad
- [[Data & Concept Drift]] — Monitoreo de calidad de datos
- [[ML Pipelines]] — Observabilidad para fallos en pipelines
