---
tags: [mlops, experimentation, advanced]
status: growing
created: 2026-06-27
---

# Pruebas A/B

## 1. Escenario de aprendizaje

Tu equipo desarrolló un nuevo modelo de recomendación que muestra una mejora del 1% en precisión offline. ¿Deberías desplegarlo en producción? La respuesta no es obvia — esa diferencia podría ser ruido estadístico, o el nuevo modelo podría tener fallos ocultos que solo aparecen con usuarios reales. Las pruebas A/B te permiten comparar dos modelos con tráfico real y tomar decisiones basadas en datos, no en intuición.

Construiste un nuevo modelo. Obtiene un 1% más de precisión que el modelo actual en producción. ¿Deberías desplegarlo? La respuesta no es obvia — la diferencia podría ser ruido, o el nuevo modelo podría tener modos de fallo ocultos.

Las pruebas A/B son cómo **comparas de manera fiable dos modelos (o dos estrategias) en producción** con usuarios reales. Es el estándar para tomar decisiones de despliegue basadas en datos.

---

## 2. Diseño de Pruebas A/B

### 2.1 Configuración Básica

- **Control (A)**: modelo actual en producción
- **Tratamiento (B)**: nuevo modelo a evaluar
- **Métrica**: lo que mides (tasa de conversión, precisión, ingresos)
- **Unidad de aleatorización**: quién recibe cada modelo (usuario, sesión, evento)

Los usuarios se asignan aleatoriamente a A o B. Su experiencia difiere solo en qué modelo les sirve.

Valida siempre los diseños de experimentos con [[Experiment Tracking]] para registrar parámetros y resultados de la prueba.

### 2.2 Tamaño de Muestra

$$n = \frac{(Z_{\alpha/2} + Z_\beta)^2 \cdot 2\sigma^2}{\delta^2}$$

- $\delta$: efecto mínimo que quieres detectar (MDE)
- $\alpha$: nivel de significancia (por defecto 0.05)
- $\beta$: tasa de error Tipo II (por defecto 0.2, potencia = 0.8)
- $\sigma^2$: varianza de la métrica

**Ejemplo concreto** — detectar una mejora del 1% en la tasa de conversión:
- CTR base: 5% → $\sigma^2 \approx 0.05 \times 0.95 \approx 0.048$
- MDE: 1% absoluto (5% → 6%)
- $\alpha = 0.05$, $\beta = 0.2$
- $n \approx 3,500$ por grupo

Si tienes 10,000 usuarios/día, la prueba dura ~1 día. Si tienes 500 usuarios/día, dura ~2 semanas.

### 2.3 Duración

**Duración mínima**: al menos un ciclo completo de negocio (1 semana mínimo para capturar patrones semanales).

**Regla de parada**: NO detengas la prueba antes cuando los resultados parezcan significativos — la parada temprana infla los falsos positivos.

**Problema de ojeo**: verificar los resultados cada día y detenerse cuando p < 0.05 hace que la tasa de error Tipo I real sea mucho mayor del 5%.

---

## 3. Métricas

| Tipo de Métrica | Ejemplos | Notas |
|---|---|---|
| **Primaria** | Tasa de conversión, ingresos por usuario | La que optimizas |
| **Secundaria** | Tasa de clics, duración de sesión | Contexto adicional |
| **Barrera** | Latencia, tasa de error, tickets de soporte | No debe degradarse |
| **Segmento** | Métricas por tipo de usuario, región | Detectar efectos locales |

Define siempre **métricas de barrera** — cosas que no deben empeorar. Un modelo podría mejorar la conversión pero aumentar la latencia a 10 segundos (destruyendo la experiencia del usuario).

---

## 4. Pruebas Estadísticas

| Tipo de Dato | Prueba | Ejemplo |
|---|---|---|
| **Binario** (¿hizo clic?, ¿convirtió?) | Z-test para proporciones | Comparación de CTR |
| **Continuo** (ingresos, tiempo) | t-test | Valor promedio del pedido |
| **Conteo** (compras por usuario) | Poisson / binomial negativa | Frecuencia de compra |
| **Rango** (calificaciones, preferencias) | Mann-Whitney U | Puntuaciones de satisfacción |

### 4.1 Comparaciones Múltiples

Si pruebas 10 métricas con $\alpha = 0.05$, tienes ~40% de probabilidad de al menos un falso positivo.

**Correcciones**:
- **Bonferroni**: dividir $\alpha$ por número de pruebas (conservador)
- **FDR** (False Discovery Rate): menos conservador, controla la proporción esperada de falsos positivos

---

## 5. Errores Comunes

| Error | Problema | Solución |
|---|---|---|
| **Ojeo** | Parada temprana infla falsos positivos | Pre-registrar duración |
| **Falta de observabilidad** | Ciego a la causa raíz del cambio métrico | Usar [[Observability]] para diagnosticar |
| **Efecto novedad** | Los usuarios interactúan más con cualquier cosa nueva | Ejecutar prueba el tiempo suficiente para que la novedad desaparezca |
| **Efectos de red** | El tratamiento afecta al control (ej., plataforma social) | Aleatorización por conglomerados |
| **Sesgo de selección** | Asignación no aleatoria | Aleatorización adecuada |
| **Paradoja de Simpson** | El efecto general se invierte dentro de segmentos | Pre-registrar análisis por segmento |

---

## 6. Pruebas A/B Específicas de ML

### 6.1 Desafíos

- **Retroalimentación retrasada**: la predicción de default de préstamos tarda meses en validarse
- **No estacionariedad**: la estacionalidad afecta a ambos grupos
- **Interferencia**: las predicciones del modelo A afectan los datos del modelo B (ej., sistemas de recomendación)

### 6.2 Interleaving

Para modelos de ranking/recomendación, en lugar de A/B (cada usuario ve un modelo), **entremezcla** los resultados:

```
El usuario ve: Resultado Modelo A 1, Resultado Modelo B 1, Resultado Modelo A 2, Resultado Modelo B 2...
Clic en: Resultado Modelo B 1 → preferencia por Modelo B
```

Más sensible que A/B para tareas de ranking (detecta diferencias más pequeñas).

### 6.3 Multi-Armed Bandits

En lugar de una división fija 50/50, asigna dinámicamente más tráfico al modelo con mejor rendimiento. Integra la lógica de bandidos en [[ML Pipelines]] para despliegues automatizados de modelos:

```
Día 1: A=50%, B=50%  (exploración)
Día 2: A=45%, B=55%  (B es ligeramente mejor)
Día 3: A=30%, B=70%  (B es claramente mejor)
Día 7: A=5%,  B=95%  (B está desplegado)
```

Minimiza el costo de oportunidad durante las pruebas. Más complejo de analizar.

---

## 7. Common Mistakes

1. **Detenerse demasiado pronto**: "el p-valor es 0.04 después de 2 días" — sigue ejecutando. La parada temprana invalida la prueba.

2. **Ignorar la significancia práctica**: una mejora del 0.1% puede ser estadísticamente significativa (p < 0.05) pero no vale el costo de ingeniería del despliegue.

3. **No considerar la retroalimentación retrasada**: un modelo de préstamos parece mejor durante 6 meses, luego los defaults empiezan a aparecer.

4. **Comparar más de 2 variantes sin corrección**: comparar 5 modelos requiere corregir para comparaciones múltiples.

5. **Ejecutar demasiado tiempo sin verificar**: si una prueba se ejecuta durante meses, los usuarios pueden estar expuestos a un modelo peor innecesariamente. Usa bandidos para pruebas de larga duración.

---

## 8. Check Your Understanding

1. Tu prueba A/B ha estado ejecutándose durante 3 días. El p-valor es 0.03. ¿Deberías detenerte y declarar a B como ganador? (No — la parada temprana infla los falsos positivos. Espera hasta la duración pre-registrada.)

2. El modelo A aumenta la conversión en un 2% pero aumenta la latencia de 50ms a 2s. ¿Es A el ganador? (No — se violó la métrica de barrera.)

3. Pruebas 20 métricas y 1 muestra p < 0.05. ¿Es esto evidencia de un efecto real? (Quizás no — con 20 pruebas, 1 resultado significativo se espera por azar bajo la hipótesis nula.)

4. ¿Por qué interleaving detecta diferencias más pequeñas para modelos de ranking? (Cada usuario ve ambos modelos, controlando la varianza a nivel de usuario.)

5. Un multi-armed bandit asigna el 80% del tráfico al modelo B después de 1 semana. ¿Puedes confiar en el resultado? (Con precaución — la concentración temprana puede basarse en estimaciones ruidosas. Usa una tasa de exploración mínima.)

---

## 9. Resumen

A/B testing is how you reliably compare models in production. Design the test before running it: define the metric, calculate sample size, set duration. Use guardrail metrics to catch regressions. Avoid peeking (checking results early). For ranking tasks, consider interleaving. For minimizing opportunity cost, use multi-armed bandits. The key principle: pre-register the analysis plan and do not deviate.

---

## 10. Where to Go Next

- [[Model Evaluation]] — Evaluación offline para decidir qué probar online
- [[Statistics]] — Fundamentos de pruebas de hipótesis
- [[Model Monitoring]] — Monitoreo de modelos desplegados durante y después de pruebas A/B
- [[Feature Engineering]] — Diseño de características para consistencia en experimentos
