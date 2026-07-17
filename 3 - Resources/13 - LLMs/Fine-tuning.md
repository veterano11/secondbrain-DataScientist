---
tags: [llm, fine-tuning, advanced]
status: growing
created: 2026-06-27
---

# Fine-tuning

## 1. Escenario de aprendizaje

Tienes un modelo base como Llama 3 que sabe de todo un poco pero necesitas que genere informes médicos con terminología especializada y un formato específico de tu hospital. El modelo genérico produce texto correcto pero no útil — usa lenguaje impreciso y omite secciones críticas. El fine-tuning te permite adaptar el modelo a tu dominio, transformando un modelo genérico en una herramienta especializada.

Los LLMs pre-entrenados son generalistas — saben un poco de todo pero no sobresalen en nada específico. El fine-tuning los adapta a un dominio, tarea o comportamiento particular — la idea central detrás de [[Transfer Learning]]. Así es como conviertes un modelo genérico en una herramienta especializada.

El fine-tuning es la diferencia entre un modelo que "sabe acerca de" tu código y uno que "escribe código en tu estilo." Entre un modelo que "entiende" terminología médica y uno que "diagnostica" correctamente a partir de informes de radiología.

---

## 2. El Espectro del Fine-Tuning

### 2.1 Fine-Tuning Completo

Actualizar todos los parámetros. Potente pero costoso.

**Memoria**:
- Pesos del modelo: 70B × 2 bytes (BF16) = 140 GB
- Estados del optimizador (Adam): 70B × 4 × 4 bytes = 1.12 TB
- Gradientes: 70B × 4 bytes = 280 GB
- **Total**: ~1.5 TB por GPU — imposible. Distribuido entre GPUs.

**Cuándo usarlo**: conjunto de datos objetivo grande, datos de alta calidad, presupuesto de cómputo suficiente.

### 2.2 Fine-Tuning Eficiente en Parámetros (PEFT)

Actualizar una fracción minúscula de parámetros. Casi tan bueno como el fine-tuning completo para la mayoría de las tareas.

#### LoRA (Low-Rank Adaptation)

$$W' = W + BA, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$$

**Paso a paso**:
1. Congela la matriz de pesos original $W$
2. Añade dos matrices pequeñas $A$ y $B$ con rango $r$ (típicamente 8-64)
3. Solo se entrenan $A$ y $B$
4. Después del entrenamiento, $BA$ puede fusionarse en $W$ (sin sobrecarga en inferencia)

**Por qué funciona**: las actualizaciones de pesos durante el fine-tuning tienen "rango intrínseco" bajo — los cambios reales están en un subespacio de baja dimensión.

**Memoria**: un modelo de 7B con LoRA se entrena en una sola GPU con 24GB de memoria.

#### QLoRA

LoRA + cuantización de 4 bits del modelo base. Permite fine-tuning de un modelo de 70B en una sola GPU de 48GB.

**Técnicas**:
- Cuantización NF4 (normal float 4-bit)
- Doble cuantización (cuantizar las constantes de cuantización)
- Optimizadores paginados (descarga a CPU cuando se excede la memoria GPU)

#### Adaptadores

Insertar pequeñas capas de cuello de botella entre bloques transformer:

```
Entrada → Adaptador(abajo) → ReLU → Adaptador(arriba) → Salida
         d → r               r → d
```

Solo los parámetros del adaptador se entrenan. Menos eficiente que LoRA (añade latencia en inferencia).

#### Prefix Tuning

Aprender "tokens virtuales" antepuestos a las claves/valores de cada capa. No se añaden pesos nuevos, solo embeddings aprendidos.

---

## 3. Fine-tuning por Instrucciones

### 3.1 El Formato

```
{
  "instruction": "Traduce al francés",
  "input": "Hola, ¿cómo estás?",
  "output": "Bonjour, comment allez-vous?"
}
```

El modelo aprende a seguir instrucciones en el formato visto durante el entrenamiento. Así es como los modelos base se convierten en modelos de chat/asistente.

### 3.2 Plantilla de Chat

Los modelos usan plantillas de chat específicas que estructuran la conversación:

```python
# Plantilla de chat de Llama 3
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Eres un asistente útil.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
¿Qué es ML?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
El machine learning es...
```

Usa siempre la plantilla de chat correcta del modelo. Usar la plantilla incorrecta degrada el rendimiento significativamente.

---

## 4. Calidad de los Datos

### 4.1 Cantidad vs Calidad

100 ejemplos de alta calidad > 10,000 ejemplos ruidosos.

**Criterios de calidad**:
- **Correcto**: la salida es factualmente correcta
- **Consistente**: sigue el formato y estilo esperados
- **Diverso**: cubre el rango de entradas que el modelo verá
- **No tóxico**: sin contenido dañino en las salidas

### 4.2 Preparación de Datos

1. **Recolectar**: recopilar ejemplos del comportamiento deseado
2. **Limpiar**: eliminar duplicados, corregir formato, verificar corrección
3. **Formatear**: aplicar la plantilla de chat apropiada
4. **Dividir**: entrenamiento (90%), validación (10%)
5. **Deduplicar**: eliminar casi duplicados (basado en LLM o similitud de embeddings)

---

## 5. RLHF — Reinforcement Learning from Human Feedback

### 5.1 El Problema

El fine-tuning por instrucciones enseña al modelo QUÉ hacer, pero no CÓMO comportarse. [[RLHF & Preference Optimization]] alinea el modelo con las preferencias humanas.

### 5.2 Los Tres Pasos

**Paso 1: Supervised Fine-Tuning (SFT)**
- Fine-tuning en demostraciones humanas de alta calidad — una forma de [[Supervised Learning]]
- Enseña al modelo el formato y estilo deseados

**Paso 2: Entrenamiento del Modelo de Recompensa**
- Para cada prompt, generar múltiples salidas del modelo SFT
- Humanos clasifican las salidas (comparaciones por pares)
- Entrenar un modelo de recompensa para predecir preferencias humanas

**Paso 3: PPO (Proximal Policy Optimization)**
- Usar el modelo de recompensa para puntuar las salidas del LLM
- Optimizar el LLM para maximizar la recompensa
- Añadir penalización KL para evitar que el modelo se aleje demasiado del modelo SFT

### 5.3 Alternativas a RLHF

| Método | Descripción | Pros | Contras |
|---|---|---|---|
| **DPO** | Direct Preference Optimization | Más simple, sin modelo de recompensa | Puede no escalar igual |
| **ORPO** | SFT + alineación combinados | Una sola etapa | Nuevo, menos probado |
| **KTO** | Kahneman-Tversky Optimization | Solo necesita feedback binario | Menos detallado |

DPO es la alternativa más popular: optimiza directamente la política en pares de preferencia sin entrenar un modelo de recompensa separado.

---

## 6. Lista de Verificación Práctica

```
☐ Datos: mínimo 100 ejemplos de alta calidad
☐ Formato: plantilla de chat correcta
☐ Rango r: 8-64 (más alto para tareas más diversas)
☐ Módulos objetivo: q_proj, v_proj (común), o todas las capas lineales
☐ LR: 1e-4 a 5e-4 (LoRA), 1e-5 a 5e-5 (completo)
☐ Tamaño de lote: acumulación de gradientes a 64-128 muestras
☐ Épocas: 1-3 (más = riesgo de sobreajuste)
☐ Evaluación: conjunto de validación reservado
☐ Monitorear: curvas de pérdida, calidad de generación en validación — usa [[Experiment Tracking]] para registrar estas métricas
```

---

## 7. Common Mistakes

1. **Demasiadas épocas**: los LLMs se sobreajustan rápidamente. 1-3 épocas suele ser suficiente. Más épocas perjudican la generalización.

2. **Plantilla de chat incorrecta**: usar la plantilla equivocada causa salidas incoherentes. Verifica que el formato coincida exactamente.

3. **Olvidar fusionar los pesos de LoRA para despliegue**: los pesos de LoRA deben fusionarse o cargarse por separado en inferencia.

4. **Datos de baja calidad**: basura entra, basura sale. El fine-tuning amplifica los patrones en los datos de entrenamiento — incluyendo errores.

5. **No evaluar antes/después**: si no puedes medir la mejora, no sabes si el fine-tuning ayudó. Evalúa siempre en un conjunto de prueba reservado.

---

## 8. Check Your Understanding

1. El rango r=8 de LoRA usa ¿qué fracción de una matriz de pesos 4096×4096? (8×4096 + 4096×8 = 65K de 4096² ≈ 16.8M → 0.4%)

2. ¿Por qué QLoRA permite fine-tuning de un modelo de 70B en una sola GPU? (La cuantización de 4 bits reduce la memoria del modelo base en 4×.)

3. Fine-tuning en 500 ejemplos por 10 épocas. La pérdida de entrenamiento es casi cero pero la perplejidad de validación es peor que antes. ¿Qué sucedió? (Sobreajuste.)

4. ¿Para qué sirve la penalización KL en PPO? (Evita que el modelo se aleje demasiado del modelo SFT y pierda capacidades generales.)

5. DPO no requiere un modelo de recompensa. ¿Cómo optimiza las preferencias humanas? (Optimiza directamente la política en pares de preferencia usando una pérdida similar a la entropía cruzada binaria.)

---

## 9. Resumen

Fine-tuning adapts LLMs to specific tasks. Full fine-tuning is powerful but expensive. PEFT methods (LoRA, QLoRA) achieve similar results with 100× less memory. Instruction fine-tuning teaches task following. RLHF and DPO align models with human preferences. Data quality matters more than quantity. Start with 100-1000 high-quality examples, use LoRA, and evaluate rigorously.

---

## 10. Where to Go Next

- [[RAG]] — Una alternativa al fine-tuning para tareas intensivas en conocimiento
- [[Prompt Engineering]] — La forma más simple de adaptación de tareas
- [[Training Techniques]] — Trucos avanzados de entrenamiento para fine-tuning
