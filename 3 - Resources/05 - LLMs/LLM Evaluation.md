---
tags: [llm, evaluation, advanced]
status: growing
created: 2026-06-27
---

# Evaluación de LLMs

## 1. Escenario de aprendizaje

Tu equipo ha fine-tuneado un modelo para atención al cliente y necesitas decidir si reemplaza al modelo actual. La precisión mejoró 2% en tus pruebas, pero ¿eso significa que será mejor con usuarios reales? ¿Y si el nuevo modelo es menos seguro? ¿O si alucina más cuando no sabe la respuesta? Evaluar un LLM no es mirar una sola métrica — es entender sus fortalezas y debilidades en múltiples dimensiones antes de arriesgar la experiencia de tus usuarios.

"¿Qué tan bueno es este LLM?" no es una pregunta simple. Un modelo que sobresale en matemáticas puede fallar en escritura creativa. Uno que sigue instrucciones perfectamente puede alucinar libremente. La evaluación de LLMs es la práctica de medir capacidades específicas para entender lo que un modelo puede y no puede hacer.

Sin evaluación — la base de [[Model Evaluation]] — no puedes:
- Elegir entre modelos para tu aplicación
- Saber si el fine-tuning mejoró el modelo
- Detectar regresiones después de actualizaciones del modelo
- Entender dónde fallará tu aplicación

---

## 2. Benchmarks Estándar

### 2.1 Conocimiento y Razonamiento

| Benchmark | Qué Prueba | Formato |
|---|---|---|
| **MMLU** | 57 materias (STEM, humanidades, ciencias sociales) | Preguntas de 4 opciones |
| **MMLU-Pro** | MMLU más difícil (más opciones, preguntas más complejas) | Preguntas de 10 opciones |
| **GPQA** | Ciencia a nivel de posgrado | Preguntas escritas por expertos |
| **ARC** | Ciencia escolar | Opción múltiple |
| **HellaSwag** | Razonamiento de sentido común | Completar oraciones |

### 2.2 Matemáticas y Codificación

| Benchmark | Qué Prueba | Formato |
|---|---|---|
| **GSM8K** | Problemas matemáticos escolares | Respuesta libre |
| **MATH** | Matemáticas de competencia (AMC, AIME) | Respuesta libre |
| **HumanEval** | Completar funciones en Python | Pass@k (corrección funcional) |
| **MBPP** | Programación básica en Python | Pass@k |
| **SWE-bench** | Ingeniería de software real (issues de GitHub) | Corrección de parches |

### 2.3 Lenguaje y Diálogo

| Benchmark | Qué Prueba |
|---|---|
| **MT-Bench** | Capacidad conversacional multi-turno (evaluado por LLM-as-judge) |
| **Chatbot Arena** | Rankings de preferencia humana (sistema ELO, 1M+ votos humanos) |
| **TruthfulQA** | Veracidad (preguntas adversariales que provocan creencias falsas) |
| **AlpacaEval** | Seguimiento de instrucciones (comparado con modelo de referencia) |

---

## 3. Evaluación Automatizada

### 3.1 LLM-as-Judge

Usar un LLM para evaluar las salidas de otro LLM:

```python
prompt = f"""
Estás evaluando la respuesta de un asistente de IA.

[Pregunta]
{pregunta}

[Respuesta del Asistente]
{respuesta}

Evalúa en una escala del 1-5 para:
1. Utilidad: ¿Aborda la pregunta del usuario?
2. Precisión: ¿La información es correcta?
3. Inocuidad: ¿Evita contenido dañino?

Salida JSON: {{"utilidad": int, "precision": int, "inocuidad": int}}
"""
```

**Desafíos**:
- Los jueces tienen sesgos (prefieren respuestas más largas, están de acuerdo consigo mismos)
- Sesgo de posición (prefieren la primera o última respuesta en una comparación)
- Sesgo de automejora (prefieren modelos similares a sí mismos)

**Mitigaciones**:
- Usar un modelo diferente y confiable como juez (ej., GPT-4 evalúa a Llama)
- Aleatorizar el orden de las respuestas en comparaciones
- Usar puntuación multidimensional (puntuaciones separadas para diferentes aspectos)
- Calibrar jueces contra evaluaciones humanas

### 3.2 Métricas

| Métrica | Qué Mide | Limitaciones |
|---|---|---|
| **Perplejidad** | Qué tan bien predice el modelo el siguiente token | No correlacionada con calidad de salida para modelos de chat |
| **ROUGE** | Solapamiento de n-gramas con referencia | Superficial — ignora calidad semántica |
| **BLEU** | Precisión de n-gramas (traducción) | Pobre para tareas creativas/abstractivas |
| **BERTScore** | Similitud de embeddings con referencia | Mejor que ROUGE/BLEU, aún depende de referencia |
| **Perplejidad de salidas** | Qué tan naturales son las generaciones | Mide fluidez, no corrección |

---

## 4. Detección de Alucinaciones

### 4.1 Tipos de Alucinación

| Tipo | Descripción | Ejemplo |
|---|---|---|
| **Factual** | Afirmación falsa de un hecho | "Einstein inventó internet" |
| **Intrínseca** | Contradice el contexto proporcionado | Modelo RAG ignorando documentos recuperados |
| **Extrínseca** | Añade información no presente en el contexto | "El informe dice X" — el informe no dice nada |
| **Lógica** | Internamente inconsistente | "Nací en 1990 y tengo 40" (debería ser 34) |

### 4.2 Métodos de Detección

- **SelfCheckGPT**: generar múltiples respuestas, verificar consistencia
- **Basado en NLI**: usar un modelo NLI para verificar si cada afirmación está implicada por el contexto
- **Estimación de confianza**: las probabilidades de token del propio modelo (más bajas → más probable alucinación)
- **Validación de citas**: para RAG, verificar que cada cita apunte a un fragmento que respalde la afirmación

---

## 5. Evaluación Humana

### 5.1 Cuándo Usar Evaluación Humana

- Evaluación final antes del despliegue
- Cuando las métricas automatizadas no son fiables (tareas creativas)
- Calibración de jueces automatizados
- Detección de sesgos sutiles o problemas de seguridad

### 5.2 Protocolos Comunes

| Protocolo | Costo | Fiabilidad |
|---|---|---|
| **Cara a cara (A/B)** | Bajo | Buena — comparando dos salidas |
| **Escala Likert** | Medio | Moderada — escala subjetiva |
| **Ranking por pares (Elo)** | Alto | Excelente — elimina sesgos |
| **Chatbot Arena** | Muy alto | Excelente — 1M+ juicios humanos |

---

## 6. Evaluación del Fine-Tuning

Siempre compara ANTES y DESPUÉS usando [[Experiment Tracking]] para registrar resultados sistemáticamente:

```python
# Antes
puntuacion_base = evaluar(modelo_base, conjunto_prueba)

# Después
puntuacion_ft = evaluar(modelo_ajustado, conjunto_prueba)

# ¿Mejoró?
print(f"Δ = {puntuacion_ft - puntuacion_base:+.3f}")
```

**Métricas clave para evaluar fine-tuning**:
- Precisión en la tarea (la métrica para la que optimizaste)
- Retención de capacidades generales (MMLU, HellaSwag — ¿el fine-tuning degradó habilidades generales?)
- Cumplimiento del formato de salida (% de salidas en el formato correcto)
- Rendimiento en el peor caso (¿hay entradas donde el modelo retrocedió?)

---

## 7. Common Mistakes

1. **Evaluar solo una dimensión**: precisión sin fluidez, o fluidez sin veracidad. La cobertura de múltiples capacidades es esencial.

2. **Usar los mismos datos de prueba para desarrollo**: si ajustas hiperparámetros basándote en MMLU, MMLU ya no es una evaluación imparcial. Usa un conjunto separado reservado, una práctica estándar en [[Supervised Learning]].

3. **Ignorar la varianza**: una mejora del 0.5% en MMLU puede no ser estadísticamente significativa. Reporta intervalos de confianza (una mejor práctica de [[Statistics]]) o ejecuta múltiples pruebas.

4. **Evaluar solo métricas agregadas**: un modelo puede puntuar bien en general pero fallar catastróficamente en categorías específicas (ej., seguridad, matemáticas, no inglés).

5. **Confiar demasiado en LLM-as-Judge**: los jueces automatizados son útiles pero sesgados. Valida con evaluación humana para decisiones críticas.

---

## 8. Check Your Understanding

1. Un modelo obtiene 90% en MMLU pero 30% en TruthfulQA. ¿Qué te dice esto? (Alto conocimiento pero baja veracidad — no fiable factualmente.)

2. ¿Por qué la perplejidad no se correlaciona bien con la calidad de chat? (La perplejidad mide predicción del siguiente token, no calidad de salida. Un modelo muy conservador que dice "No lo sé" puede tener baja perplejidad pero ser inútil.)

3. Tu modelo fine-tuneado puntúa mejor en tu tarea pero peor en MMLU. ¿Deberías desplegarlo? (Depende — si el rendimiento en la tarea supera la degradación general, sí. Monitorea casos extremos.)

4. Dos anotadores dan puntuaciones diferentes a la misma salida. ¿Qué haces? (Calcular el acuerdo entre anotadores. Los desacuerdos pueden indicar rúbricas poco claras.)

5. Un LLM-as-Judge prefiere consistentemente respuestas más largas. ¿Cómo mitigas esto? (Controlar por longitud comparando respuestas de longitud similar o usando puntuación calibrada por longitud.)

---

## 9. Summary

LLM evaluation is multi-dimensional: knowledge, reasoning, coding, truthfulness, safety, and instruction following. Use a mix of standard benchmarks (MMLU, HumanEval, MT-Bench) for general capability, automated judges for task-specific evaluation, and human evaluation for final assessment. Always compare against a baseline, measure multiple dimensions, and be aware of measurement biases (judge preferences, position bias, variance).

---

## 10. Where to Go Next

- [[Prompt Engineering]] — La calidad de los prompts afecta la evaluación
- [[Fine-tuning]] — Evaluando antes y después del fine-tuning
- [[RAG]] — Evaluando recuperación y generación por separado
