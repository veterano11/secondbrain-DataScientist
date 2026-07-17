---
tags: [llm, prompting, core]
status: growing
created: 2026-06-27
---

# Ingeniería de Prompts

## 1. Escenario de aprendizaje

Trabajas en una empresa que quiere usar un LLM para atender clientes automáticamente. El primer intento da respuestas genéricas e inconsistentes: a veces el modelo es demasiado técnico, otras veces inventa información. El problema no es el modelo — es cómo le estás pidiendo que responda. Aprender a comunicar la intención correcta a un LLM es la habilidad más rentable que puedes desarrollar: un buen prompt puede hacer que un modelo de 7B supere a uno de 70B con un prompt pobre.

Los LLMs son capaces de una amplia gama de tareas — traducción, codificación, razonamiento, creatividad — pero extraer el comportamiento correcto requiere **el prompt adecuado**. La ingeniería de prompts es la habilidad de comunicar la intención a un LLM de manera efectiva.

Importa porque un prompt bien diseñado puede hacer que un modelo de 7B supere a uno de 70B con un prompt pobre — entender la [[Transformer Architecture]] subyacente ayuda a explicar por qué. Es la forma más barata y rápida de mejorar las salidas de un LLM.

---

## 2. Diseño Básico de Prompts

### 2.1 La Anatomía de un Prompt

```
[Mensaje de Sistema] — establece el rol, tono, restricciones
[Ejemplos Few-shot] — demuestra el comportamiento deseado
[Consulta del Usuario] — la solicitud real
[Respuesta del Asistente] — salida del modelo
```

### 2.2 Prompts de Sistema

El prompt de sistema define la personalidad del modelo y las restricciones de comportamiento:

```
Eres un científico de datos experto. Responde preguntas técnicas con claridad y proporciona ejemplos de código cuando sea relevante. Si no estás seguro de algo, dilo — no alucines.
```

Un buen prompt de sistema es específico sobre:
- **Rol**: quién es el modelo (experto, asistente, crítico)
- **Tono**: formal, casual, educativo, conciso
- **Restricciones**: qué evitar (alucinaciones, respuestas vagas)
- **Formato de salida**: markdown, JSON, viñetas

### 2.3 Zero-Shot vs Few-Shot

**Zero-shot**: darle al modelo una tarea sin ejemplos.

```
Traduce al español: "Hello, how are you?"
```

**Few-shot**: dar 1-3 ejemplos del patrón entrada-salida deseado.

```
Inglés: "Hello" → Español: "Hola"
Inglés: "Good morning" → Español: "Buenos días"
Inglés: "Thank you" → Español: "Gracias"
Inglés: "How are you?" → Español:
```

Few-shot es más confiable para formatos o tareas inusuales, funcionando de manera similar al [[Supervised Learning]] con un puñado de ejemplos etiquetados.

---

## 3. Técnicas Avanzadas de Prompting

### 3.1 Cadena de Pensamiento (Chain-of-Thought, CoT)

**Idea clave**: pedirle al modelo que razone paso a paso antes de responder.

**Sin CoT**:
```
P: Una tienda tiene 20 manzanas y vende 3 por día. ¿Cuántas quedan después de 5 días?
R: 5
(Incorrecto — el modelo adivinó)
```

**Con CoT**:
```
P: Una tienda tiene 20 manzanas y vende 3 por día. ¿Cuántas quedan después de 5 días?
R: Pensemos paso a paso.
1. Manzanas iniciales: 20
2. Vendidas por día: 3
3. Total vendidas después de 5 días: 3 × 5 = 15
4. Restantes: 20 - 15 = 5
Respuesta: 5
```

CoT mejora drásticamente el rendimiento en tareas de matemáticas, lógica y razonamiento de múltiples pasos. Funciona porque:
- Externaliza el proceso de razonamiento
- El modelo puede corregir sus propios pasos intermedios
- Descompone un problema difícil en subproblemas más fáciles

### 3.2 Autoconsistencia (Self-Consistency)

Ejecuta el mismo prompt múltiples veces con temperatura > 0, luego toma la respuesta mayoritaria. Esto mejora la fiabilidad en tareas de razonamiento porque:
- Diferentes caminos de razonamiento deberían converger a la misma respuesta
- Los errores aleatorios de un camino son superados por votación

### 3.3 Árbol de Pensamientos (Tree-of-Thoughts)

A diferencia de CoT (una cadena), Tree-of-Thoughts explora múltiples caminos de razonamiento simultáneamente:

```
Camino 1: A → B → C → D → Resultado
Camino 2: A → B → E → F → Resultado
Camino 3: A → G → H → Resultado
```

En cada paso, el modelo evalúa qué caminos son prometedores y poda el resto. Más costoso pero más potente que CoT.

### 3.4 ReAct (Razonamiento + Acción)

Combina razonamiento con uso de herramientas:

```
Pensamiento: Necesito encontrar la población de Tokio.
Acción: buscar("población de Tokio")
Observación: 14 millones (2023)
Pensamiento: Ahora puedo responder la pregunta.
Acción: respuesta("Tokio tiene 14 millones de habitantes.")
```

El modelo genera pensamientos, acciones y observa resultados en un bucle. Esta es la base de los **sistemas agentivos**.

---

## 4. Control de Salida

### 4.1 Salida Estructurada

Forzar al modelo a producir una salida procesable:

```json
Eres un extractor de datos. Dado un texto, genera JSON con:
- "fecha": la fecha mencionada (YYYY-MM-DD o null)
- "entidades": lista de personas mencionadas
- "resumen": resumen de una oración

Texto: "El 27 de junio de 2026, Alice y Bob publicaron sus hallazgos."

Salida:
{
  "fecha": "2026-06-27",
  "entidades": ["Alice", "Bob"],
  "resumen": "Alice y Bob publicaron hallazgos el 27 de junio de 2026."
}
```

### 4.2 Control de Verbosidad

```
Sé conciso. Responde en 2-3 oraciones como máximo.
```

vs.

```
Proporciona un análisis detallado con ejemplos y contraejemplos.
```

### 4.3 Mitigación de Alucinaciones

**Técnicas a nivel de prompt**:
- "Usa solo información del contexto a continuación."
- "Si no estás seguro, di 'No lo sé.'"
- "Cita fuentes específicas para cada afirmación."

Para sistemas RAG, el prompt debe indicar explícitamente al modelo que prefiera el contexto recuperado sobre el conocimiento paramétrico. Las técnicas de [[Model Evaluation]] pueden ayudar a medir si esta guía es efectiva.

---

## 5. Patrones de Prompts

### 5.1 Patrón de Persona

```
Actúa como un científico de datos senior revisando el código de un colega.
```

### 5.2 Patrón de Plantilla

```
Completa el siguiente análisis:

Análisis de Rendimiento del Modelo:
- Precisión: {nombre_modelo} alcanzó {precision}% en {dataset}
- Mejor característica: {mejor_característica} con importancia {importancia}
- Recomendación: {recomendacion}
```

### 5.3 Patrón de Cadena

Encadena múltiples prompts donde cada salida alimenta la siguiente:

```
Paso 1: "Genera 5 hipótesis sobre por qué aumentó la rotación de clientes."
Paso 2: "Para cada hipótesis, propón una forma de probarla con datos."
Paso 3: "Dados los resultados de la prueba a continuación, ¿qué hipótesis tiene más respaldo?"
```

---

## 6. Common Mistakes

1. **Prompts excesivamente complejos**: un prompt con muchas instrucciones confunde al modelo. Prioriza las reglas más importantes.

2. **Asumir que el modelo recuerda el contexto de la conversación**: después de ~8K-128K tokens, el modelo pierde el contexto inicial. Para conversaciones largas, resume periódicamente.

3. **No probar sistemáticamente**: lo que funciona una vez puede no funcionar consistentemente. Prueba los prompts con múltiples entradas y diferentes temperaturas, usando [[Experiment Tracking]] para registrar resultados sistemáticamente.

4. **Demasiadas restricciones**: "sé conciso, pero completo, pero usa ejemplos, pero no seas demasiado técnico" — las instrucciones conflictivas degradan el rendimiento.

5. **Descuidar los prompts de sistema**: el mensaje de sistema es la herramienta más potente para establecer el comportamiento. Úsalo explícitamente.

---

## 7. Check Your Understanding

1. Le preguntas a un modelo "¿Cuánto es 23 × 47?" y obtienes una respuesta incorrecta. ¿Cómo reescribirías el prompt para mejorar la precisión?

2. Chain-of-Thought mejora el razonamiento matemático. ¿Por qué funciona? (Descompone el problema, externaliza el razonamiento, permite la autocorrección.)

3. Tu sistema RAG recupera documentos relevantes pero el modelo los ignora. ¿Qué cambios de prompt harías?

4. Necesitas salida JSON pero el modelo ocasionalmente añade comentarios. ¿Cómo lo solucionarías?

5. Self-Consistency ejecuta múltiples muestras. ¿Cuándo valdría la pena el costo computacional?

---

## 8. Resumen

La ingeniería de prompts es la habilidad de comunicar intención a los LLMs. Un buen prompt especifica rol, tono, formato y restricciones. Técnicas avanzadas como Cadena de Pensamiento y ReAct mejoran el razonamiento y el uso de herramientas. La salida estructurada (JSON) permite el procesamiento posterior. La clave es la prueba sistemática — varía los prompts, mide resultados, itera.

---

## 9. Dónde Ir Ahora

- [[RAG]] — Combinando prompts con contexto recuperado
- [[Sistemas Agentivos]] — Razonamiento multi-paso con uso de herramientas
- [[Evaluación de LLMs]] — Midiendo la efectividad de los prompts
