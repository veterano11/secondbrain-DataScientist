---
tags: [llm, agents, advanced]
status: growing
created: 2026-06-27
---

# Sistemas Agentivos

## 1. Escenario de aprendizaje

Necesitas un asistente que investigue competidores: que busque en la web sus últimos lanzamientos, lea los artículos, extraiga datos clave y genere un informe comparativo. Un LLM normal solo puede conversar contigo. Pero si le das herramientas — búsqueda web, un lector de páginas, una hoja de cálculo — puede orquestar todo el flujo de trabajo autónomamente. Eso es un sistema agentivo.

Un LLM por sí mismo es un generador de texto construido sobre la [[Transformer Architecture]]. Le das texto, te da texto. Pero cuando le das **herramientas** — búsqueda web, ejecución de código, consultas a bases de datos, acceso a archivos — se convierte en un **agente** que puede tomar acciones, observar resultados y planificar soluciones de múltiples pasos.

Los sistemas agentivos son la frontera más emocionante en los LLMs aplicados. Un agente bien diseñado puede responder preguntas que requieren investigación, escribir y depurar código, gestionar flujos de trabajo complejos (automatizando tareas de [[CLI & Productivity]]), y actuar autónomamente dentro de límites definidos.

---

## 2. El Bucle del Agente

### 2.1 Bucle Principal

```
1. Observar (entrada del usuario o salida de herramienta)
2. Pensar (razonar sobre qué hacer a continuación)
3. Actuar (responder o llamar a una herramienta)
4. Observar (resultado de la herramienta)
5. Repetir hasta que la tarea esté completa
```

### 2.2 Un Ejemplo Concreto

```
Usuario: "¿Cuál era el precio de la acción de NVIDIA el 1 de junio de 2026?"

Pensamiento del LLM: Necesito buscar datos del precio de la acción de NVIDIA.
Acción: buscar_web(consulta="precio acción NVIDIA 1 junio 2026")
Observación: "NVDA cerró a $185.42 el 1 de junio de 2026"

Pensamiento del LLM: Tengo la respuesta, debo responder.
Acción: respuesta("El precio de la acción de NVIDIA el 1 de junio de 2026 fue $185.42.")
```

El LLM decide cuándo usar herramientas y cuándo responder directamente.

---

## 3. Uso de Herramientas

### 3.1 Diseño de Herramientas

Cada herramienta necesita:
- **Nombre**: descriptivo y único
- **Descripción**: explica cuándo y cómo usar la herramienta
- **Parámetros**: esquema (JSON Schema) que define las entradas
- **Implementación**: la función real que se ejecuta

```python
tools = [
    {
        "name": "buscar_web",
        "description": "Buscar en la web información actual",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "La consulta de búsqueda"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "ejecutar_codigo",
        "description": "Ejecutar código Python en un entorno aislado",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Código Python a ejecutar"
                }
            },
            "required": ["code"]
        }
    }
]
```

### 3.2 Llamada a Funciones

Los LLMs que soportan **function calling** (GPT-4, Claude, Llama 3.1+) pueden generar llamadas a herramientas estructuradas:

```json
{
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "buscar_web",
        "arguments": "{\"query\": \"precio acción NVIDIA 1 junio 2026\"}"
      }
    }
  ]
}
```

El framework ejecuta la herramienta y devuelve el resultado al LLM.

---

## 4. Patrones de Razonamiento

### 4.1 ReAct (Razonamiento + Acción)

El patrón más común — intercalar razonamiento y acciones:

```
Pensamiento: Necesito encontrar la capital de Francia.
Acción: buscar("capital de Francia")
Observación: París
Pensamiento: Encontré la respuesta. Ahora debo responder.
Acción: respuesta("La capital de Francia es París.")
```

### 4.2 Planificar y Ejecutar

1. El LLM genera un plan de múltiples pasos:
   ```
   Plan:
   1. Buscar el último informe trimestral de la empresa
   2. Extraer cifras de ingresos y ganancias
   3. Comparar con el trimestre anterior
   4. Resumir los resultados
   ```
2. Ejecutar cada paso secuencialmente
3. Re-planificar si un paso falla

### 4.3 Reflexión

Después de completar una tarea, el agente reflexiona sobre la calidad de su solución:

```
Pensamiento: Mi análisis encontró un aumento de ingresos del 15%.
Pero solo miré un trimestre. Para tener una imagen completa,
también debería verificar el crecimiento interanual.
Acción: buscar("ingresos Q2 2025 vs Q2 2024")
```

La reflexión es lo que separa a los agentes simples de los efectivos.

---

## 5. Sistemas Multi-Agente

### 5.1 ¿Por Qué Múltiples Agentes?

- **Especialización**: un agente investiga, otro escribe, un tercero revisa
- **Debate**: los agentes desafían el razonamiento de los demás
- **Supervisión**: un agente monitorea a otro para detectar errores o problemas de seguridad

### 5.2 Patrones Comunes

| Patrón | Descripción | Ejemplo |
|---|---|---|
| **Supervisor** | Un agente delega en trabajadores especializados | Coordinador asigna tareas de investigación, escritura, verificación |
| **Debate** | Los agentes discuten diferentes perspectivas | Dos agentes argumentan a favor/en contra de una hipótesis |
| **Equipo RAG** | Enrutador → recuperador → generador → validador | Cada paso tiene un agente especializado |
| **Equipo de código** | Escribir → revisar → probar → corregir — un flujo de trabajo típico de [[Git]] | Generación iterativa de código con retroalimentación |

---

## 6. Seguridad y Barreras de Protección

### 6.1 Por Qué Importan las Barreras

Los agentes pueden tomar acciones autónomamente. Sin salvaguardas, un agente podría:
- Ejecutar código destructivo (borrar archivos, hacer compras)
- Acceder a datos no autorizados
- Quedar atrapado en bucles infinitos (gastando dinero en llamadas API)
- Tomar acciones que violen políticas

### 6.2 Implementando Barreras

| Mecanismo | Descripción |
|---|---|
| **Humano en el bucle** | Requerir aprobación humana para acciones críticas |
| **Validación de entrada** | Sanitizar y validar todas las entradas del usuario |
| **Validación de salida** | Verificar las salidas de las herramientas antes de devolverlas al LLM |
| **Límite de tasa** | Limitar el número de acciones por minuto |
| **Límites de presupuesto** | Costo máximo por sesión |
| **Listas de permitir/denegar** | Lista blanca de herramientas y acciones aprobadas |
| **Aislamiento** | Ejecutar código en entornos aislados |

### 6.3 Ejemplo: Límite de Presupuesto

```
Presupuesto de sesión: $0.50
Acción 1: buscar_web (costo: $0.01)
Acción 2: ejecutar_codigo (costo: $0.02)
Total: $0.03 restante: $0.47
...
Límite de sesión alcanzado: denegar más llamadas a herramientas, preguntar al usuario
```

---

## 7. Desafíos

| Desafío | Descripción | Mitigación |
|---|---|---|
| **Costo** | Cada llamada a herramienta cuesta tokens y llamadas API | Limitar pasos, operaciones por lote |
| **Propagación de errores** | Un paso incorrecto corrompe todos los pasos siguientes | Validación, autocorrección |
| **Alucinación en acciones** | El LLM inventa salidas de herramientas | Validar salidas de herramientas |
| **Bucles** | El agente repite la misma acción sin progreso | Máximo de iteraciones, detección de bucles |
| **Límites de contexto** | Ejecuciones largas del agente exceden la ventana de contexto | Resumir historial, recortar mensajes |

---

## 8. Frameworks

| Framework | Lenguaje | Características |
|---|---|---|
| **LangChain / LangGraph** | [[Python for Data Science]] | Uso de herramientas, memoria, grafos multi-agente |
| **CrewAI** | Python | Sistemas multi-agente basados en roles |
| **AutoGen** (Microsoft) | Python | Conversaciones multi-agente, ejecución de código |
| **Haystack** | Python | Pipelines RAG + agente |
| **smolagents** (HuggingFace) | Python | Agentes de código (escriben y ejecutan Python) |
| **Vercel AI SDK** | TypeScript | Streaming, uso de herramientas para apps web |

---

## 9. Common Mistakes

1. **Demasiadas herramientas**: un LLM tiene dificultades para elegir entre 20 herramientas. Comienza con 3-5 y expande cuidadosamente.

2. **Descripciones de herramientas pobres**: el LLM no puede usar una herramienta que no entiende. Escribe descripciones claras y específicas con ejemplos.

3. **Sin manejo de errores**: si una herramienta falla (error de red, límite de tasa), el agente debe manejarlo adecuadamente, no colapsar.

4. **Sin iteraciones máximas**: un agente puede repetirse indefinidamente. Siempre establece un número máximo de pasos.

5. **Ignorar el costo**: cada paso del agente cuesta dinero. Establece presupuestos y monitorea el gasto.

---

## 10. Check Your Understanding

1. Un agente busca en la web, encuentra un resultado y responde. ¿Cuántas llamadas al LLM se hicieron? (Al menos 3: la llamada inicial que decide buscar, el procesamiento de la observación y la generación de la respuesta.)

2. ¿Por qué "humano en el bucle" mejora la seguridad pero reduce la autonomía? (Requiere aprobación humana para acciones — más seguro pero más lento.)

3. Tu agente llama a `eliminar_archivo` en un servidor de producción. ¿Qué salió mal? (No había barreras — la herramienta no debería existir o debería requerir aprobación humana.)

4. Un agente se repite: buscar → no encontrar nada → buscar de nuevo con la misma consulta → repetir. ¿Cómo lo previenes? (Detectar acciones repetidas, variar las consultas de búsqueda, establecer iteraciones máximas.)

5. Un sistema de debate multi-agente tiene dos agentes discutiendo posiciones opuestas. ¿Cómo decide el sistema quién tiene razón? (Un tercer agente o juez humano evalúa los argumentos.)

---

## 11. Summary

Agentic systems extend LLMs from text generators to autonomous actors. The core loop is Observe → Think → Act → Repeat. Tools enable actions beyond text generation. Reasoning patterns like ReAct and Plan-and-Execute structure multi-step tasks. Multiple agents specialize and collaborate. Safety guardrails (human-in-the-loop, rate limits, sandboxing) prevent harmful actions. The key challenges are cost, error propagation, and maintaining context over long trajectories.

---

## 12. Where to Go Next

- [[RAG]] — Los agentes usan recuperación como herramienta
- [[Prompt Engineering]] — Diseñando prompts efectivos para agentes
- [[Fine-tuning]] — Entrenando modelos para mejor uso de herramientas
