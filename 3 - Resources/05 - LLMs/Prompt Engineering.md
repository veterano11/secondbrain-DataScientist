---
tags: [llm, prompting, core]
status: growing
created: 2026-06-27
---

# Prompt Engineering

## 1. Why This Matters

LLMs are capable of a vast range of tasks — translation, coding, reasoning, creativity — but extracting the right behavior requires **the right prompt**. Prompt engineering is the skill of communicating intent to an LLM effectively.

It matters because a well-designed prompt can make a 7B model outperform a 70B model with a poor prompt — understanding the underlying [[Transformer Architecture]] helps explain why. It is the cheapest and fastest way to improve LLM outputs.

---

## 2. Basic Prompt Design

### 2.1 The Anatomy of a Prompt

```
[System Message] — sets the role, tone, constraints
[Few-shot Examples] — demonstrates desired behavior
[User Query] — the actual request
[Assistant Response] — model's output
```

### 2.2 System Prompts

The system prompt defines the model's persona and behavioral constraints:

```
You are an expert data scientist. Answer technical questions clearly and provide code examples when relevant. If you are unsure about something, say so — do not hallucinate.
```

A good system prompt is specific about:
- **Role**: who the model is (expert, assistant, critic)
- **Tone**: formal, casual, educational, concise
- **Constraints**: what to avoid (hallucinations, vague answers)
- **Output format**: markdown, JSON, bullet points

### 2.3 Zero-Shot vs Few-Shot

**Zero-shot**: give the model a task without examples.

```
Translate to Spanish: "Hello, how are you?"
```

**Few-shot**: give 1-3 examples of the desired input-output pattern.

```
English: "Hello" → Spanish: "Hola"
English: "Good morning" → Spanish: "Buenos días"
English: "Thank you" → Spanish: "Gracias"
English: "How are you?" → Spanish:
```

Few-shot is more reliable for unusual formats or tasks, functioning much like [[Supervised Learning]] with a handful of labeled examples.

---

## 3. Advanced Prompting Techniques

### 3.1 Chain-of-Thought (CoT)

**Key idea**: prompt the model to reason step-by-step before answering.

**Without CoT**:
```
Q: A store has 20 apples and sells 3 per day. How many after 5 days?
A: 5
(Incorrect — the model guessed)
```

**With CoT**:
```
Q: A store has 20 apples and sells 3 per day. How many after 5 days?
A: Let's think step by step.
1. Initial apples: 20
2. Sold per day: 3
3. Total sold after 5 days: 3 × 5 = 15
4. Remaining: 20 - 15 = 5
Answer: 5
```

CoT dramatically improves performance on math, logic, and multi-step reasoning tasks. It works because:
- It externalizes the reasoning process
- The model can correct its own intermediate steps
- It decomposes a hard problem into easier subproblems

### 3.2 Self-Consistency

Run the same prompt multiple times with temperature > 0, then take the majority answer. This improves reliability on reasoning tasks because:
- Different reasoning paths should converge to the same answer
- Random errors from one path are outvoted

### 3.3 Tree-of-Thoughts

Unlike CoT (one chain), Tree-of-Thoughts explores multiple reasoning paths simultaneously:

```
Path 1: A → B → C → D → Result
Path 2: A → B → E → F → Result
Path 3: A → G → H → Result
```

At each step, the model evaluates which paths are promising and prunes the rest. More expensive but more powerful than CoT.

### 3.4 ReAct (Reasoning + Acting)

Combines reasoning with tool use:

```
Thought: I need to find the population of Tokyo.
Action: search("Tokyo population")
Observation: 14 million (2023)
Thought: Now I can answer the question.
Action: answer("Tokyo has 14 million people.")
```

The model generates thoughts, actions, and observes results in a loop. This is the foundation of **agentic systems**.

---

## 4. Output Control

### 4.1 Structured Output

Force the model to produce parseable output:

```json
You are a data extractor. Given text, output JSON with:
- "date": the date mentioned (YYYY-MM-DD or null)
- "entities": list of people mentioned
- "summary": one-sentence summary

Text: "On June 27, 2026, Alice and Bob published their findings."

Output:
{
  "date": "2026-06-27",
  "entities": ["Alice", "Bob"],
  "summary": "Alice and Bob published findings on June 27, 2026."
}
```

### 4.2 Controlling Verbosity

```
Be concise. Answer in 2-3 sentences maximum.
```

vs.

```
Provide a detailed analysis with examples and counterexamples.
```

### 4.3 Mitigating Hallucination

**Prompt-level techniques**:
- "Only use information from the context below."
- "If you are unsure, say 'I don't know.'"
- "Cite specific sources for each claim."

For RAG systems, the prompt should explicitly tell the model to prefer retrieved context over parametric knowledge. [[Model Evaluation]] techniques can help measure whether this guidance is effective.

---

## 5. Prompt Patterns

### 5.1 Persona Pattern

```
Act as a senior data scientist reviewing a colleague's code.
```

### 5.2 Template Pattern

```
Complete the following analysis:

Analysis of Model Performance:
- Accuracy: {model_name} achieved {accuracy}% on {dataset}
- Best feature: {best_feature} with importance {importance}
- Recommendation: {recommendation}
```

### 5.3 Chain Pattern

Chain multiple prompts where each output feeds into the next:

```
Step 1: "Generate 5 hypotheses for why customer churn increased."
Step 2: "For each hypothesis, propose a way to test it with data."
Step 3: "Given the test results below, which hypothesis is most supported?"
```

---

## 6. Common Mistakes

1. **Overly complex prompts**: a prompt with many instructions confuses the model. Prioritize the most important rules.

2. **Assuming the model remembers conversation context**: after ~8K-128K tokens, the model loses early context. For long conversations, summarize periodically.

3. **Not testing systematically**: what works once may not work consistently. Test prompts with multiple inputs and different temperatures, using [[Experiment Tracking]] to systematically record results.

4. **Too many constraints**: "be concise, but thorough, but use examples, but don't be too technical" — conflicting instructions degrade performance.

5. **Neglecting system prompts**: the system message is the most powerful tool for setting behavior. Use it explicitly.

---

## 7. Check Your Understanding

1. You ask a model "What is 23 × 47?" and get a wrong answer. How would you rewrite the prompt to improve accuracy?

2. Chain-of-Thought improves math reasoning. Why does it work? (Decomposes the problem, externalizes reasoning, enables self-correction.)

3. Your RAG system retrieves relevant documents but the model ignores them. What prompt changes would you make?

4. You need JSON output but the model occasionally adds commentary. How would you fix this?

5. Self-consistency runs multiple samples. When would the computational cost be worth it?

---

## 8. Summary

Prompt engineering is the skill of communicating intent to LLMs. A good prompt specifies role, tone, format, and constraints. Advanced techniques like Chain-of-Thought and ReAct improve reasoning and tool use. Structured output (JSON) enables downstream processing. The key is systematic testing — vary prompts, measure results, iterate.

---

## 9. Where to Go Next

- [[RAG]] — Combining prompts with retrieved context
- [[Agentic Systems]] — Multi-step reasoning with tool use
- [[LLM Evaluation]] — Measuring prompt effectiveness
