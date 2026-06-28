---
tags: [llm, agents, advanced]
status: growing
created: 2026-06-27
---

# Agentic Systems

## 1. Why This Matters

An LLM by itself is a text generator built on the [[Transformer Architecture]]. You give it text, it gives you text. But when you give it **tools** — web search, code execution, database queries, file access — it becomes an **agent** that can take actions, observe results, and plan multi-step solutions.

Agentic systems are the most exciting frontier in applied LLMs. A well-designed agent can answer questions that require research, write and debug code, manage complex workflows (automating [[CLI & Productivity]] tasks), and act autonomously within defined boundaries.

---

## 2. The Agent Loop

### 2.1 Core Loop

```
1. Observe (user input or tool output)
2. Think (reason about what to do next)
3. Act (respond or call a tool)
4. Observe (tool result)
5. Repeat until task is complete
```

### 2.2 A Concrete Example

```
User: "What was the stock price of NVIDIA on June 1, 2026?"

LLM thought: I need to search for NVIDIA stock price data.
Action: search_web(query="NVIDIA stock price June 1 2026")
Observation: "NVDA closed at $185.42 on June 1, 2026"

LLM thought: I have the answer, I should respond.
Action: answer("NVIDIA's stock price on June 1, 2026 was $185.42.")
```

The LLM decides when to use tools and when to respond directly.

---

## 3. Tool Use

### 3.1 Tool Design

Each tool needs:
- **Name**: descriptive and unique
- **Description**: explains when and how to use the tool
- **Parameters**: schema (JSON Schema) defining inputs
- **Implementation**: the actual function that runs

```python
tools = [
    {
        "name": "search_web",
        "description": "Search the web for current information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "run_code",
        "description": "Execute Python code in a sandboxed environment",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute"
                }
            },
            "required": ["code"]
        }
    }
]
```

### 3.2 Function Calling

LLMs that support **function calling** (GPT-4, Claude, Llama 3.1+) can output structured tool calls:

```json
{
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "search_web",
        "arguments": "{\"query\": \"NVIDIA stock price June 1 2026\"}"
      }
    }
  ]
}
```

The framework executes the tool and returns the result to the LLM.

---

## 4. Reasoning Patterns

### 4.1 ReAct (Reasoning + Acting)

The most common pattern — interleave reasoning and actions:

```
Thought: I need to find the capital of France.
Action: search("capital of France")
Observation: Paris
Thought: I found the answer. Now I should respond.
Action: answer("The capital of France is Paris.")
```

### 4.2 Plan-and-Execute

1. LLM generates a multi-step plan:
   ```
   Plan:
   1. Search for the company's latest quarterly report
   2. Extract revenue and profit numbers
   3. Compare with previous quarter
   4. Summarize the results
   ```
2. Execute each step sequentially
3. Re-plan if a step fails

### 4.3 Reflection

After completing a task, the agent reflects on the quality of its solution:

```
Thought: My analysis found a 15% revenue increase.
But I only looked at one quarter. For a complete picture,
I should also check year-over-year growth.
Action: search("revenue Q2 2025 vs Q2 2024")
```

Reflection is what separates simple agents from effective ones.

---

## 5. Multi-Agent Systems

### 5.1 Why Multiple Agents?

- **Specialization**: one agent researches, another writes, a third reviews
- **Debate**: agents challenge each other's reasoning
- **Oversight**: one agent monitors another for errors or safety issues

### 5.2 Common Patterns

| Pattern | Description | Example |
|---|---|---|
| **Supervisor** | One agent delegates to specialized workers | Coordinator assigns research, writing, verification tasks |
| **Debate** | Agents discuss different perspectives | Two agents argue for/against a hypothesis |
| **RAG team** | Router → retriever → generator → validator | Each step has a specialized agent |
| **Code team** | Write → review → test → fix — a typical [[Git]] collaboration workflow | Iterative code generation with feedback |

---

## 6. Safety and Guardrails

### 6.1 Why Guardrails Matter

Agents can take actions autonomously. Without safeguards, an agent might:
- Execute destructive code (delete files, make purchases)
- Access unauthorized data
- Get stuck in infinite loops (spending money on API calls)
- Take actions that violate policies

### 6.2 Implementing Guardrails

| Mechanism | Description |
|---|---|
| **Human-in-the-loop** | Require human approval for critical actions |
| **Input validation** | Sanitize and validate all user inputs |
| **Output validation** | Check tool outputs before feeding back to LLM |
| **Rate limiting** | Cap the number of actions per minute |
| **Budget limits** | Maximum cost per session |
| **Allow/deny lists** | Whitelist approved tools and actions |
| **Sandboxing** | Execute code in isolated environments |

### 6.3 Example: Budget Limiting

```
Session budget: $0.50
Action 1: search_web (cost: $0.01)
Action 2: run_code (cost: $0.02)
Total: $0.03 remaining: $0.47
...
Session limit reached: deny further tool calls, ask user
```

---

## 7. Challenges

| Challenge | Description | Mitigation |
|---|---|---|
| **Cost** | Each tool call costs tokens and API calls | Limit steps, batch operations |
| **Error propagation** | One wrong step corrupts all subsequent steps | Validation, self-correction |
| **Hallucination in actions** | LLM invents tool outputs | Validate tool outputs |
| **Looping** | Agent repeats the same action without progress | Max iterations, loop detection |
| **Context limits** | Long agent runs exceed the context window | Summarize history, trim messages |

---

## 8. Frameworks

| Framework | Language | Features |
|---|---|---|
| **LangChain / LangGraph** | [[Python for Data Science]] | Tool use, memory, multi-agent graphs |
| **CrewAI** | Python | Role-based multi-agent systems |
| **AutoGen** (Microsoft) | Python | Multi-agent conversations, code execution |
| **Haystack** | Python | RAG + agent pipelines |
| **smolagents** (HuggingFace) | Python | Code agents (write and execute Python) |
| **Vercel AI SDK** | TypeScript | Streaming, tool use for web apps |

---

## 9. Common Mistakes

1. **Too many tools**: an LLM struggles to choose between 20 tools. Start with 3-5 and expand carefully.

2. **Poor tool descriptions**: the LLM cannot use a tool it does not understand. Write clear, specific descriptions with examples.

3. **No error handling**: if a tool fails (network error, rate limit), the agent should handle it gracefully, not crash.

4. **No max iterations**: an agent can loop indefinitely. Always set a maximum number of steps.

5. **Ignoring cost**: each agent step costs money. Set budgets and monitor spending.

---

## 10. Check Your Understanding

1. An agent searches the web, finds a result, and responds. How many LLM calls were made? (At least 3: the initial call that decides to search, the observation processing, and the response generation.)

2. Why does "human-in-the-loop" improve safety but reduce autonomy? (Requires human approval for actions — safer but slower.)

3. Your agent calls `delete_file` on a production server. What went wrong? (No guardrails — the tool should not exist or should require human approval.)

4. An agent loops: search → find nothing → search again with same query → repeat. How do you prevent this? (Detect repeated actions, vary search queries, set max iterations.)

5. A multi-agent debate system has two agents arguing opposite positions. How does the system decide who is right? (A third agent or human judge evaluates the arguments.)

---

## 11. Summary

Agentic systems extend LLMs from text generators to autonomous actors. The core loop is Observe → Think → Act → Repeat. Tools enable actions beyond text generation. Reasoning patterns like ReAct and Plan-and-Execute structure multi-step tasks. Multiple agents specialize and collaborate. Safety guardrails (human-in-the-loop, rate limits, sandboxing) prevent harmful actions. The key challenges are cost, error propagation, and maintaining context over long trajectories.

---

## 12. Where to Go Next

- [[RAG]] — Agents use retrieval as a tool
- [[Prompt Engineering]] — Designing effective agent prompts
- [[Fine-tuning]] — Training models for better tool use
