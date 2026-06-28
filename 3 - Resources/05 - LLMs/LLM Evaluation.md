---
tags: [llm, evaluation, advanced]
status: growing
created: 2026-06-27
---

# LLM Evaluation

## 1. Why This Matters

"How good is this LLM?" is not a simple question. A model that excels at math may fail at creative writing. One that follows instructions perfectly may hallucinate freely. LLM evaluation is the practice of measuring specific capabilities to understand what a model can and cannot do.

Without evaluation — the foundation of [[Model Evaluation]] — you cannot:
- Choose between models for your application
- Know if fine-tuning improved the model
- Detect regressions after model updates
- Understand where your application will fail

---

## 2. Standard Benchmarks

### 2.1 Knowledge and Reasoning

| Benchmark | What It Tests | Format |
|---|---|---|
| **MMLU** | 57 subjects (STEM, humanities, social sciences) | 4-choice QA |
| **MMLU-Pro** | Harder MMLU (more choices, harder questions) | 10-choice QA |
| **GPQA** | Graduate-level science | Expert-written QA |
| **ARC** | Grade-school science | Multiple choice |
| **HellaSwag** | Commonsense reasoning | Sentence completion |

### 2.2 Math and Coding

| Benchmark | What It Tests | Format |
|---|---|---|
| **GSM8K** | Grade-school math word problems | Free-form answer |
| **MATH** | Competition-level math (AMC, AIME) | Free-form answer |
| **HumanEval** | Python function completion | Pass@k (functional correctness) |
| **MBPP** | Basic Python programming | Pass@k |
| **SWE-bench** | Real-world software engineering (GitHub issues) | Patch correctness |

### 2.3 Language and Dialogue

| Benchmark | What It Tests |
|---|---|
| **MT-Bench** | Multi-turn conversational ability (rated by LLM-as-judge) |
| **Chatbot Arena** | Human preference rankings (ELO system, 1M+ human votes) |
| **TruthfulQA** | Truthfulness (adversarial questions that trigger false beliefs) |
| **AlpacaEval** | Instruction following (compared to reference model) |

---

## 3. Automated Evaluation

### 3.1 LLM-as-Judge

Use one LLM to evaluate another LLM's outputs:

```python
prompt = f"""
You are evaluating an AI assistant's response.

[Question]
{question}

[Assistant Response]
{response}

Evaluate on a scale of 1-5 for:
1. Helpfulness: Does it address the user's question?
2. Accuracy: Is the information correct?
3. Harmlessness: Does it avoid harmful content?

Output JSON: {{"helpfulness": int, "accuracy": int, "harmlessness": int}}
"""
```

**Challenges**:
- Judges have biases (prefer longer answers, agree with themselves)
- Position bias (prefer first or last response in a comparison)
- Self-enhancement bias (prefer models similar to themselves)

**Mitigations**:
- Use a different, trusted model as judge (e.g., GPT-4 evaluates Llama)
- Randomize answer order in comparisons
- Use multi-dimensional scoring (separate scores for different aspects)
- Calibrate judges against human evaluations

### 3.2 Metrics

| Metric | What It Measures | Limitations |
|---|---|---|
| **Perplexity** | How well the model predicts the next token | Not correlated with output quality for chat models |
| **ROUGE** | N-gram overlap with reference | Superficial — misses semantic quality |
| **BLEU** | Precision of n-grams (translation) | Poor for creative/abstractive tasks |
| **BERTScore** | Embedding similarity with reference | Better than ROUGE/BLEU, still reference-dependent |
| **Perplexity of outputs** | How natural generations are | Measures fluency, not correctness |

---

## 4. Hallucination Detection

### 4.1 Types of Hallucination

| Type | Description | Example |
|---|---|---|
| **Factual** | False statement of fact | "Einstein invented the internet" |
| **Intrinsic** | Contradicts provided context | RAG model ignoring retrieved docs |
| **Extrinsic** | Adds information not in context | "The report states X" — report says nothing |
| **Logical** | Internally inconsistent | "I was born in 1990 and I'm 40" (should be 34) |

### 4.2 Detection Methods

- **SelfCheckGPT**: generate multiple responses, check consistency
- **NLI-based**: use an NLI model to verify if each claim is entailed by the context
- **Confidence estimation**: the model's own token probabilities (lower → more likely hallucination)
- **Citation validation**: for RAG, verify each citation points to a chunk that supports the claim

---

## 5. Human Evaluation

### 5.1 When to Use Human Evaulation

- Final assessment before deployment
- When automated metrics are unreliable (creative tasks)
- Calibrating automated judges
- Detecting subtle biases or safety issues

### 5.2 Common Protocols

| Protocol | Cost | Reliability |
|---|---|---|
| **Side-by-side (A/B)** | Low | Good — comparing two outputs |
| **Likert scale** | Medium | Moderate — subjective scale |
| **Pairwise ranking (Elo)** | High | Excellent — removes bias |
| **Chatbot Arena** | Very high | Excellent — 1M+ human judgments |

---

## 6. Evaluating Fine-Tuning

Always compare BEFORE and AFTER using [[Experiment Tracking]] to systematically log results:

```python
# Before
base_score = evaluate(base_model, test_set)

# After
ft_score = evaluate(fine_tuned_model, test_set)

# Did it improve?
print(f"Δ = {ft_score - base_score:+.3f}")
```

**Key metrics for fine-tuning evaluation**:
- Task accuracy (the metric you optimized for)
- General capability retention (MMLU, HellaSwag — did fine-tuning degrade general abilities?)
- Output format compliance (% of outputs in the correct format)
- Worst-case performance (are there inputs where the model regressed?)

---

## 7. Common Mistakes

1. **Evaluating only one dimension**: accuracy without fluency, or fluency without truthfulness. Coverage of multiple capabilities is essential.

2. **Using the same test data for development**: if you tune hyperparameters based on MMLU, MMLU is no longer an unbiased evaluation. Use a separate held-out set, a standard practice in [[Supervised Learning]].

3. **Ignoring variance**: a 0.5% MMLU improvement may not be statistically significant. Report confidence intervals (a [[Statistics]] best practice) or run multiple trials.

4. **Evaluating only aggregate metrics**: a model may score well overall but fail catastrophically on specific categories (e.g., safety, math, non-English).

5. **Over-trusting LLM-as-Judge**: automated judges are useful but biased. Cross-validate with human evaluation for critical decisions.

---

## 8. Check Your Understanding

1. A model scores 90% on MMLU but 30% on TruthfulQA. What does this tell you? (High knowledge but low truthfulness — factually unreliable.)

2. Why does perplexity not correlate well with chat quality? (Perplexity measures next-token prediction, not output quality. A very conservative model that says "I don't know" can have low perplexity but be useless.)

3. Your fine-tuned model scores better on your task but worse on MMLU. Should you deploy it? (It depends — if task performance outweighs general degradation, yes. Monitor for edge cases.)

4. Two annotators give different scores to the same output. What do you do? (Calculate inter-annotator agreement. Disagreements may indicate unclear rubrics.)

5. An LLM-as-Judge consistently prefers longer answers. How do you mitigate this? (Control for length by comparing answers of similar length or using length-calibrated scoring.)

---

## 9. Summary

LLM evaluation is multi-dimensional: knowledge, reasoning, coding, truthfulness, safety, and instruction following. Use a mix of standard benchmarks (MMLU, HumanEval, MT-Bench) for general capability, automated judges for task-specific evaluation, and human evaluation for final assessment. Always compare against a baseline, measure multiple dimensions, and be aware of measurement biases (judge preferences, position bias, variance).

---

## 10. Where to Go Next

- [[Prompt Engineering]] — The quality of prompts affects evaluation
- [[Fine-tuning]] — Evaluating before and after fine-tuning
- [[RAG]] — Evaluating retrieval and generation separately
