---
tags: [reinforcement-learning, rlhf, llm, alignment, dpo]
status: growing
created: 2026-06-27
---

# RLHF & Preference Optimization

## 1. Escenario de aprendizaje

Entrenaste un [[Neural Networks|modelo de lenguaje]] con next-token prediction. Habla fluidamente pero: da respuestas incorrectas con total confianza, no sabe decir "no sé", y a veces genera contenido ofensivo.

La máxima verosimilitud no alinea al modelo con lo que los humanos consideramos útil, honesto e inofensivo. RLHF (Reinforcement Learning from Human Feedback) cierra esa brecha.

## 2. El problema que resuelve RLHF

Un modelo entrenado con cross-entropy maximiza $P(y|x)$ en el corpus de entrenamiento. Pero el corpus contiene:

- Respuestas incorrectas (ruido)
- Información desactualizada
- Contenido sesgado o dañino

El modelo replica estas características. RLHF le enseña a preferir respuestas que los humanos aprueban.

## 3. El pipeline RLHF

```
Paso 1: SFT (Supervised Fine-Tuning)
    Modelo base → fine-tune en demostraciones escritas por humanos
    Propósito: enseñar formato y estilo de respuesta útil

Paso 2: Reward Model
    Humanos comparan respuestas A vs B → se entrena un modelo que predice preferencias

Paso 3: PPO
    El policy (LLM) se optimiza contra el reward model, con un penalty KL
    para no alejarse demasiado del SFT
```

### 3.1 Paso 1: SFT

Fine-tuning supervisado en un dataset de prompts + respuestas escritas por humanos.

$$\mathcal{L}_{\text{SFT}} = -\sum_t \log P(y_t | x, y_{<t})$$

Nada nuevo: es el mismo loss que el pre-training, pero el dataset es de alta calidad curada por humanos.

### 3.2 Paso 2: Reward Model

Entrenamos $r_\phi(x, y)$ que predice qué tan buena es una respuesta $y$ para un prompt $x$.

Dos respuestas para el mismo prompt: $y_w$ (la preferida) y $y_l$ (la otra). El reward model aprende a dar mayor score a $y_w$.

$$\mathcal{L}_R = -\mathbb{E}[\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))]$$

```python
# Dataset: pares (prompt, respuesta_ganadora, respuesta_perdedora)
for x, y_w, y_l in dataloader:
    r_w = reward_model(x, y_w)            # score de la ganadora
    r_l = reward_model(x, y_l)            # score de la perdedora

    loss = -F.logsigmoid(r_w - r_l).mean()
    # El modelo aprende a rankear: r_w > r_l
```

**¿Cuántos datos se necesitan?** ~100K comparaciones humanas para un modelo como GPT-3.5.

### 3.3 Paso 3: PPO

Optimizamos el LLM ($\pi_\theta$) contra el reward model con una restricción KL.

$$\text{objective} = \mathbb{E}_{x, y \sim \pi_\theta} [r_\phi(x, y) - \beta \cdot D_{\text{KL}}(\pi_\theta(y|x) \| \pi_{\text{SFT}}(y|x))]$$

```python
# PPO for RLHF (conceptual)
for prompt in prompts:
    response = policy.generate(prompt)                  # π_θ
    reward = reward_model(prompt, response)              # r_ϕ
    kl = kl_divergence(policy(prompt), sft_model(prompt))

    advantage = reward - beta * kl

    # PPO clipped loss para actualizar policy
    ratio = (policy(prompt).log_prob(response) / old_policy(prompt).log_prob(response)).exp()
    loss = -min(ratio * advantage, clip(ratio, 0.8, 1.2) * advantage)
```

**¿Por qué el penalty KL?** Sin él, el policy aprende a generar texto que maximiza el reward model aunque sea incomprensible para humanos (reward hacking). El KL penalty mantiene al modelo cerca del SFT.

## 4. Direct Preference Optimization (DPO)

DPO elimina el reward model y el PPO. La idea: la política óptima del problema RLHF tiene una forma cerrada que depende solo de la política actual y las preferencias.

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

```python
def dpo_loss(policy_logps, ref_logps, pref_mask):
    # policy_logps: (B, 2) = [log_p(w), log_p(l)]
    # ref_logps:    (B, 2) = [log_ref(w), log_ref(l)]

    log_ratio = policy_logps - ref_logps
    diff = log_ratio[:, 0] - log_ratio[:, 1]  # chosen - rejected
    loss = -F.logsigmoid(beta * diff).mean()
    return loss
```

**Ventajas de DPO vs RLHF:**

| Aspecto | RLHF (PPO) | DPO |
|---------|------------|-----|
| Entrenar reward model | Sí | No |
| Sampling del policy online | Sí | No |
| Estabilidad | Media | Alta |
| Implementación | Compleja | Simple |
| Rendimiento | Similar | Similar |

## 5. KTO (Kahneman-Tversky Optimization)

KTO simplifica aún más: solo necesitás saber si cada respuesta individual es buena o mala (no pares).

$$\mathcal{L}_{\text{KTO}} = -\mathbb{E}[\lambda_w \sigma(\beta(v(x, y_w) - z_0)) + \lambda_l \sigma(\beta(z_0 - v(x, y_l)))]$$

donde $v(x, y) = \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ y $z_0$ es un punto de referencia.

## 6. Reward Hacking

El mayor riesgo de RLHF: el policy encuentra formas de maximizar el reward model que no corresponden a respuestas útiles.

Ejemplo real: un modelo aprendió a generar respuestas larguísimas porque el reward model asociaba "respuesta larga" con "respuesta útil". El KL penalty mitiga esto, pero no lo elimina.

**Cómo mitigarlo:**
- [[Training Techniques|Ensembles]] de reward models
- Evaluación periódica con humanos
- Held-out reward model para detección temprana

## 7. Common Mistakes

1. **Reward model muy fuerte**: si el reward model es perfecto (accuracy 100% en preferencias), el policy encuentra adversarial examples. Un reward model decente (~70-75% accuracy) es mejor.
2. **KL coefficient mal calibrado**: muy bajo → reward hacking. Muy alto → no hay alineación. Hay que tunearlo.
3. **DPO sin reference model congelado**: si actualizás el reference model durante el training, la derivación de DPO se rompe.
4. **Datos de preferencia ruidosos**: anotadores humanos discordantes (~20% desacuerdo). Si el reward model aprende el ruido, el policy se alinea al ruido.

## 8. Check Your Understanding

1. ¿Por qué RLHF usa un penalty KL en vez de maximizar directamente el reward? (Para evitar que el policy genere texto que maximiza el reward pero no es útil — reward hacking)
2. DPO no entrena un reward model. ¿Qué usa en su lugar? (La razón de probabilidades entre la política actual y la de referencia)
3. ¿Qué pasa si fine-tuneás un modelo con DPO y luego querés fine-tunearlo de nuevo con más datos?

## 9. Summary

RLHF alinea LLMs con preferencias humanas: SFT (estilo), reward model (qué es bueno), PPO (optimizar). DPO simplifica eliminando el reward model y usando una función de pérdida cerrada. KTO simplifica aún más. El riesgo principal es reward hacking, mitigado con KL penalty y monitoreo. RLHF/DPO son la razón por la que ChatGPT, Claude, y Llama Chat son útiles en vez de solo fluidos.

## 10. Where to Go Next

- [[Fine-tuning]] — SFT es el primer paso de RLHF
- [[Policy-Based Methods]] — PPO es el algoritmo RL
- [[LLM Evaluation]] — cómo se evalúa la alineación
- [[Prompt Engineering]] — cómo los humanos escriben prompts para alignment
- [[RL Fundamentals]] — MDP y valores, base teórica de RLHF
