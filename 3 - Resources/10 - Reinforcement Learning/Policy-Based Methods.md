---
tags: [reinforcement-learning, policy-based, ppo, actor-critic]
status: growing
created: 2026-06-27
---

# Policy-Based Methods

## 1. Escenario de aprendizaje

Querés que un brazo robótico aprenda a agarrar objetos. Las acciones son continuas (ángulos de cada articulación, fuerza del agarre). Value-based methods (DQN) no sirven para acciones continuas — necesitarías discretizar el espacio y sería enorme.

Policy-based methods aprenden la política $\pi(a|s)$ directamente, sin pasar por $Q$. Funcionan con acciones continuas y pueden ser estocásticas (útiles para exploración).

## 2. Policy Gradient: la idea central

Optimizamos directamente el retorno esperado $J(\theta) = \mathbb{E}_{\pi_\theta}[G_0]$.

El gradiente es:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot G_t \right]$$

**Intuición**: 
- Si $G_t$ es positivo → aumentamos $\log \pi_\theta(a|s)$ → la acción es más probable
- Si $G_t$ es negativo → disminuimos $\log \pi_\theta(a|s)$ → la acción es menos probable

**No es supervisado**: no hay "acción correcta". Hay acciones que llevaron a alta recompensa.

## 3. REINFORCE (Monte Carlo Policy Gradient)

```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, action_dim),
        )

    def forward(self, s):
        logits = self.net(s)
        return Categorical(logits=logits)  # distribución sobre acciones

def reinforce(env, policy, lr=1e-3, gamma=0.99):
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    for ep in range(1000):
        # 1. Colectar trayectoria completa
        estados, acciones, recompensas = [], [], []
        s, _ = env.reset()
        while True:
            dist = policy(torch.FloatTensor(s))
            a = dist.sample()
            s2, r, done, _ = env.step(a.item())
            estados.append(s); acciones.append(a); recompensas.append(r)
            s = s2
            if done: break

        # 2. Calcular G (retorno descontado) para cada paso
        G = 0
        returns = []
        for r in reversed(recompensas):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)

        # 3. Gradient step
        log_probs = []
        for s, a in zip(estados, acciones):
            dist = policy(torch.FloatTensor(s))
            log_probs.append(dist.log_prob(a))

        loss = - (torch.stack(log_probs) * returns).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**Problema**: alta varianza. Si un episodio tiene G=5 y otro G=-3, el gradiente fluctúa mucho. Diferentes semillas de ruido pueden dar resultados muy distintos.

## 4. Actor-Critic

Reducimos varianza restando una **línea de base** (baseline) — típicamente $V(s)$:

$$\nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot \underbrace{(Q(s,a) - V(s))}_{A(s,a)}]$$

$A(s,a)$ es la **ventaja**: qué tan mejor es esta acción comparada con el promedio.

```python
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, s):
        logits = self.actor(s)
        value = self.critic(s)
        return Categorical(logits=logits), value
```

**Actor** (policy): decide qué acción tomar. **Critic** (value): evalúa qué tan buena es la situación actual.

## 5. PPO (Proximal Policy Optimization)

El método más usado hoy. Soluciona el problema de que un paso de gradiente muy grande puede arruinar la política.

$$\mathcal{L}^{\text{CLIP}} = \mathbb{E} \left[ \min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]$$

donde $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$.

```python
def ppo_update(policy, optimizer, batch, epsilon=0.2):
    estados, acciones, ventajas, target_values = batch

    # Calcular ratio
    dist = policy(estados)
    new_log_probs = dist.log_prob(acciones)
    ratio = (new_log_probs - old_log_probs).exp()

    # Clipped objective
    obj1 = ratio * ventajas
    obj2 = torch.clamp(ratio, 1-epsilon, 1+epsilon) * ventajas
    actor_loss = -torch.min(obj1, obj2).mean()

    # Critic loss
    values = policy.critic(estados)
    critic_loss = F.mse_loss(values, target_values)

    loss = actor_loss + 0.5 * critic_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**¿Por qué PPO funciona?** Clip a $1\pm\epsilon$ (típicamente 0.2). Si un cambio propuesto es muy grande (ratio=3), se trunca a 1.2. Esto evita que un mal batch arruine la política.

## 6. SAC (Soft Actor-Critic)

Para control continuo (robótica, juegos). Maximiza recompensa + entropía:

$$J(\pi) = \sum_t \mathbb{E}[r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))]$$

- $\mathcal{H}$: entropía de la política (cuánto explora)
- $\alpha$: temperatura (trade-off explotación-exploración)

**¿Cuándo usar cuál?**

| Situación | Método |
|-----------|--------|
| Acciones discretas, espacio pequeño | DQN |
| Acciones continuas, control | SAC |
| [[Fine-tuning|Tuning fino]] de LLM | PPO |
| Problema con recompensas esporádicas | PPO + GAE |
| Simulación rápida (muchos episodios) | REINFORCE (simple) |

## 7. Common Mistakes

1. **Policy gradient sin baseline**: la varianza es tan alta que el entrenamiento no converge. Siempre usar actor-critic o al menos restar un baseline.
2. **PPO: múltiples updates por batch**: PPO puede hacer varias épocas de gradient descent en el mismo batch, pero si son demasiadas (K>10), el clipping no alcanza y la política colapsa.
3. **SAC: temperatura fija**: si $\alpha$ es fijo, puede ser muy alto (mucho ruido) o muy bajo (poca exploración). SAC moderno aprende $\alpha$ automáticamente.
4. **No normalizar rewards**: ranges muy distintos dificultan el aprendizaje. [[Training Techniques|Normalizar rewards por episodio]] ayuda.

## 8. Check Your Understanding

1. ¿Por qué REINFORCE tiene alta varianza? (Usa el retorno G de trayectorias completas, que es muy ruidoso)
2. ¿Qué ventaja tiene PPO sobre REINFORCE? (PPO limita el tamaño del paso, evitando que un mal batch destruya la política)
3. ¿Cuándo preferirías SAC a PPO? (Acciones continuas, especialmente control robótico)
4. ¿Qué rol juega el critic en Actor-Critic? (Provee el baseline V(s) para reducir varianza del gradiente)

## 9. Summary

Policy-based methods aprenden la política directamente. REINFORCE es simple pero ruidoso. Actor-Critic reduce varianza restando una línea de base. PPO agrega clipping para estabilidad y es el estándar industrial. SAC maximiza entropía para exploración robusta en control continuo. Para acciones continuas y problemas complejos, policy-based superan a value-based.

## 10. Where to Go Next

- [[Value-Based Methods]] — enfoque alternativo
- [[RL Fundamentals]] — teoría de políticas y valores
- [[RLHF & Preference Optimization]] — PPO para alinear LLMs
- [[Model-Based RL]] — combinando modelos del mundo con policy learning
