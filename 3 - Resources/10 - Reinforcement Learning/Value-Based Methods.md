---
tags: [reinforcement-learning, value-based, q-learning, dqn]
status: growing
created: 2026-06-27
---

# Value-Based Methods

## 1. Escenario de aprendizaje

Volvamos al robot en el laberinto 3×3 de [[RL Fundamentals]]. Queremos que aprenda la ruta óptima sin conocer el mapa. La idea: mantenemos una tabla Q que estima qué tan buena es cada acción en cada estado. El robot prueba acciones, actualiza la tabla, y con suficientes intentos converge a la política óptima.

Value-based methods aprenden $Q(s,a)$ y derivan la política eligiendo la acción con mayor Q.

## 2. Dynamic Programming (model-based)

Si conocemos el modelo del mundo ($P$ y $R$), podemos computar $V$ directamente sin explorar.

**Policy evaluation**: actualizar $V$ para una política fija:

$$V_{k+1}(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R + \gamma V_k(s')]$$

**Policy improvement**: mejorar la política basándose en $V$:

$$\pi'(s) = \arg\max_a \sum_{s'} P(s'|s,a)[R + \gamma V^\pi(s')]$$

**Value iteration**: combinar ambos en un solo paso:

$$V_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a)[R + \gamma V_k(s')]$$

**Problema**: necesitás $P$ y $R$ conocidos. En la mayoría de problemas reales no los tenés.

## 3. Q-Learning (model-free)

No necesitás conocer $P$, solo necesitás interactuar con el ambiente.

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s,a) \right]$$

**Paso a paso**:

```python
def q_learning(env, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    Q = np.zeros((env.n_estados, env.n_acciones))
    recompensas = []

    for ep in range(episodes):
        s = env.reset()
        total = 0

        while True:
            # ε-greedy para elegir acción
            if np.random.random() < epsilon:
                a = np.random.choice(env.n_acciones)
            else:
                a = np.argmax(Q[s])

            s2, r, done = env.step(a)

            # Q-learning update
            Q[s][a] += alpha * (r + gamma * np.max(Q[s2]) - Q[s][a])

            total += r
            s = s2
            if done:
                break

        recompensas.append(total)

        if ep % 200 == 0:
            print(f"Ep {ep}: reward = {total}")

    return Q, recompensas
```

**Off-policy**: Q-learning aprende la política óptima $\pi^*$ mientras sigue una política exploratoria ($\varepsilon$-greedy). Esto es lo opuesto a SARSA.

### Ejemplo numérico (laberinto 3×3)

```
Estado inicial (0,0): mover derecha → recompensa = -1 (cada paso)
                      Q[(0,0), derecha] = Q[(0,0), derecha] + 0.1 × [-1 + 0.9 × max(Q[(0,1)]) - Q[(0,0), derecha]]
                    
Si Q[(0,0), derecha] = 0 y max(Q[(0,1)]) = 5 (cerca de la meta):
  Nuevo Q = 0 + 0.1 × [-1 + 0.9 × 5 - 0] = 0 + 0.1 × 3.5 = 0.35

Q[(0,0), derecha] subió de 0 a 0.35 — aprendió que ir a la derecha es bueno.
```

## 4. SARSA (On-Policy)

Similar pero usa la acción **real** del siguiente paso (la que la política actual elegiría), no la mejor acción posible:

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma Q(s', a') - Q(s,a) \right]$$

**¿Cuándo usar SARSA vs Q-learning?**

| Situación | Recomendación |
|-----------|--------------|
| Entorno peligroso (errores cuestan caro) | SARSA (más conservador) |
| Entorno seguro, querés máxima performance | Q-learning |
| Exploración con riesgo de catástrofe | SARSA |
| Simulación (puedo equivocarme gratis) | Q-learning |

**Ejemplo**: si hay un precipicio al lado del camino óptimo, Q-learning aprende a caminar por el borde (porque asume que siempre elegirá la mejor acción). SARSA aprende a caminar más lejos (porque considera que a veces la política exploratoria se tropezará).

## 5. Deep Q-Network (DQN)

Cuando el espacio de estados es grande (píxeles de una pantalla de Atari), una tabla Q no alcanza. DQN usa una [[Neural Networks|red neuronal]] para aproximar $Q(s,a)$.

### 5.1 Tres innovaciones clave

**Experience Replay**: guardamos experiencias $(s, a, r, s')$ en un buffer y muestreamos batches aleatorios. Rompe la correlación entre experiencias consecutivas.

```python
# Experience replay
buffer = deque(maxlen=100_000)
buffer.append((s, a, r, s2, done))

batch = random.sample(buffer, batch_size)
# cada elemento del batch: experiencia independiente
```

**Target Network**: una copia congelada de la red Q que se actualiza cada C pasos. Estabiliza el target.

```python
# target network actualizado cada 100 pasos
if step % 100 == 0:
    target_net.load_state_dict(q_net.state_dict())
```

**Gradient clipping**: limita la magnitud del gradiente para estabilidad.

```python
# Pérdida con clipping (Huber loss)
loss = F.smooth_l1_loss(q_pred, q_target)
```

### 5.2 Pérdida de DQN

$$\mathcal{L} = \mathbb{E}_{(s,a,r,s')} \left[ \left( r + \gamma \max_{a'} Q_{\text{target}}(s', a') - Q_\theta(s,a) \right)^2 \right]$$

```python
# Código conceptual del update DQN
q_values = q_net(states)                    # Q_θ(s)
q_current = q_values.gather(1, actions)     # Q_θ(s, a)

with torch.no_grad():
    next_q = target_net(next_states)        # Q_target(s')
    max_next_q = next_q.max(dim=1)[0]       # max_a' Q(s', a')
    q_target = rewards + gamma * max_next_q * (1 - dones)

loss = F.mse_loss(q_current, q_target)
loss.backward()
```

### 5.3 Mejoras sobre DQN

| Extensión | Problema que resuelve | Cambio |
|-----------|----------------------|--------|
| **Double DQN** | Sobreestimación de Q | Usa Q_online para elegir acción, Q_target para evaluar |
| **Dueling DQN** | Estados con valores similares | Separa $V(s) + A(s,a)$ en dos streams |
| **PER** | Experiencias poco informativas | Muestrea más frecuentemente experiencias con alto TD error |
| **N-step** | Recompensas lentas en propagarse | Bootstrapping a n pasos en vez de 1 |

```python
# Double DQN: elegir acción con online, evaluar con target
accion_elegida = q_net(next_states).argmax(dim=1)
q_target = target_net(next_states).gather(1, accion_elegida.unsqueeze(1)).squeeze()
```

## 6. Common Mistakes

1. **Q-learning sobreestima Q**: como usa $\max_a Q(s',a')$, siempre tiende a sobreestimar. Double DQN corrige esto.
2. **[[Training Techniques|Catastrophic forgetting]] sin replay**: sin experience replay, la red olvida experiencias pasadas al ver solo la última trayectoria.
3. **Espacio de acciones grande**: DQN escala mal a miles de acciones. Para eso están los policy-based methods.
4. **Hiperparámetros de DQN**: learning rate, tamaño del buffer, frecuencia de actualización del target — todo importa. Usá los valores de los papers (lr=1e-4, buffer=1e6, target update cada 1e4 steps).

## 7. Check Your Understanding

1. Q-learning es off-policy y SARSA on-policy. ¿Qué significa exactamente?
2. ¿Por qué DQN necesita experience replay y una target network? (Correlación en datos y targets no-estacionarios)
3. En un entorno donde los rewards son muy escasos (solo al final del episodio), ¿qué mejora de DQN ayuda más?
4. Duelling DQN separa $V$ y $A$. ¿Qué ventaja tiene?

## 8. Summary

Value-based methods aprenden $Q(s,a)$ y eligen la mejor acción. Tabular Q-Learning funciona para espacios discretos chicos. DQN extiende Q-learning a espacios grandes con redes neuronales, experience replay, y target networks. La familia DQN (Double, Dueling, PER) resolvió Atari games desde píxeles. La limitación principal: espacios de acción discretos y pequeños. Para acciones continuas o muchas acciones, necesitás policy-based methods.

## 9. Where to Go Next

- [[Policy-Based Methods]] — alternativa para acciones continuas
- [[RL Fundamentals]] — MDP y teoría de valores
- [[Model-Based RL]] — combinando value learning con planificación
- [[RLHF & Preference Optimization]] — Q-learning para alinear LLMs
