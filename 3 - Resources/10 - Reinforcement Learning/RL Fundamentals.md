---
tags: [reinforcement-learning, fundamentals, mdp]
status: growing
created: 2026-06-27
---

# RL Fundamentals

## 1. Escenario de aprendizaje

Imaginá que tenés un robot en un laberinto de 3×3. Sale de la posición (0,0) y debe llegar a la meta en (2,2). Cada paso consume batería (reward = -1). Llegar a la meta da reward = +10. Chocar contra una pared da reward = -5 y vuelve al inicio.

El robot no sabe nada del laberinto. Debe descubrir la mejor ruta probando caminos, recordando qué funcionó, y equilibrando exploración (probar rutas nuevas) con explotación (usar la mejor ruta conocida).

Ese es el problema que resuelve Reinforcement Learning.

## 2. El formalismo: Markov Decision Process (MDP)

Un MDP es una tupla de 5 elementos que describe cualquier problema de RL:

$$(\mathcal{S}, \mathcal{A}, P, R, \gamma)$$

| Símbolo | Significado | En el laberinto |
|---------|-------------|-----------------|
| $\mathcal{S}$ | Estados posibles | Las 9 celdas del grid |
| $\mathcal{A}$ | Acciones | Arriba, abajo, izquierda, derecha |
| $P(s'\|s,a)$ | Probabilidad de llegar a $s'$ tomando $a$ en $s$ | 1.0 si la acción es válida, 0 si es pared |
| $R(s,a,s')$ | Recompensa inmediata | -1 por paso, +10 por meta, -5 por pared |
| $\gamma$ | Factor de descuento [0,1) | 0.9 (preferimos recompensas cercanas) |

**Propiedad de Markov**: el futuro solo depende del presente, no del pasado.

$$P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_1, a_1, ..., s_t, a_t)$$

Esto es clave: no necesitamos recordar toda la historia, solo el estado actual.

## 3. Política, valor, retorno

### 3.1 Política $\pi(a|s)$

La estrategia: qué acción tomar en cada estado.

- **Determinística**: $\pi(s) = a$ (en (1,1) siempre ir a la derecha)
- **Estocástica**: $\pi(a|s) = 0.7$ para ir derecha, $0.3$ para las demás

### 3.2 Retorno $G_t$

La suma de recompensas futuras descontadas desde el momento $t$:

$$G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$$

**Ejemplo**: en el laberinto, un camino que tarda 4 pasos hasta la meta ($-1-1-1-1+10$) con $\gamma=0.9$:

$$G = -1 + 0.9(-1) + 0.9^2(-1) + 0.9^3(-1) + 0.9^4(10)$$
$$G = -1 - 0.9 - 0.81 - 0.729 + 6.561 = 3.122$$

### 3.3 Value Functions

**State-value** $V^\pi(s)$: el retorno esperado desde el estado $s$ siguiendo $\pi$.

$$V^\pi(s) = \mathbb{E}_\pi[G_t | S_t = s]$$

**Action-value** $Q^\pi(s,a)$: el retorno esperado tomando acción $a$ en $s$ y luego siguiendo $\pi$.

$$Q^\pi(s,a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a]$$

### 3.4 Bellman Equation

Las value functions tienen una forma recursiva — el valor de un estado depende del valor de los próximos estados:

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^\pi(s')]$$

Esto permite computar $V$ iterativamente sin tener trayectorias completas.

## 4. Exploración vs Explotación

El dilema fundamental: ¿pruebo algo nuevo (quizás mejor) o uso lo que sé que funciona?

### 4.1 $\varepsilon$-greedy

Con probabilidad $\varepsilon$ tomamos una acción aleatoria (explorar). Con $1-\varepsilon$ tomamos la mejor acción conocida (explotar).

```python
def epsilon_greedy(Q, state, epsilon=0.1):
    if np.random.random() < epsilon:
        return np.random.choice(len(Q[state]))  # explorar
    else:
        return np.argmax(Q[state])              # explotar
```

**$\varepsilon$ típico**: 0.1 (10% exploración). Suele decaer con el tiempo.

### 4.2 UCB (Upper Confidence Bound)

En lugar de explorar al azar, UCB prefiere acciones con alta incertidumbre:

$$a_t = \arg\max_a \left[ Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \right]$$

- $N_t(a)$: cuántas veces se probó $a$
- Si una acción se probó pocas veces, el término de incertidumbre crece
- Si se probó muchas veces y es mala, el término decae

### 4.3 Thompson Sampling

Enfoque [[Probability|bayesiano]]: mantenemos una distribución posterior sobre $Q(s,a)$, muestreamos un valor de la posterior, tomamos la mejor acción según esa muestra.

## 5. Tipos de RL

| Categoría | Qué aprende | Ejemplos |
|-----------|-------------|----------|
| **Value-based** | $Q(s,a)$ → política derivada | Q-Learning, DQN |
| **Policy-based** | $\pi(a\|s)$ directamente | REINFORCE, PPO |
| **Actor-Critic** | Ambos $\pi$ y $V$ | A2C, SAC |
| **Model-based** | Modelo del mundo ($P, R$) | MuZero, Dreamer |
| **Model-free** | Sin modelo, directo de experiencia | DQN, PPO |

## 6. Ciclo de entrenamiento paso a paso

```python
def entrenar_agente(env, Q, episodios=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    recompensas_por_episodio = []

    for ep in range(episodios):
        estado, _ = env.reset()
        recompensa_total = 0
        done = False

        while not done:
            # 1. Elegir acción (ε-greedy)
            accion = epsilon_greedy(Q, estado, epsilon)

            # 2. Ejecutar acción en el ambiente
            estado_sig, reward, done, _ = env.step(accion)

            # 3. Actualizar Q (Q-learning update)
            mejor_sig = np.max(Q[estado_sig])
            Q[estado][accion] += alpha * (reward + gamma * mejor_sig - Q[estado][accion])

            # 4. Avanzar
            estado = estado_sig
            recompensa_total += reward

        recompensas_por_episodio.append(recompensa_total)

        if ep % 100 == 0:
            print(f"Episodio {ep}: recompensa = {recompensa_total}")

    return Q, recompensas_por_episodio
```

## 7. Common Mistakes

1. **Reward shaping incorrecto**: si le das reward negativo por cada paso, el robot aprende a terminar rápido, pero quizás también aprende a chocar contra paredes para terminar el episodio antes. Diseñar rewards es arte.

2. **$\gamma$ muy bajo**: con $\gamma=0.5$, las recompensas a más de 5 pasos de distancia prácticamente no importan. Para problemas de largo plazo, $\gamma \geq 0.9$.

3. **$\varepsilon$ constante toda la vida**: al principio necesitás explorar mucho, al final casi nada. Es mejor decaer $\varepsilon$ linealmente.

4. **Usar Q-table con espacios continuos**: no funciona. Para problemas con estados continuos necesitás function approximation ([[Neural Networks|redes neuronales]]).

## 8. Check Your Understanding

1. En el laberinto 3×3, ¿cuántos pares (estado, acción) hay? (9 estados × 4 acciones = 36 valores Q)
2. Si $\gamma=0$, ¿qué política óptima esperás? (La que maximiza la recompensa inmediata — ignora el futuro)
3. ¿Por qué Q-learning se considera off-policy? (Aprende la política óptima mientras sigue una política exploratoria ε-greedy)
4. En UCB, cuando $N_t(a)=0$, ¿cuánto vale la incertidumbre?

**Respuestas rápidas:**
2. Con $\gamma=0$, el robot solo le importa el siguiente paso. Probablemente no llegue a la meta si requiere varios pasos.
3. Off-policy: aprende sobre $\pi^*$ mientras ejecuta $\pi_{behavior}$. SARSA es on-policy: aprende sobre la política que está ejecutando.
4. Es infinita — UCB garantiza que cada acción se pruebe al menos una vez.

## 9. Summary

RL formaliza el aprendizaje por prueba y error como un MDP: estados, acciones, transiciones, recompensas. El agente busca maximizar el retorno descontado aprendiendo una política. El dilema exploración-explotación se resuelve con ε-greedy, UCB, o Thompson sampling. Value-based methods aprenden $Q(s,a)$, policy-based aprenden $\pi(a|s)$ directamente. Todo RL se reduce a este framework.

## 10. Where to Go Next

- [[Value-Based Methods]] — Q-Learning y DQN
- [[Policy-Based Methods]] — REINFORCE, PPO, Actor-Critic
- [[Model-Based RL]] — planificación con modelos del mundo
- [[RLHF & Preference Optimization]] — RL para alinear LLMs
- [[Markov Chain Monte Carlo]] — MCMC comparte el formalismo de Markov
