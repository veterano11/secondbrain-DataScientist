---
tags: [reinforcement-learning, model-based, planning, world-models]
status: growing
created: 2026-06-27
---

# Model-Based RL

## 1. Escenario de aprendizaje

Imagina que entrenas un agente para jugar un videojuego. Con un enfoque sin modelo (model-free), el agente necesita millones de interacciones con el entorno para aprender algo útil. Pero si el agente pudiera aprender un modelo del entorno —predecir cómo cambia el estado tras cada acción— podría planificar sus movimientos y aprender mucho más rápido. Así funciona AlphaGo: juega miles de partidas contra sí misma en su cabeza, no en el mundo real.

El RL basado en modelos aprende un modelo del entorno y lo usa para planificar, mejorando drásticamente la eficiencia de muestreo. Es la razón por la que AlphaGo y MuZero alcanzan rendimiento sobrehumano con muchas menos partidas reales.

## 2. Conceptos Fundamentales

### El Modelo

Un modelo $M$ aproxima la dinámica del entorno:

$$M(s, a) \rightarrow (\hat{s}', \hat{r}, \text{done})$$

**Tipos de modelos:**
- **Tabla de búsqueda** para espacios de estado pequeños y discretos
- **Regresión lineal** para dinámicas simples
- **[[Neural Networks|Red neuronal]]** para dinámicas complejas (píxeles, física)

```python
class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, state_dim + 1)  # next_state + reward
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        out = self.net(x)
        next_state = out[:, :-1]
        reward = out[:, -1:]
        return next_state, reward
```

### Dyna-Q: Integrando Aprendizaje y Planificación

Dyna-Q es el algoritmo basado en modelos más simple: aprende Q a partir de experiencia real, pero también genera experiencia *simulada* a partir del modelo para actualizar Q aún más.

```python
def dyna_q(env, model, Q, planning_steps=50):
    s = env.reset()
    for step in range(total_steps):
        a = epsilon_greedy(Q[s], epsilon)
        s2, r, done = env.step(a)
        # Aprender modelo a partir de experiencia real
        model.learn(s, a, s2, r)
        # Actualizar Q a partir de experiencia real
        Q[s][a] += alpha * (r + gamma * max(Q[s2]) - Q[s][a])
        # Planificación: alucinar experiencia a partir del modelo
        for _ in range(planning_steps):
            s_plan = random_visited_state()
            a_plan = random_action()
            s2_plan, r_plan = model.predict(s_plan, a_plan)
            Q[s_plan][a_plan] += alpha * (r_plan + gamma * max(Q[s2_plan]) - Q[s_plan][a_plan])
        s = s2
```

### Monte Carlo Tree Search (MCTS)

El algoritmo de planificación detrás de AlphaGo. Construye un árbol de búsqueda incrementalmente:

1. **Selección**: recorre el árbol usando UCB hasta llegar a un nodo hoja
2. **Expansión**: agrega un nuevo nodo hijo
3. **Rollout**: juega aleatoriamente desde la hoja para estimar el valor
4. **Retropropagación**: actualiza los conteos de visita y los valores hacia arriba en el árbol

```python
def mcts(root, num_simulations):
    for _ in range(num_simulations):
        node = root
        path = [node]
        # Selection
        while node.is_expanded and not node.is_terminal:
            node = select_child(node)  # ucb
            path.append(node)
        # Expansion
        if not node.is_terminal:
            node.expand()
            node = random_child(node)
            path.append(node)
        # Rollout
        value = rollout(node)
        # Backpropagation
        for n in reversed(path):
            n.visit_count += 1
            n.total_value += value
    return select_best_action(root)
```

### MuZero

El sucesor de AlphaGo que aprende el modelo *sin* conocer las reglas:

- **Función de representación**: $h(s_t) \rightarrow \text{estado oculto}$
- **Función de dinámica**: $g(h_t, a_{t+1}) \rightarrow (\hat{h}_{t+1}, \hat{r}_{t+1})$
- **Función de predicción**: $f(h_t) \rightarrow (\pi_t, v_t)$

MuZero aprende las tres a partir de auto-juego, planificando mediante MCTS en el espacio latente aprendido. Esto le permite dominar Go, Ajedrez, Shogi y Atari con el mismo algoritmo.

### Dreamer

RL basado en modelos para control visual:

1. **Aprendizaje del modelo del mundo**: entrena un VAE para comprimir píxeles en estados latentes, un predictor recurrente para transiciones y un predictor de recompensa
2. **Aprendizaje del comportamiento**: entrena un actor-critic enteramente sobre *imaginación* (trayectorias latentes del modelo del mundo)

```python
# Bucle conceptual de Dreamer
for epoch in range(num_epochs):
    # Recolectar datos reales
    trajectories = collect_data(agent, env)
    # Entrenar modelo del mundo con datos reales
    world_model.train(trajectories)        # VAE + RSSM
    # Entrenar actor-critic con datos imaginados
    for _ in range(imagination_steps):
        latent_states = world_model.imagine(initial_states, actor)
        actor.update(latent_states)
        critic.update(latent_states)
```

Dreamer logra un rendimiento comparable a los métodos sin modelo con 5–50 veces menos interacciones con el entorno.

### Cuándo el RL Basado en Modelos Funciona y Cuándo Falla

| Funciona bien | Falla |
|---|---|
| Entornos simulados (juegos, simuladores físicos) | Entornos reales con dinámicas difíciles de modelar |
| Tareas donde la interacción es costosa (robótica) | Entornos altamente estocásticos |
| Planificación a largo plazo necesaria | Cuando los errores del modelo se acumulan catastróficamente |
| Acciones discretas con reglas claras | Espacios de acción continuos de alta dimensión |

## 3. Common Pitfalls

- **Model exploitation**: the policy exploits errors in the model (does things that the model thinks work but don't in reality). [[Training Techniques|Ensemble models]] help
- **Compounding error**: one-step prediction is fine, but multi-step rollouts diverge exponentially. Use $\lambda$ returns or short horizons
- **Computational cost**: planning during inference (MCTS) adds latency; distill the planner into a policy network after training
- **Reward model bias**: if the learned reward model is wrong, the policy optimizes the wrong thing

## 4. Check Your Understanding

1. How does Dyna-Q differ from MuZero's approach to model learning?
2. Why does MCTS with UCB balance exploration and exploitation during planning?
3. When would you choose Dreamer over a model-free method like PPO?

## 5. Where to Go Next

- [[Value-Based Methods]] — Dyna is built on Q-learning
- [[Policy-Based Methods]] — Dreamer uses actor-critic in imagination
- [[RL Fundamentals]] — the MDP theory that models approximate
- [[Transfer Learning]] — model-based methods transfer between similar environments
