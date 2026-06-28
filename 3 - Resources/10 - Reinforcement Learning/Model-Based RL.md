---
tags: [reinforcement-learning, model-based, planning, world-models]
status: growing
created: 2026-06-27
---

# Model-Based RL

## Motivation

Model-free RL (DQN, PPO) learns directly from experience, requiring millions of interactions. Model-based RL learns a *model* of the environment and uses it for planning — dramatically improving sample efficiency. This is how AlphaGo and MuZero achieve superhuman performance with far fewer real games.

## Core Concepts

### The Model

A model $M$ approximates the environment dynamics:

$$M(s, a) \rightarrow (\hat{s}', \hat{r}, \text{done})$$

**Types of models:**
- **Lookup table** for discrete, small state spaces
- **Linear regression** for simple dynamics
- **[[Neural Networks|Neural network]]** for complex dynamics (pixels, physics)

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

### Dyna-Q: Integrating Learning and Planning

Dyna-Q is the simplest model-based algorithm: learn Q from real experience, but also generate *simulated* experience from the model to update Q further.

```python
def dyna_q(env, model, Q, planning_steps=50):
    s = env.reset()
    for step in range(total_steps):
        a = epsilon_greedy(Q[s], epsilon)
        s2, r, done = env.step(a)
        # Learn model from real experience
        model.learn(s, a, s2, r)
        # Update Q from real experience
        Q[s][a] += alpha * (r + gamma * max(Q[s2]) - Q[s][a])
        # Planning: hallucinate experience from model
        for _ in range(planning_steps):
            s_plan = random_visited_state()
            a_plan = random_action()
            s2_plan, r_plan = model.predict(s_plan, a_plan)
            Q[s_plan][a_plan] += alpha * (r_plan + gamma * max(Q[s2_plan]) - Q[s_plan][a_plan])
        s = s2
```

### Monte Carlo Tree Search (MCTS)

The planning algorithm behind AlphaGo. Builds a search tree incrementally:

1. **Selection**: traverse tree using UCB until a leaf node
2. **Expansion**: add a new child node
3. **Rollout**: play randomly from the leaf to estimate value
4. **Backpropagation**: update visit counts and values up the tree

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

AlphaGo's successor that learns the model *without* being given the rules:

- **Representation function**: $h(s_t) \rightarrow \text{hidden state}$
- **Dynamics function**: $g(h_t, a_{t+1}) \rightarrow (\hat{h}_{t+1}, \hat{r}_{t+1})$
- **Prediction function**: $f(h_t) \rightarrow (\pi_t, v_t)$

MuZero learns all three from self-play, planning via MCTS in the learned latent space. This means it can master Go, Chess, Shogi, and Atari with the same algorithm.

### Dreamer

World Models-based RL for visual control:

1. **World model learning**: train a VAE to compress pixels into latent states, a recurrent predictor for transitions, and a reward predictor
2. **Behavior learning**: train an actor-critic entirely on *imagination* (latent trajectories from the world model)

```python
# Conceptual Dreamer loop
for epoch in range(num_epochs):
    # Collect real data
    trajectories = collect_data(agent, env)
    # Train world model on real data
    world_model.train(trajectories)        # VAE + RSSM
    # Train actor-critic on imagined data
    for _ in range(imagination_steps):
        latent_states = world_model.imagine(initial_states, actor)
        actor.update(latent_states)
        critic.update(latent_states)
```

Dreamer achieves comparable performance to model-free methods with 5–50× fewer environment interactions.

### When Model-Based Excels vs When It Fails

| Excels | Fails |
|--------|-------|
| Simulated environments (games, physics sims) | Real-world with hard-to-model dynamics |
| Tasks where interaction is expensive (robotics) | Highly stochastic environments |
| Long-horizon planning needed | When model errors compound catastrophically |
| Discrete actions with clear rules | Continuous high-dimensional action spaces |

## Common Pitfalls

- **Model exploitation**: the policy exploits errors in the model (does things that the model thinks work but don't in reality). [[Training Techniques|Ensemble models]] help
- **Compounding error**: one-step prediction is fine, but multi-step rollouts diverge exponentially. Use $\lambda$ returns or short horizons
- **Computational cost**: planning during inference (MCTS) adds latency; distill the planner into a policy network after training
- **Reward model bias**: if the learned reward model is wrong, the policy optimizes the wrong thing

## Check Your Understanding

1. How does Dyna-Q differ from MuZero's approach to model learning?
2. Why does MCTS with UCB balance exploration and exploitation during planning?
3. When would you choose Dreamer over a model-free method like PPO?

## Where to Go Next

- [[Value-Based Methods]] — Dyna is built on Q-learning
- [[Policy-Based Methods]] — Dreamer uses actor-critic in imagination
- [[RL Fundamentals]] — the MDP theory that models approximate
- [[Transfer Learning]] — model-based methods transfer between similar environments
