---
tags: [optimization, discrete, constraints, linear-programming]
status: growing
created: 2026-06-27
---

# Constraint & Discrete Optimization

## Motivation

Not all optimization variables are continuous and smooth. Many real-world problems involve discrete choices (which warehouse to open, which items to pack, which route to take) or constraints that make standard gradient methods inapplicable. This note covers the tools for combinatorial and constrained problems.

## Core Concepts

### Linear Programming (LP)

$$\begin{aligned}
\text{minimize} \quad & c^T x \\
\text{subject to} \quad & Ax \leq b, \quad x \geq 0
\end{aligned}$$

All variables are continuous. LP es un problema fundamental de [[Linear Algebra]], resuelto eficientemente por el simplex o métodos de punto interior (polynomial time).

```python
from scipy.optimize import linprog

c = [-5, -3]           # objective: maximize 5x + 3y
A = [[2, 1], [1, 1]]
b = [40, 30]
res = linprog(c, A_ub=A, b_ub=b, bounds=[(0, None), (0, None)])
print(res.fun, res.x)
```

### Integer Linear Programming (ILP)

$x \in \mathbb{Z}^n$ (or a subset). NP-hard in general.

$$\begin{aligned}
\text{minimize} \quad & c^T x \\
\text{subject to} \quad & Ax \leq b, \quad x \in \mathbb{Z}^n_{\geq 0}
\end{aligned}$$

**Common approach**: solve LP relaxation, then use branch-and-bound:

1. Solve LP relaxation (ignore integrality)
2. If solution is integer → optimal
3. Otherwise, branch: fix a fractional variable to 0 or 1, solve two subproblems
4. Prune using bounds

```python
from pulp import *

prob = LpProblem("Knapsack", LpMaximize)
x = [LpVariable(f"x_{i}", cat='Binary') for i in range(n)]
prob += lpSum([values[i] * x[i] for i in range(n)])
prob += lpSum([weights[i] * x[i] for i in range(n)]) <= capacity
prob.solve()
```

### Mixed-Integer Programming (MIP)

Mix of continuous and integer variables. Solved with branch-and-cut (branch-and-bound + cutting planes).

**Common solvers**: Gurobi, CPLEX (commercial, but free for academics), SCIP, CBC (open source).

**OR-Tools** (Google): free, competitive solver for practical problems.

### Dynamic Programming

Optimal substructure + overlapping subproblems. Canonical for sequential decision problems.

**Knapsack (0/1)**:

```python
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w-weights[i-1]] + values[i-1])
            else:
                dp[i][w] = dp[i-1][w]
    return dp[n][capacity]
```

**Time complexity**: $\mathcal{O}(n \cdot \text{capacity})$ — pseudo-polynomial (capacity is numeric, not input length).

### Branch and Bound (Exact)

Systematic enumeration with pruning:

1. **Upper bound**: best known feasible solution
2. **Lower bound**: relaxation (LP, Lagrangian)
3. Prune if lower bound ≥ upper bound

### Greedy & Approximation Algorithms

| Problem | Greedy Approach | Guarantee |
|---------|----------------|-----------|
| Graph coloring | Sequential coloring | $\leq \Delta + 1$ colors |
| Set cover | Pick set covering most uncovered | $O(\log n)$ approximation |
| Vertex cover | Pick edge endpoints | 2-approximation |
| Max cut | Random assignment | 0.5-approximation (expected) |

### Metaheuristics

For problems where exact methods are too slow:

- **Genetic algorithms**: crossover + mutation on a population of solutions
- **Simulated annealing**: accept worse solutions with decreasing [[Probability|probability]] to escape local minima
- **Ant colony optimization**: agents deposit pheromones along good paths
- **Constraint programming**: inference + search for satisfaction problems (CSP)

```python
# Simulated annealing for TSP
def simulated_annealing(dist, temp_start=100, temp_end=0.1, cooling=0.995):
    n = len(dist)
    tour = list(range(n))
    random.shuffle(tour)
    best_tour = tour[:]
    best_cost = tour_cost(tour, dist)
    temp = temp_start

    while temp > temp_end:
        i, j = random.sample(range(n), 2)
        tour[i], tour[j] = tour[j], tour[i]  # swap
        new_cost = tour_cost(tour, dist)
        if new_cost < best_cost or random.random() < exp((best_cost - new_cost) / temp):
            best_cost = new_cost
            best_tour = tour[:]
        else:
            tour[i], tour[j] = tour[j], tour[i]  # undo
        temp *= cooling

    return best_tour, best_cost
```

### Applications in ML & Data Science

| Problem | Optimization Type | Use Case |
|---------|-----------------|----------|
| Feature selection | MIP (binary variables) | Select $k$ of $p$ features en [[Supervised Learning]] |
| Model compression | Integer programming | Choose which weights to prune |
| Resource allocation | LP / MIP | Budget-constrained A/B test assignment |
| Hyperparameter tuning | Discrete/continuous mix | Not discrete, but many HPs are integer |
| Clustering (k-medoids) | MIP | Assign points to exactly $k$ clusters |
| Graph neural nets | Constrained | Molecule property prediction with valency constraints |

## Common Pitfalls

- **ILP vs LP**: solving an LP and rounding doesn't guarantee optimal or even feasible integer solutions. Use proper MIP solvers
- **NP-hardness in practice**: an NP-hard problem with $n=50$ can be trivial with good MIP solvers; $n=1000$ can be impossible. Always test
- **Branch-and-bound patience**: the solver can "hang" for hours. Set a time limit
- **Memory blowup**: DP for large capacity values (e.g., $C = 10^9$) is impossible. Use meet-in-the-middle or MIP

## Check Your Understanding

1. How does branch-and-bound prove optimality without enumerating all $2^n$ solutions?
2. Why is the knapsack problem NP-hard even though its DP is $O(nC)$?
3. If your optimization has 80% continuous variables and 20% binary variables, which solver approach would you try?

## Where to Go Next

- [[Convex Optimization]] — theory for continuous relaxations
- [[Hyperparameter Tuning]] — discrete HPs (batch size, number of layers)
- [[Gradient-Based Optimization]] — continuous vs discrete landscapes
- [[RL Fundamentals]] — dynamic programming for MDPs
