---
tags: [optimization, discrete, constraints, linear-programming]
status: growing
created: 2026-06-27
---

# Constraint & Discrete Optimization

## 1. Escenario de aprendizaje

Trabajas en la planificación logística de una empresa: necesitas decidir qué almacenes abrir, qué rutas de entrega usar y qué productos asignar a cada camión. Estas son decisiones discretas —abres o no abres un almacén— y están llenas de restricciones: capacidad, presupuesto, tiempo de entrega. Los métodos de optimización continua, como el descenso por gradiente, no funcionan aquí.

La optimización con restricciones y discreta es la herramienta para problemas combinatorios: seleccionar un subconjunto de opciones bajo restricciones del mundo real.

## 2. Conceptos Fundamentales

### Programación Lineal (LP)

$$\begin{aligned}
\text{minimizar} \quad & c^T x \\
\text{sujeto a} \quad & Ax \leq b, \quad x \geq 0
\end{aligned}$$

Todas las variables son continuas. LP es un problema fundamental de [[Linear Algebra]], resuelto eficientemente por el simplex o métodos de punto interior (tiempo polinomial).

```python
from scipy.optimize import linprog

c = [-5, -3]           # objetivo: maximizar 5x + 3y
A = [[2, 1], [1, 1]]
b = [40, 30]
res = linprog(c, A_ub=A, b_ub=b, bounds=[(0, None), (0, None)])
print(res.fun, res.x)
```

### Programación Lineal Entera (ILP)

$x \in \mathbb{Z}^n$ (o un subconjunto). NP-difícil en general.

$$\begin{aligned}
\text{minimizar} \quad & c^T x \\
\text{sujeto a} \quad & Ax \leq b, \quad x \in \mathbb{Z}^n_{\geq 0}
\end{aligned}$$

**Enfoque común**: resolver la relajación LP, luego usar branch-and-bound:

1. Resolver la relajación LP (ignorar la integralidad)
2. Si la solución es entera → óptima
3. Si no, ramificar: fijar una variable fraccionaria a 0 o 1, resolver dos subproblemas
4. Podar usando cotas

```python
from pulp import *

prob = LpProblem("Mochila", LpMaximize)
x = [LpVariable(f"x_{i}", cat='Binary') for i in range(n)]
prob += lpSum([values[i] * x[i] for i in range(n)])
prob += lpSum([weights[i] * x[i] for i in range(n)]) <= capacity
prob.solve()
```

### Programación Mixta-Entera (MIP)

Mezcla de variables continuas y enteras. Se resuelve con branch-and-cut (branch-and-bound + planos de corte).

**Solvers comunes**: Gurobi, CPLEX (comerciales, pero gratuitos para académicos), SCIP, CBC (código abierto).

**OR-Tools** (Google): solver gratuito y competitivo para problemas prácticos.

### Programación Dinámica

Subestructura óptima + subproblemas superpuestos. Canónico para problemas de decisión secuencial.

**Mochila (0/1)**:

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

**Complejidad temporal**: $\mathcal{O}(n \cdot \text{capacidad})$ — pseudo-polinomial (la capacidad es numérica, no longitud de la entrada).

### Branch and Bound (Exacto)

Enumeración sistemática con poda:

1. **Cota superior**: mejor solución factible conocida
2. **Cota inferior**: relajación (LP, Lagrangiana)
3. Podar si la cota inferior ≥ cota superior

### Algoritmos Voraces y de Aproximación

| Problema | Enfoque Voraz | Garantía |
|---|---|---|
| Coloreo de grafos | Coloreo secuencial | $\leq \Delta + 1$ colores |
| Set cover | Elegir el conjunto que cubre más elementos no cubiertos | $O(\log n)$ aproximación |
| Vertex cover | Elegir extremos de aristas | 2-aproximación |
| Max cut | Asignación aleatoria | 0.5-aproximación (esperada) |

### Metaheurísticas

Para problemas donde los métodos exactos son demasiado lentos:

- **Algoritmos genéticos**: cruce + mutación sobre una población de soluciones
- **Recocido simulado (Simulated Annealing)**: acepta soluciones peores con [[Probability|probabilidad]] decreciente para escapar de mínimos locales
- **Optimización por colonia de hormigas**: agentes depositan feromonas a lo largo de buenas rutas
- **Programación con restricciones**: inferencia + búsqueda para problemas de satisfacción (CSP)

```python
# Recocido simulado para el TSP
def simulated_annealing(dist, temp_start=100, temp_end=0.1, cooling=0.995):
    n = len(dist)
    tour = list(range(n))
    random.shuffle(tour)
    best_tour = tour[:]
    best_cost = tour_cost(tour, dist)
    temp = temp_start

    while temp > temp_end:
        i, j = random.sample(range(n), 2)
        tour[i], tour[j] = tour[j], tour[i]  # intercambiar
        new_cost = tour_cost(tour, dist)
        if new_cost < best_cost or random.random() < exp((best_cost - new_cost) / temp):
            best_cost = new_cost
            best_tour = tour[:]
        else:
            tour[i], tour[j] = tour[j], tour[i]  # deshacer
        temp *= cooling

    return best_tour, best_cost
```

### Aplicaciones en ML y Ciencia de Datos

| Problema | Tipo de Optimización | Caso de Uso |
|---|---|---|
| Selección de características | MIP (variables binarias) | Seleccionar $k$ de $p$ características en [[Supervised Learning]] |
| Compresión de modelos | Programación entera | Elegir qué pesos podar |
| Asignación de recursos | LP / MIP | Asignación de tests A/B con presupuesto limitado |
| Ajuste de hiperparámetros | Mixto discreto/continuo | No es discreto puro, pero muchos HPs son enteros |
| Clustering (k-medoids) | MIP | Asignar puntos a exactamente $k$ clusters |
| Redes neuronales en grafos | Con restricciones | Predicción de propiedades moleculares con restricciones de valencia |

## 3. Errores Comunes

- **ILP vs LP**: resolver un LP y redondear no garantiza soluciones enteras óptimas ni factibles. Usa solvers MIP adecuados.
- **NP-hardness en la práctica**: un problema NP-hard con $n=50$ puede ser trivial con buenos solvers MIP; con $n=1000$ puede ser imposible. Siempre prueba.
- **Paciencia con branch-and-bound**: el solver puede "colgarse" por horas. Establece un límite de tiempo.
- **Explosión de memoria**: PD para valores de capacidad grandes (ej. $C = 10^9$) es imposible. Usa meet-in-the-middle o MIP.

## 4. Resumen

1. La optimización discreta modela problemas con decisiones binarias o enteras (abrir/cerrar, seleccionar, asignar).
2. LP se resuelve eficientemente; ILP/MIP son NP-difíciles pero manejables con branch-and-bound y solvers modernos.
3. La programación dinámica resuelve problemas con subestructura óptima en tiempo pseudo-polinomial.
4. Los algoritmos voraces y de aproximación ofrecen soluciones rápidas con garantías de calidad.
5. Las metaheurísticas (genéticos, recocido simulado) son útiles cuando los métodos exactos son muy lentos.

## 5. Check Your Understanding

1. ¿Cómo prueba branch-and-bound la optimalidad sin enumerar las $2^n$ soluciones?
2. ¿Por qué el problema de la mochila es NP-hard si su PD es $O(nC)$?
3. Si tu optimización tiene 80% variables continuas y 20% binarias, ¿qué enfoque de solver probarías?

## 6. Where to Go Next

- [[Convex Optimization]] — teoría para relajaciones continuas
- [[Hyperparameter Tuning]] — HPs discretos (batch size, número de capas)
- [[Gradient-Based Optimization]] — paisajes continuos vs discretos
- [[RL Fundamentals]] — programación dinámica para MDPs
