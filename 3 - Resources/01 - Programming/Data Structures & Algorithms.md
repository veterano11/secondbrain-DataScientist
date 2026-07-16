---
tags: [programming, algorithms, foundational]
status: growing
created: 2026-06-27
---

# Estructuras de Datos y Algoritmos

## 1. Escenario de aprendizaje

Trabajas en una empresa de comercio electrónico con 10 millones de productos. El equipo de negocio te pide: "encuentra productos similares a este en menos de 100 milisegundos". Usar una lista para buscar entre 10 millones de elementos toma segundos; un conjunto (hash set) lo hace en microsegundos. La diferencia entre O(1) y O(n) no importa con 100 filas, pero con 100 millones de filas es la diferencia entre milisegundos y horas. Elegir la estructura de datos correcta es la habilidad más práctica que puedes desarrollar.

En ciencia de datos, entender las estructuras de datos te ayuda a:
- Elegir el contenedor adecuado para tus datos (lista, conjunto, diccionario, array)
- Saber por qué pandas es rápido (usa arrays de NumPy internamente)
- Diseñar bucles de ingeniería de características eficientes
- Entender la indexación de bases de datos (índices hash, B-trees)
- Aprobar entrevistas técnicas — pero más importante, escribir código de producción eficiente

---

## 2. Notación Big O — El lenguaje del rendimiento

Big O describe cómo crecen el tiempo de ejecución o la memoria **a medida que aumenta el tamaño de la entrada**.

### 2.1 La intuición

No cuentes operaciones exactas — cuenta **cómo escala el algoritmo**.

| Notación | Nombre | Si la entrada se duplica... |
|---|---|---|
| O(1) | Constante | El tiempo permanece igual |
| O(log n) | Logarítmica | El tiempo aumenta en 1 paso |
| O(n) | Lineal | El tiempo se duplica |
| O(n log n) | Linealítmica | El tiempo aumenta un poco más del doble |
| O(n²) | Cuadrática | El tiempo se cuadruplica |
| O(2ⁿ) | Exponencial | El tiempo se vuelve enorme |

### 2.2 Ejemplos concretos

```python
# O(1) — tiempo constante
def get_first(items):
    return items[0]          # acceso directo a memoria

# O(n) — tiempo lineal
def contains(items, target):
    for item in items:       # podría revisar todos los elementos
        if item == target:
            return True
    return False

# O(n²) — tiempo cuadrático
def pairwise_distance(points):
    result = []
    for p1 in points:        # n veces
        for p2 in points:    # n veces = n² total
            result.append(dist(p1, p2))
    return result

# O(log n) — tiempo logarítmico
def binary_search(sorted_list, target):
    low, high = 0, len(sorted_list) - 1
    while low <= high:       # divide el espacio de búsqueda a la mitad cada iteración
        mid = (low + high) // 2
        if sorted_list[mid] == target:
            return mid
        elif sorted_list[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
```

### 2.3 Por qué importa Big O en ciencia de datos

| Operación | Estructura de datos | Big O | Ejemplo |
|---|---|---|---|
| Búsqueda por clave | dict | O(1) | Búsqueda de características, caché |
| Búsqueda por índice | list | O(1) | Acceso a muestras de entrenamiento |
| Prueba de pertenencia | set | O(1) | Verificar si un ID existe |
| Prueba de pertenencia | list | O(n) | Lento para listas grandes |
| Ordenamiento | list | O(n log n) | Ordenar características por importancia |
| Encontrar máximo/mínimo | list | O(n) | Encontrar el mejor/peor caso |

La lección: si necesitas verificar pertenencia frecuentemente, usa un **set** (O(1)), no una **list** (O(n)).

---

## 3. Estructuras de datos esenciales

### 3.1 Array / Lista

- **Memoria**: bloque contiguo de memoria
- **Acceso**: O(1) por índice (cálculo directo de dirección de memoria)
- **Insertar/eliminar al final**: O(1) amortizado
- **Insertar/eliminar al inicio**: O(n) (desplazar todos los elementos siguientes)

**En Python**: `list` es un array dinámico. Cuando se queda sin espacio, asigna ~1.125× más memoria y copia todo — de ahí el O(1) "amortizado" para append.

```python
arr = [1, 2, 3]
arr.append(4)            # O(1) amortizado
arr.insert(0, 0)         # O(n) — desplaza todo
```

**Arrays de NumPy**: tipo fijo, memoria contigua, amigables con la caché. Mucho más rápidos para operaciones numéricas porque las operaciones están vectorizadas (se ejecutan en C).

```python
arr = np.array([1, 2, 3])
arr * 2                  # vectorizado — se ejecuta en C, no en Python
```

### 3.2 Tabla Hash (dict / set)

**Cómo funciona**: una función hash mapea claves a índices del array. Las colisiones se manejan mediante encadenamiento (varias claves en el mismo índice almacenadas en una lista enlazada).

- **Insertar**: O(1) promedio, O(n) peor caso (muchas colisiones)
- **Buscar**: O(1) promedio, O(n) peor caso
- **Eliminar**: O(1) promedio

**La magia**: las buenas funciones hash distribuyen las claves uniformemente, haciendo que el peor caso sea prácticamente imposible.

```python
lookup = {"alice": 25, "bob": 30, "charlie": 35}
lookup["alice"]           # O(1) — hashea "alice", salta a esa posición

# set es un dict solo con claves (sin valores)
ids = {101, 102, 103}
102 in ids                # O(1)
```

### 3.3 Pila (LIFO)

Último en entrar, primero en salir. En Python, una `list` funciona perfectamente como pila.

```python
stack = []
stack.append(1)           # push
stack.append(2)
stack.pop()               # 2 — último en entrar, primero en salir
```

**Casos de uso**: búsqueda en profundidad, operaciones de deshacer, verificación de paréntesis.

### 3.4 Cola (FIFO)

Primero en entrar, primero en salir. Usa `collections.deque` para O(1) en append/pop desde ambos extremos.

```python
from collections import deque

queue = deque()
queue.append(1)            # encolar (lado derecho)
queue.append(2)
queue.popleft()            # 1 — primero en entrar, primero en salir
```

**Casos de uso**: búsqueda en anchura, planificación de tareas, datos en streaming.

### 3.5 Árbol

Una estructura jerárquica con una raíz e hijos.

**Árbol Binario de Búsqueda (BST)**:
- Hijo izquierdo < padre < hijo derecho
- Búsqueda: O(log n) promedio, O(n) peor caso (desequilibrado)
- Variantes balanceadas (AVL, Rojo-Negro) garantizan O(log n)

En ciencia de datos, los árboles aparecen como **árboles de decisión** (y sus ensembles: Random Forest, XGBoost). [[Object-Oriented Programming]] cubre la implementación de estructuras de árbol con clases.

### 3.6 Heap (Cola de Prioridad)

Siempre te da el elemento más pequeño (min-heap) o más grande (max-heap).

```python
import heapq

data = [5, 3, 7, 1, 9]
heapq.heapify(data)        # min-heap: [1, 3, 7, 5, 9]
heapq.heappop(data)        # 1 (el más pequeño)
heapq.heappush(data, 2)    # inserción O(log n)
```

**Casos de uso**: encontrar los k elementos principales, algoritmo de Dijkstra, planificación por prioridad.

---

## 4. Algoritmos clave para ciencia de datos

### 4.1 Ordenamiento

Python usa **Timsort** (O(n log n) peor caso), una combinación de merge sort e insertion sort optimizada para datos del mundo real (que a menudo tienen orden parcial).

```python
sorted_list = sorted(unsorted_list)       # retorna una nueva lista
unsorted_list.sort()                      # ordena in-place
```

### 4.2 Búsqueda

- **Búsqueda lineal**: O(n) — datos no ordenados
- **Búsqueda binaria**: O(log n) — requiere datos ordenados

```python
import bisect

sorted_data = [1, 3, 5, 7, 9]
pos = bisect.bisect_left(sorted_data, 6)  # 3 (entre 5 y 7)
```

### 4.3 Programación Dinámica

"Resolver un problema dividiéndolo en subproblemas superpuestos y resolviendo cada uno una sola vez."

**Fibonacci — naive (exponencial) vs DP (lineal)**:

```python
# Sin DP: O(2ⁿ) — recalcula los mismos valores exponencialmente
def fib_naive(n):
    return n if n <= 1 else fib_naive(n-1) + fib_naive(n-2)

# Con memoización: O(n) — cada valor se calcula una vez
from functools import lru_cache

@lru_cache(maxsize=None)
def fib_memo(n):
    return n if n <= 1 else fib_memo(n-1) + fib_memo(n-2)
```

**En ciencia de datos**: distancia de edición (NLP), algoritmo de Viterbi (HMM), dynamic time warping (series temporales), alineamiento de secuencias (bioinformática). [[Functional Programming]] y sus funciones puras se alinean naturalmente con los subproblemas sin estado de la programación dinámica.

---

## 5. Errores Comunes

1. **Usar `list` donde `set` es apropiado**: si solo necesitas pruebas de pertenencia, `set` es O(1) vs `list` O(n). Con 1M de elementos, la diferencia es microsegundos vs segundos.

2. **No entender el costo amortizado**: `list.append()` es O(1) en promedio. Pero ocasionalmente es O(n) al redimensionar. Esto importa para aplicaciones sensibles a la latencia.

3. **Asumir que O(n) es siempre malo**: para n pequeño (n < 1000), O(n) puede ser más rápido que O(log n) con factores constantes altos. Perfil antes de optimizar. [[Calidad de Código]] cubre herramientas de perfilado para medir rendimiento del mundo real.

4. **Profundidad de recursión**: Python tiene un límite de recursión (~1000). Usar iteración para recursión profunda. La programación dinámica de todas formas evita la recursión.

---

## 6. Verifica tu Comprensión

1. Necesitas verificar si un ID de usuario específico existe en una colección de 10 millones de IDs. ¿Usas una lista o un set? ¿Por qué?
2. ¿Cuál es el Big O de `dict.get(key)`? ¿Cómo logra esto una tabla hash?
3. Un árbol de decisión entrenado con 1000 muestras toma 0.1s. ¿Aproximadamente cuánto para 1,000,000 de muestras? (Pista: piensa en la complejidad del árbol)
4. ¿Cuándo sería O(n) más rápido que O(1) en la práctica?
5. ¿Por qué `list.append()` se documenta como "amortized O(1)"? ¿Qué significa "amortizado"?

---

## 7. Resumen

Las estructuras de datos se tratan de compensaciones: los arrays dan acceso rápido pero inserción lenta; las tablas hash dan búsqueda rápida pero usan más memoria; los árboles dan acceso ordenado pero tienen sobrecarga. La notación Big O cuantifica estas compensaciones. La lección más práctica: usa `dict`/`set` para búsquedas rápidas, `list` para secuencias ordenadas, `deque` para colas, y `heapq` para prioridad. Entender estas decisiones separa el código eficiente del código lento.

---

## 8. Dónde Ir Ahora

- [[Fundamentos de Python]] — Listas, diccionarios, sets en la práctica
- [[Python para Ciencia de Datos]] — Arrays de NumPy optimizados para datos numéricos
- [[Aprendizaje Supervisado]] — Los árboles de decisión son árboles binarios
- [[Programación Orientada a Objetos]] — Implementar estructuras de datos como clases
- [[Programación Funcional]] — Funciones puras para diseño de algoritmos
