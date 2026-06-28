---
tags: [programming, functional, core]
status: growing
created: 2026-06-27
---

# Programación Funcional

## 1. Escenario de aprendizaje

Tienes datos sin procesar de múltiples fuentes: CSVs, APIs, bases de datos. Necesitas aplicar una secuencia de transformaciones: limpiar valores nulos, normalizar columnas, codificar variables categóricas y calcular métricas. Cada transformación debe ser testeable de forma independiente y el pipeline completo debe poder ejecutarse en paralelo sin efectos secundarios. La programación funcional (FP) no es solo una curiosidad académica — es el **paradigma natural para el procesamiento de datos**.

Cada operación de pandas (`map`, `apply`, `groupby`), cada transformación de Spark y cada operación vectorizada de NumPy sigue principios funcionales. Si OOP se trata de organizar código alrededor de **objetos** (estado + comportamiento), FP se trata de organizar código alrededor de **transformaciones de datos** (entrada → salida, sin efectos secundarios). Entender FP te permite escribir código de datos más predecible, testeable y paralelizable.

---

## 2. Conceptos fundamentales

### 2.1 Funciones puras

Una función pura tiene dos propiedades:
1. **Misma entrada → misma salida** (sin estado oculto)
2. **Sin efectos secundarios** (no modifica el estado externo)

```python
# Pura
def add(a, b):
    return a + b

# Impura — depende del estado externo
total = 0
def add_to_total(x):
    global total
    total += x
    return total
```

**Por qué importan las funciones puras en ciencia de datos:**
- **Testeables**: sin configuración previa, solo llamar con entradas y verificar salidas
- **Almacenables en caché**: la misma entrada siempre da la misma salida → `@functools.lru_cache`
- **Paralelizables**: sin estado compartido, seguras para ejecutar en cualquier orden
- **Componibles**: combina funciones pequeñas en otras más grandes — [[Testing for Data Science]] aprovecha estas propiedades para pruebas confiables

### 2.2 Inmutabilidad

Los datos no deben modificarse — en su lugar, crea nuevos datos con los cambios aplicados.

```python
# Mutable (estilo OOP)
df["new_col"] = df["a"] + df["b"]   # modifica df in-place

# Immutable (estilo FP)
new_df = df.assign(new_col=df["a"] + df["b"])  # retorna un nuevo DataFrame
```

**Por qué importa la inmutabilidad:**
- Previene la mutación accidental (una fuente común de errores)
- Hace que el código sea más fácil de razonar (los datos solo fluyen hacia adelante)
- Permite la detección de cambios (comparar versión antigua vs nueva)

En pandas: `df.assign()` (inmutable) vs `df["col"] = ...` (mutable). En PyTorch: `tensor.clone()` y `torch.no_grad()` imponen la separación entre cómputo y modificación.

### 2.3 Funciones de primera clase y de orden superior

**Funciones de primera clase**: las funciones son valores — pueden asignarse a variables, pasarse como argumentos y retornarse desde otras funciones.

```python
def square(x): return x ** 2
def cube(x): return x ** 3

operations = [square, cube]  # funciones en una lista
for op in operations:
    print(op(3))             # 9, 27
```

**Funciones de orden superior**: funciones que toman otras funciones como argumentos o retornan funciones.

```python
def apply_transformation(func, data):
    return [func(x) for x in data]

apply_transformation(square, [1, 2, 3])  # [1, 4, 9]
```

Esta es la base de `map`, `filter`, `reduce` y `apply` de pandas.

---

## 3. Los tres pilares: Map, Filter, Reduce

Estas tres funciones forman el núcleo del procesamiento de datos en FP.

### 3.1 map — Transformar cada elemento

**Patrón**: tomar una colección, aplicar una función a cada elemento, retornar una nueva colección del mismo tamaño.

```python
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x ** 2, numbers))    # estilo FP
squared = [x ** 2 for x in numbers]                # pitónico (preferido)

# equivalente en pandas
df["normalized"] = df["value"].map(lambda x: (x - min_) / (max_ - min_))
```

**Nota pitónica**: las list comprehensions son generalmente preferidas sobre `map()` en Python porque son más legibles. Usa `map()` cuando ya tengas una función con nombre.

### 3.2 filter — Conservar elementos que coinciden

**Patrón**: tomar una colección, conservar los elementos que satisfacen un predicado, retornar una colección (potencialmente más pequeña).

```python
numbers = [1, 2, 3, 4, 5, 6]
evens = list(filter(lambda x: x % 2 == 0, numbers))  # [2, 4, 6]
evens = [x for x in numbers if x % 2 == 0]           # pitónico (preferido)
```

### 3.3 reduce — Acumular

**Patrón**: tomar una colección, combinar elementos secuencialmente en un solo valor.

```python
from functools import reduce

numbers = [1, 2, 3, 4]
product = reduce(lambda a, b: a * b, numbers)  # 1 * 2 * 3 * 4 = 24

# Uso real: encadenar transformaciones
from functools import reduce

def pipeline(data, *functions):
    return reduce(lambda x, f: f(x), functions, data)

result = pipeline(raw_data, clean, normalize, encode)
# Equivalente a: encode(normalize(clean(raw_data)))
```

`reduce` es menos común en Python (Clojure, Haskell lo usan mucho) pero invaluable para componer pipelines.

---

## 4. Composición de funciones en ciencia de datos

### 4.1 El patrón Pipeline

Los pipelines FP encadenan transformaciones:

```python
# Sin composición (se lee de abajo arriba o de adentro hacia afuera)
predictions = model.predict(encode(normalize(clean(raw_data))))

# Con composición (se lee de arriba abajo)
def compose(*functions):
    def composed(data):
        for f in functions:
            data = f(data)
        return data
    return composed

pipeline = compose(clean, normalize, encode, model.predict)
predictions = pipeline(raw_data)
```

Este patrón componible es central para la API `Pipeline` de scikit-learn en [[Python for Data Science]].

### 4.2 Aplicación parcial

Fija algunos argumentos de una función, creando una nueva función con menos argumentos:

```python
from functools import partial

def scale(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)

scale_age = partial(scale, min_val=0, max_val=100)
scale_income = partial(scale, min_val=20000, max_val=200000)

ages_normalized = list(map(scale_age, ages))
```

### 4.3 Currying

Convierte una función que toma múltiples argumentos en una cadena de funciones que cada una toma un argumento:

```python
def multiply(a):
    def by(b):
        return a * b
    return by

double = multiply(2)
double(5)  # 10
```

Raro en Python (común en Haskell, Scala) pero útil para crear transformaciones reutilizables.

---

## 5. FP vs OOP — Elegir la herramienta correcta

### 5.1 Usa Programación Funcional cuando...

| Escenario | Por qué FP |
|---|---|
| **Transformación de datos** | Pipeline de funciones puras: clean → normalize → encode |
| **Cálculo de métricas** | Sin estado: `accuracy(y_true, y_pred)` |
| **Ingeniería de características** | Transformar → componer → aplicar |
| **Procesamiento paralelo** | Sin estado compartido → seguro de ejecutar en paralelo |
| **Pipelines ETL** | Datos entran, datos salen, sin efectos secundarios |
| **Lógica de validación** | Las funciones puras son triviales de probar |

### 5.2 Usa Programación Orientada a Objetos cuando...

| Escenario | Por qué OOP |
|---|---|
| **Ciclo de vida del modelo** | El modelo tiene estado (pesos) y comportamiento (fit, predict) |
| **Seguimiento de experimentos** | Acumular métricas a lo largo del tiempo |
| **Interacción con el usuario** | Múltiples operaciones relacionadas sobre datos compartidos |
| **Gestión de recursos** | Conexiones a bases de datos, manejadores de archivos |
| **Sistemas de plugins/extensiones** | Polimorfismo a través de herencia |

### 5.3 El enfoque híbrido (Recomendado)

La mayoría del código real de ciencia de datos usa ambos:

```python
class DataScienceProject:          # OOP: esqueleto, estado, interfaz
    def __init__(self, config):
        self.config = config
        self.model = None

    def _clean(self, df):          # FP: transformación pura
        return df.dropna()

    def _normalize(self, df):      # FP: transformación pura
        return (df - df.mean()) / df.std()

    def _build_features(self, df):  # FP: composición de funciones
        return compose(self._clean, self._normalize)(df)

    def train(self, X, y):
        X = self._build_features(X)
        self.model = RandomForestClassifier(**self.config)
        self.model.fit(X, y)
```

**Regla**: usa OOP para estructurar el programa, FP para procesar los datos.

---

## 6. Paso a paso: Refactorizando de imperativo a funcional

**Antes** (imperativo, difícil de paralelizar, probar o componer):

```python
def process_data(data):
    result = []
    for row in data:
        if row["age"] > 18:
            row["adult"] = True
        else:
            row["adult"] = False
        row["income_log"] = math.log(row["income"] + 1)
        result.append(row)
    return result
```

**Después** (funcional, componible, testeable):

```python
def is_adult(row):                                # pura, testeable
    return {**row, "adult": row["age"] > 18}

def log_income(row):                              # pura, testeable
    return {**row, "income_log": math.log(row["income"] + 1)}

def process_data(data):                           # composición
    return [log_income(is_adult(row)) for row in data]

# Aún mejor: separar responsabilidades
def transform_row(row):
    return log_income(is_adult(row))

process_data = partial(map, transform_row)        # reutilizable, componible
```

---

## 7. Common Mistakes

1. **Mutating data in functions**: modifying the input silently alters data for the caller. Always return a new copy.

2. **Over-using reduce for simple operations**: `sum(list)` is clearer than `reduce(lambda a, b: a + b, list)`.

3. **Using FP where state is needed**: do not force statelessness on inherently stateful operations (e.g., training a neural network).

4. **Ignoring Python's functional tools**: `itertools` (chain, cycle, groupby), `functools` (partial, lru_cache, reduce) are powerful and underused. [[Code Quality]] covers idiomatic usage of these utilities.

5. **Side effects inside map/filter**: if `f(x)` prints something or writes to a file, it is not a true map — it is a disguised loop.

---

## 8. Check Your Understanding

1. A function reads a configuration from a global variable. Is it pure? Why or why not?
2. You have a list of numbers. Write the FP way to keep only positive values and square them.
3. What is the difference between `map` and a list comprehension? Which is more Pythonic?
4. Refactor this to FP: find the average of all even numbers in a list.
5. Why does `pandas.apply()` embody a functional pattern?

---

## 9. Summary

Functional programming is about **data transformations without side effects**. Pure functions, immutability, and higher-order functions (map, filter, reduce) make your code testable, parallelizable, and composable. In data science, use FP for transformations, metrics, and pipelines. Combine it with OOP (which handles state and interfaces) for the best of both paradigms.

---

## 10. Where to Go Next

- [[Object-Oriented Programming]] — The complementary paradigm
- [[Python Fundamentals]] — List comprehensions, generators, lambdas
- [[Python for Data Science]] — pandas, NumPy, and functional patterns
