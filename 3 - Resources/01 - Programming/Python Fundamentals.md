---
tags: [programming, python, foundational]
status: growing
created: 2026-06-27
---

# Fundamentos de Python

## 1. Escenario de aprendizaje

Imagina que acabas de unirte a un equipo de ciencia de datos y te entregan un archivo CSV con 10 millones de registros de clientes. Necesitas limpiar los datos, detectar valores atípicos y calcular métricas de negocio clave, todo antes del final del día. Python no es solo el lenguaje que usarás — es la herramienta que determinará si cumples con el plazo o no. Comprender sus fundamentos (tipos, estructuras de control, funciones, comprensiones de colecciones) marca la diferencia entre escribir código lento y frágil o código eficiente y legible desde el primer intento. [[Python for Data Science]] construye sobre estos fundamentos.

---

## 2. Filosofía principal

El diseño de Python se guía por principios resumidos en "El Zen de Python" (`import this`):

- **La legibilidad cuenta**: el código se escribe una vez pero se lee muchas veces
- **Explícito es mejor que implícito**: haz las intenciones claras
- **Simple es mejor que complejo**: prefiere soluciones directas
- **Debe haber una — y preferiblemente solo una — manera obvia de hacerlo**

Estos principios moldean todo, desde la sintaxis hasta el diseño de bibliotecas. Cuando escribes Python para ciencia de datos, no solo estás escribiendo código — estás comunicando una intención.

### 2.1 Características clave

**Tipado dinámico**: las variables no declaran su tipo. El tipo se infiere y puede cambiar.

```python
x = 5        # x es int
x = "hello"  # ahora x es str — perfectamente válido
```

Esto es conveniente para exploración pero peligroso para producción. Las anotaciones de tipo (ver más abajo) mitigan esto.

**Todo es un objeto**: las funciones son objetos, las clases son objetos, incluso los tipos son objetos. Esto significa que puedes pasar funciones como argumentos, retornarlas desde otras funciones y almacenarlas en estructuras de datos — esencial para programación funcional y decoradores.

**La indentación define el ámbito**: los bloques se delimitan con indentación (4 espacios, no tabulaciones). Esto impone código legible.

---

## 3. Tipos incorporados

### 3.1 Tipos numéricos

```python
int:    age = 25
float:  pi = 3.14159
complex: c = 3 + 4j    # usado en procesamiento de señales, raramente en DS
```

Los enteros en Python tienen precisión arbitraria — nunca se desbordan. Esto es tanto una bendición (nunca preocuparse por desbordamiento) como una maldición (enteros grandes son lentos).

### 3.2 Tipos secuencia

| Tipo | ¿Mutable? | ¿Ordenado? | ¿Duplicados? | Caso de uso |
|---|---|---|---|---|
| `list` | Sí | Sí | Sí | Colección de propósito general |
| `tuple` | No | Sí | Sí | Datos fijos, retornos de funciones |
| `range` | No | Sí | N/A | Iterar sin almacenar |

**Las listas** son el caballo de batalla de la ciencia de datos en Python. Almacenan cualquier tipo, tienen tamaño dinámico y soportan inserciones eficientes al final. Sin embargo, para datos numéricos, `numpy.ndarray` es más rápido y eficiente en memoria. [[Data Structures & Algorithms]] explora las diferencias de rendimiento entre estos contenedores.

**Las tuplas** son listas inmutables. Úsalas cuando los datos no deban cambiar (ej. coordenadas, retornos de funciones).

### 3.3 Tipos de mapeo y conjuntos

**dict**: pares clave-valor. Búsqueda, inserción y eliminación en O(1) promedio.

```python
person = {"name": "Alice", "age": 30, "city": "NYC"}
person.get("salary", 0)        # acceso seguro con valor por defecto
person.setdefault("role", "DS") # asigna solo si no existe
```

**set**: elementos únicos sin orden. Prueba de pertenencia en O(1).

```python
unique_ids = {1, 2, 3, 3, 3}  # {1, 2, 3}
{1, 2} | {2, 3}                # unión: {1, 2, 3}
{1, 2} & {2, 3}                # intersección: {2}
```

### 3.4 None

`None` es el valor nulo de Python. No es 0, no es False, no es una cadena vacía — es la ausencia de un valor.

```python
result = model.predict(X)  # podría retornar None si algo falla
if result is None:         # siempre usa 'is', no '=='
    handle_error()
```

---

## 4. Flujo de control

### 4.1 Condicionales

```python
if score > 90:
    grade = "A"
elif score > 80:
    grade = "B"
else:
    grade = "C"
```

**Valores falsy**: los siguientes evalúan a `False` en contexto booleano:
- `None`, `False`, `0`, `0.0`, `""`, `[]`, `{}`, `set()`
- Todo lo demás es `True`

Esto es útil para verificaciones concisas:
```python
if not data:        # verifica si data es None, lista vacía, dict vacío, etc.
    data = default_data()
```

### 4.2 Bucles

```python
# Bucle for — itera sobre cualquier iterable
for item in collection:
    process(item)

# enumerate cuando necesitas el índice
for i, item in enumerate(collection):
    print(f"{i}: {item}")

# zip para iterar múltiples colecciones en paralelo
for x, y in zip(X, y):
    print(f"Feature: {x}, Label: {y}")

# List comprehension — la forma pitónica de construir listas
squares = [x**2 for x in range(10) if x % 2 == 0]
# Equivalente a:
squares = []
for x in range(10):
    if x % 2 == 0:
        squares.append(x**2)
```

La comprensión es **más rápida y más legible**. Como regla: si estás construyendo una lista agregando elementos en un bucle, usa una comprensión.

---

## 5. Funciones

### 5.1 Definición de funciones

```python
def preprocess(data: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
    """Clean and preprocess a DataFrame.

    Args:
        data: Raw input DataFrame
        drop_na: Whether to drop missing values

    Returns:
        Cleaned DataFrame
    """
    if drop_na:
        data = data.dropna()
    return data
```

- **Las anotaciones de tipo** (`data: pd.DataFrame`, `-> pd.DataFrame`) no se aplican en tiempo de ejecución pero ayudan a la legibilidad y permiten análisis estático con mypy. [[Code Quality]] cubre anotaciones de tipo, linting y herramientas de análisis estático.
- **Los docstrings** documentan qué hace la función, sus parámetros y su valor de retorno.

### 5.2 *args y **kwargs

```python
def train_model(X, y, *args, **kwargs):
    """Train a model with flexible parameters."""
    model = RandomForestClassifier(*args, **kwargs)
    model.fit(X, y)
    return model

train_model(X, y, n_estimators=100, max_depth=5)
```

- `*args` captura argumentos posicionales adicionales como una tupla
- `**kwargs` captura argumentos de palabra clave adicionales como un diccionario

### 5.3 Funciones Lambda

Funciones anónimas pequeñas para operaciones simples:

```python
lambda x: x**2
lambda x, y: x + y
```

Se usan principalmente con `map`, `filter`, `sort` y `apply` de pandas.

```python
df["normalized"] = df["value"].apply(lambda x: (x - min_val) / (max_val - min_val))
```

---

## 6. Comprensiones y generadores

### 6.1 Comprensiones

Python tiene comprensiones para listas, diccionarios y conjuntos:

```python
# List comprehension
[x**2 for x in range(10)]

# Dict comprehension
{x: x**2 for x in range(5)}    # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# Set comprehension
{x**2 for x in [1, 1, 2, 2, 3]}  # {1, 4, 9}

# Comprensión anidada (aplanar una matriz)
[x for row in matrix for x in row]
```

### 6.2 Generadores

Los generadores producen valores **de forma perezosa** — uno a la vez, bajo demanda — sin almacenar toda la secuencia en memoria.

```python
# Expresión generadora (usa (), no [])
squares = (x**2 for x in range(10_000_000))  # instantáneo, ~0 memoria
list(squares)[:5]  # [0, 1, 4, 9, 16]

# Función generadora con yield
def read_large_file(path):
    with open(path) as f:
        for line in f:
            yield process(line)  # yield, no return

for processed in read_large_file("huge_data.csv"):
    analyze(processed)
```

**Cuándo usar generadores**: cuando procesas datos demasiado grandes para caber en memoria (streaming, archivos grandes, secuencias infinitas).

---

## 7. Administradores de contexto

Los administradores de contexto manejan la configuración y el cierre automáticamente:

```python
with open("file.txt") as f:      # cierra el archivo automáticamente
    content = f.read()

with pd.HDFStore("data.h5") as store:  # cierra el store automáticamente
    df = store["mydata"]
```

Puedes crear los tuyos con `contextlib.contextmanager`:

```python
from contextlib import contextmanager

@contextmanager
def timer(name: str):
    import time
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"{name}: {elapsed:.2f}s")

with timer("training"):
    model.fit(X, y)
```

Los administradores de contexto se usan ampliamente en scripts de [[CLI & Productivity]] para manejar archivos, subprocesos y otros recursos.

---

## 8. Errores Comunes

1. **Argumentos mutables por defecto**: los valores por defecto se evalúan una vez en la definición de la función, no en cada llamada.

```python
def add_item(item, lst=[]):  # MAL: lst se comparte entre llamadas
    lst.append(item)
    return lst

add_item(1)  # [1]
add_item(2)  # [1, 2] — ¡no [2]!
```

**Solución**: usar `None` y crear una nueva lista cada vez.

2. **Modificar una lista mientras se itera**: se saltan elementos o se causan bucles infinitos. Iterar sobre una copia en su lugar.

```python
for item in list[:]:    # iterar sobre una copia del slice
    if condition(item):
        list.remove(item)
```

3. **Usar `==` en lugar de `is` para None**: `None` es un singleton. Usar `is None` (verificación de identidad), no `== None` (verificación de igualdad).

4. **No usar `with` para operaciones de archivos**: los archivos dejados abiertos pueden causar fugas de recursos. Siempre usar administradores de contexto.

---

## 9. Verifica tu Comprensión

1. ¿Cuál es la diferencia entre una lista y una tupla? ¿Cuándo usarías cada una?
2. ¿Por qué `[0] * 5` produce `[0, 0, 0, 0, 0]` pero `[[]] * 5` produce una lista de 5 referencias a la MISMA lista vacía?
3. ¿Cuál es la salida de `[x**2 for x in range(5) if x % 2 == 1]`?
4. Un generador produce valores. ¿Cómo difiere esto de retornar una lista? ¿Cuáles son las implicaciones de memoria?
5. ¿Qué verifica `is` que `==` no verifica?

---

## 10. Resumen

La filosofía de diseño de Python — legibilidad, explicititud y simplicidad — lo convierte en el lenguaje ideal para ciencia de datos. Sus tipos incorporados (list, dict, set) cubren la mayoría de las necesidades, las comprensiones hacen el código conciso, los generadores manejan datos grandes, y los administradores de contexto aseguran una gestión limpia de recursos. Domina estos fundamentos antes de pasar a las bibliotecas especializadas.

---

## 11. Dónde Ir Ahora

- [[Python para Ciencia de Datos]] — NumPy, pandas, manipulación práctica de datos
- [[Programación Orientada a Objetos]] — Clases, herencia, patrones de diseño
- [[Programación Funcional]] — Lambdas, map, filter, reduce, inmutabilidad
