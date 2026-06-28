---
tags: [programming, python, foundational]
status: growing
created: 2026-06-27
---

# Python Fundamentals

## 1. Why This Matters

Python is the language of data science not because it is the fastest (it is not) or the most elegant (debatable), but because it is the **most readable and the most connected**. Every ML library, every deep learning framework, every data tool speaks Python first. Understanding Python deeply means understanding how these tools work under the hood — [[Python for Data Science]] builds on these fundamentals.

---

## 2. Core Philosophy

Python's design is guided by principles summarized in "The Zen of Python" (`import this`):

- **Readability counts**: code is written once but read many times
- **Explicit is better than implicit**: make intentions clear
- **Simple is better than complex**: prefer straightforward solutions
- **There should be one — and preferably only one — obvious way to do it**

These principles shape everything from syntax to library design. When you write Python for data science, you are not just writing code — you are communicating intent.

### 2.1 Key Characteristics

**Dynamic typing**: variables do not declare their type. The type is inferred and can change.

```python
x = 5        # x is int
x = "hello"  # now x is str — perfectly valid
```

This is convenient for exploration but dangerous for production. Type hints (see below) mitigate this.

**Everything is an object**: functions are objects, classes are objects, even types are objects. This means you can pass functions as arguments, return them from other functions, and store them in data structures — essential for functional programming and decorators.

**Indentation defines scope**: blocks are delimited by indentation (4 spaces, not tabs). This enforces readable code.

---

## 3. Built-in Types

### 3.1 Numeric Types

```python
int:    age = 25
float:  pi = 3.14159
complex: c = 3 + 4j    # used in signal processing, rarely in DS
```

Python integers are arbitrary precision — they do not overflow. This is both a blessing (never worry about overflow) and a curse (big integers are slow).

### 3.2 Sequence Types

| Type | Mutable? | Ordered? | Duplicates? | Use Case |
|---|---|---|---|---|
| `list` | Yes | Yes | Yes | General purpose collection |
| `tuple` | No | Yes | Yes | Fixed data, function returns |
| `range` | No | Yes | N/A | Looping without storing |

**Lists** are the workhorse of Python data science. They store any type, are dynamically sized, and support efficient appends. However, for numerical data, `numpy.ndarray` is faster and more memory-efficient. [[Data Structures & Algorithms]] explores the performance tradeoffs between these containers.

**Tuples** are immutable lists. Use them when the data should not change (e.g., coordinates, function returns).

### 3.3 Mapping and Set Types

**dict**: key-value pairs. O(1) average lookup, insert, delete.

```python
person = {"name": "Alice", "age": 30, "city": "NYC"}
person.get("salary", 0)        # safe access with default
person.setdefault("role", "DS") # set only if missing
```

**set**: unordered unique elements. O(1) membership test.

```python
unique_ids = {1, 2, 3, 3, 3}  # {1, 2, 3}
{1, 2} | {2, 3}                # union: {1, 2, 3}
{1, 2} & {2, 3}                # intersection: {2}
```

### 3.4 None

`None` is Python's null value. It is not 0, not False, not an empty string — it is the absence of a value.

```python
result = model.predict(X)  # might return None if something fails
if result is None:         # always use 'is', not '=='
    handle_error()
```

---

## 4. Control Flow

### 4.1 Conditionals

```python
if score > 90:
    grade = "A"
elif score > 80:
    grade = "B"
else:
    grade = "C"
```

**Truthiness**: the following evaluate to `False` in boolean context:
- `None`, `False`, `0`, `0.0`, `""`, `[]`, `{}`, `set()`
- Everything else is `True`

This is useful for concise checks:
```python
if not data:        # checks if data is None, empty list, empty dict, etc.
    data = default_data()
```

### 4.2 Loops

```python
# For loop — iterate over any iterable
for item in collection:
    process(item)

# enumerate when you need index
for i, item in enumerate(collection):
    print(f"{i}: {item}")

# zip to iterate multiple collections in parallel
for x, y in zip(X, y):
    print(f"Feature: {x}, Label: {y}")

# List comprehension — the Pythonic way to build lists
squares = [x**2 for x in range(10) if x % 2 == 0]
# Equivalent to:
squares = []
for x in range(10):
    if x % 2 == 0:
        squares.append(x**2)
```

The comprehension is **faster and more readable**. As a rule: if you are building a list by appending in a loop, use a comprehension instead.

---

## 5. Functions

### 5.1 Defining Functions

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

- **Type hints** (`data: pd.DataFrame`, `-> pd.DataFrame`) are not enforced at runtime but help readability and enable static analysis with mypy. [[Code Quality]] covers type hints, linting, and static analysis tools.
- **Docstrings** document what the function does, its parameters, and its return value.

### 5.2 *args and **kwargs

```python
def train_model(X, y, *args, **kwargs):
    """Train a model with flexible parameters."""
    model = RandomForestClassifier(*args, **kwargs)
    model.fit(X, y)
    return model

train_model(X, y, n_estimators=100, max_depth=5)
```

- `*args` captures extra positional arguments as a tuple
- `**kwargs` captures extra keyword arguments as a dict

### 5.3 Lambda Functions

Small anonymous functions for simple operations:

```python
lambda x: x**2
lambda x, y: x + y
```

Primarily used with `map`, `filter`, `sort`, and pandas `apply`.

```python
df["normalized"] = df["value"].apply(lambda x: (x - min_val) / (max_val - min_val))
```

---

## 6. Comprehensions and Generators

### 6.1 Comprehensions

Python has comprehensions for list, dict, and set:

```python
# List comprehension
[x**2 for x in range(10)]

# Dict comprehension
{x: x**2 for x in range(5)}    # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# Set comprehension
{x**2 for x in [1, 1, 2, 2, 3]}  # {1, 4, 9}

# Nested comprehension (flatten a matrix)
[x for row in matrix for x in row]
```

### 6.2 Generators

Generators produce values **lazily** — one at a time, on demand — without storing the entire sequence in memory.

```python
# Generator expression (uses (), not [])
squares = (x**2 for x in range(10_000_000))  # instant, ~0 memory
list(squares)[:5]  # [0, 1, 4, 9, 16]

# Generator function with yield
def read_large_file(path):
    with open(path) as f:
        for line in f:
            yield process(line)  # yield, not return

for processed in read_large_file("huge_data.csv"):
    analyze(processed)
```

**When to use generators**: when processing data too large to fit in memory (streaming, large files, infinite sequences).

---

## 7. Context Managers

Context managers handle setup and teardown automatically:

```python
with open("file.txt") as f:      # closes file automatically
    content = f.read()

with pd.HDFStore("data.h5") as store:  # closes store automatically
    df = store["mydata"]
```

You can create your own with `contextlib.contextmanager`:

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

Context managers are widely used in [[CLI & Productivity]] scripts to manage files, subprocesses, and other resources.

---

## 8. Common Mistakes

1. **Mutable default arguments**: defaults are evaluated once at function definition, not each call.

```python
def add_item(item, lst=[]):  # BAD: lst is shared across calls
    lst.append(item)
    return lst

add_item(1)  # [1]
add_item(2)  # [1, 2] — not [2]!
```

**Fix**: use `None` and create a new list each time.

2. **Modifying a list while iterating**: skip elements or cause infinite loops. Iterate over a copy instead.

```python
for item in list[:]:    # iterate over a slice copy
    if condition(item):
        list.remove(item)
```

3. **Using `==` instead of `is` for None**: `None` is a singleton. Use `is None` (identity check), not `== None` (equality check).

4. **Not using `with` for file operations**: files left open can cause resource leaks. Always use context managers.

---

## 9. Check Your Understanding

1. What is the difference between a list and a tuple? When would you use each?
2. Why does `[0] * 5` produce `[0, 0, 0, 0, 0]` but `[[]] * 5` produces a list of 5 references to the SAME empty list?
3. What is the output of `[x**2 for x in range(5) if x % 2 == 1]`?
4. A generator yields values. How is this different from returning a list? What are the memory implications?
5. What does `is` check that `==` does not?

---

## 10. Summary

Python's design philosophy — readability, explicitness, and simplicity — makes it the ideal language for data science. Its built-in types (list, dict, set) cover most needs, comprehensions make code concise, generators handle large data, and context managers ensure clean resource management. Master these fundamentals before moving to the specialized libraries.

---

## 11. Where to Go Next

- [[Python for Data Science]] — NumPy, pandas, practical data manipulation
- [[Object-Oriented Programming]] — Classes, inheritance, design patterns
- [[Functional Programming]] — Lambdas, map, filter, reduce, immutability
