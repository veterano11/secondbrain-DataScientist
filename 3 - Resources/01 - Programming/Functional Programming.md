---
tags: [programming, functional, core]
status: growing
created: 2026-06-27
---

# Functional Programming

## 1. Why This Matters

Functional programming (FP) is not just an academic curiosity — it is the **natural paradigm for data processing**. Every pandas operation (`map`, `apply`, `groupby`), every Spark transformation, and every NumPy vectorized operation follows functional principles.

If OOP is about organizing code around **objects** (state + behavior), FP is about organizing code around **data transformations** (input → output, without side effects). Understanding FP makes you write more predictable, testable, and parallelizable data code.

---

## 2. Core Concepts

### 2.1 Pure Functions

A pure function has two properties:
1. **Same input → same output** (no hidden state)
2. **No side effects** (does not modify external state)

```python
# Pure
def add(a, b):
    return a + b

# Impure — depends on external state
total = 0
def add_to_total(x):
    global total
    total += x
    return total
```

**Why pure functions matter in data science:**
- **Testable**: no setup needed, just call with inputs and check outputs
- **Cachable**: same input always gives same output → `@functools.lru_cache`
- **Parallelizable**: no shared state, safe to run in any order
- **Composable**: combine small functions into larger ones — [[Testing for Data Science]] leverages these properties for reliable tests

### 2.2 Immutability

Data should not be modified — instead, create new data with the changes applied.

```python
# Mutable (OOP style)
df["new_col"] = df["a"] + df["b"]   # modifies df in place

# Immutable (FP style)
new_df = df.assign(new_col=df["a"] + df["b"])  # returns new DataFrame
```

**Why immutability matters:**
- Prevents accidental mutation (a common source of bugs)
- Makes code easier to reason about (data only flows forward)
- Enables change detection (compare old vs new)

In pandas: `df.assign()` (immutable) vs `df["col"] = ...` (mutable). In PyTorch: `tensor.clone()` and `torch.no_grad()` enforce separation between computation and modification.

### 2.3 First-Class and Higher-Order Functions

**First-class functions**: functions are values — they can be assigned to variables, passed as arguments, and returned from other functions.

```python
def square(x): return x ** 2
def cube(x): return x ** 3

operations = [square, cube]  # functions in a list
for op in operations:
    print(op(3))             # 9, 27
```

**Higher-order functions**: functions that take other functions as arguments or return functions.

```python
def apply_transformation(func, data):
    return [func(x) for x in data]

apply_transformation(square, [1, 2, 3])  # [1, 4, 9]
```

This is the foundation of `map`, `filter`, `reduce`, and pandas `apply`.

---

## 3. The Three Pillars: Map, Filter, Reduce

These three functions form the core of data processing in FP.

### 3.1 map — Transform Each Element

**Pattern**: take a collection, apply a function to each element, return a new collection of the same size.

```python
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x ** 2, numbers))    # FP style
squared = [x ** 2 for x in numbers]                # Pythonic (preferred)

# pandas equivalent
df["normalized"] = df["value"].map(lambda x: (x - min_) / (max_ - min_))
```

**Pythonic note**: list comprehensions are generally preferred over `map()` in Python because they are more readable. Use `map()` when you already have a named function.

### 3.2 filter — Keep Matching Elements

**Pattern**: take a collection, keep elements that satisfy a predicate, return a (potentially smaller) collection.

```python
numbers = [1, 2, 3, 4, 5, 6]
evens = list(filter(lambda x: x % 2 == 0, numbers))  # [2, 4, 6]
evens = [x for x in numbers if x % 2 == 0]           # Pythonic (preferred)
```

### 3.3 reduce — Accumulate

**Pattern**: take a collection, combine elements sequentially into a single value.

```python
from functools import reduce

numbers = [1, 2, 3, 4]
product = reduce(lambda a, b: a * b, numbers)  # 1 * 2 * 3 * 4 = 24

# Real-world use: chaining transformations
from functools import reduce

def pipeline(data, *functions):
    return reduce(lambda x, f: f(x), functions, data)

result = pipeline(raw_data, clean, normalize, encode)
# Equivalent to: encode(normalize(clean(raw_data)))
```

`reduce` is less common in Python (Clojure, Haskell use it heavily) but invaluable for composing pipelines.

---

## 4. Function Composition in Data Science

### 4.1 The Pipeline Pattern

FP pipelines chain transformations:

```python
# Without composition (reads bottom-up or inside-out)
predictions = model.predict(encode(normalize(clean(raw_data))))

# With composition (reads top-to-bottom)
def compose(*functions):
    def composed(data):
        for f in functions:
            data = f(data)
        return data
    return composed

pipeline = compose(clean, normalize, encode, model.predict)
predictions = pipeline(raw_data)
```

This composable pattern is central to [[Python for Data Science]]'s scikit-learn `Pipeline` API.

### 4.2 Partial Application

Fix some arguments of a function, creating a new function with fewer arguments:

```python
from functools import partial

def scale(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)

scale_age = partial(scale, min_val=0, max_val=100)
scale_income = partial(scale, min_val=20000, max_val=200000)

ages_normalized = list(map(scale_age, ages))
```

### 4.3 Currying

Convert a function that takes multiple arguments into a chain of functions that each take one argument:

```python
def multiply(a):
    def by(b):
        return a * b
    return by

double = multiply(2)
double(5)  # 10
```

Rare in Python (common in Haskell, Scala) but useful for creating reusable transformations.

---

## 5. FP vs OOP — Choosing the Right Tool

### 5.1 Use Functional Programming When...

| Scenario | Why FP |
|---|---|
| **Data transformation** | Pipeline of pure functions: clean → normalize → encode |
| **Metric computation** | Stateless: `accuracy(y_true, y_pred)` |
| **Feature engineering** | Transform → compose → apply |
| **Parallel processing** | No shared state → safe to run in parallel |
| **ETL pipelines** | Data in, data out, no side effects |
| **Validation logic** | Pure functions are trivial to test |

### 5.2 Use Object-Oriented Programming When...

| Scenario | Why OOP |
|---|---|
| **Model lifecycle** | Model has state (weights) and behavior (fit, predict) |
| **Experiment tracking** | Accumulate metrics over time |
| **User interaction** | Multiple related operations on shared data |
| **Resource management** | Database connections, file handles |
| **Plugin/extension systems** | Polymorphism through inheritance |

### 5.3 The Hybrid Approach (Recommended)

Most real data science code uses both:

```python
class DataScienceProject:          # OOP: skeleton, state, interface
    def __init__(self, config):
        self.config = config
        self.model = None

    def _clean(self, df):          # FP: pure transformation
        return df.dropna()

    def _normalize(self, df):      # FP: pure transformation
        return (df - df.mean()) / df.std()

    def _build_features(self, df):  # FP: function composition
        return compose(self._clean, self._normalize)(df)

    def train(self, X, y):
        X = self._build_features(X)
        self.model = RandomForestClassifier(**self.config)
        self.model.fit(X, y)
```

**Rule**: use OOP to structure the program, FP to process the data.

---

## 6. Step-by-Step: Refactoring from Imperative to Functional

**Before** (imperative, hard to parallelize, test, or compose):

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

**After** (functional, composable, testable):

```python
def is_adult(row):                                # pure, testable
    return {**row, "adult": row["age"] > 18}

def log_income(row):                              # pure, testable
    return {**row, "income_log": math.log(row["income"] + 1)}

def process_data(data):                           # composition
    return [log_income(is_adult(row)) for row in data]

# Even better: separate concerns
def transform_row(row):
    return log_income(is_adult(row))

process_data = partial(map, transform_row)        # reusable, composable
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
