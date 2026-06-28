---
tags: [programming, oop, core]
status: growing
created: 2026-06-27
---

# Object-Oriented Programming

## 1. Why This Matters

Object-oriented programming (OOP) is the dominant paradigm in software engineering, and for good reason: it models the world as **objects that contain both data and behavior**. In data science, you use OOP every time you interact with a model, a DataFrame, or a pipeline — they are all objects.

Understanding OOP lets you:
- Build reusable ML pipelines and custom models
- Write code that is organized, testable, and maintainable — see [[Code Quality]] for clean code patterns
- Understand how libraries like scikit-learn and PyTorch are designed — [[Python for Data Science]] explores their OOP design
- Choose when OOP is the right tool (and when it is not)

---

## 2. The Four Pillars of OOP

### 2.1 Encapsulation

**Data and methods that operate on that data are bundled together** in a class. Internal details are hidden from the outside.

```python
class DataPipeline:
    def __init__(self, source: str):
        self.source = source
        self._data = None        # "protected" by convention
        self.__status = "init"   # "private" (name-mangled)

    def load(self) -> pd.DataFrame:
        """Public interface — hides how loading works."""
        self._data = pd.read_csv(self.source)
        self.__status = "loaded"
        return self._data
```

**Python's access control** (by convention):
- `attr`: public (use freely)
- `_attr`: protected (internal use, but accessible)
- `__attr`: private (name-mangled to `_ClassName__attr`, harder to access)

Unlike Java or C++, Python does not enforce access control — it trusts developers to respect the conventions.

### 2.2 Inheritance

**A class can inherit attributes and methods from another class**, creating an "is-a" relationship.

```python
class BaseModel:
    def __init__(self, params: dict):
        self.params = params

    def train(self, X, y):
        raise NotImplementedError  # subclasses must implement

    def predict(self, X):
        raise NotImplementedError

class RandomForestModel(BaseModel):    # RandomForestModel IS-A BaseModel
    def train(self, X, y):             # override
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
```

**When to use**: when you have a clear hierarchy with shared behavior and specialized subclasses. The base class defines the interface; subclasses implement the details.

### 2.3 Polymorphism

**The same interface can work with different types.** The caller does not need to know the specific type.

```python
models = [RandomForestModel({}), XGBoostModel({"n_estimators": 100})]
for model in models:
    model.train(X, y)       # each model.train() does different things
    predictions = model.predict(X_test)  # but the interface is the same
```

**Duck typing**: "If it walks like a duck and quacks like a duck, it is a duck." Python does not require formal interface inheritance — any object with a `.train()` method can be used where a "trainable" is expected.

### 2.4 Abstraction

**Complexity is hidden behind a simple interface.** The user of a class does not need to know how it works internally.

```python
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=10)),
    ("classifier", RandomForestClassifier()),
])
pipeline.fit(X_train, y_train)  # I do not need to know how each step works
```

---

## 3. Python OOP in Depth

### 3.1 Magic Methods

Magic methods (dunder methods) define how objects behave with Python syntax.

```python
class Dataset:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def __len__(self):
        return len(self.data)          # len(dataset)

    def __getitem__(self, idx):
        return self.data.iloc[idx]     # dataset[5]

    def __iter__(self):
        return self.data.iterrows()    # for idx, row in dataset:

    def __repr__(self):
        return f"Dataset({len(self)} rows)"  # print(dataset)

    def __add__(self, other):
        return Dataset(pd.concat([self.data, other.data]))  # dataset1 + dataset2
```

### 3.2 Properties

Properties allow controlled access to attributes with getter/setter logic:

```python
class ModelWrapper:
    def __init__(self, model):
        self._model = model
        self._is_trained = False

    @property
    def is_trained(self):
        return self._is_trained

    @property
    def feature_importances(self):
        if not self._is_trained:
            raise ValueError("Model not trained yet")
        return self._model.feature_importances_

    @is_trained.setter
    def is_trained(self, value):
        if not isinstance(value, bool):
            raise TypeError("Must be boolean")
        self._is_trained = value

model = ModelWrapper(rf)
print(model.is_trained)   # reads like an attribute, but uses the getter
```

### 3.3 Class Methods and Static Methods

```python
class DataLoader:
    format = "csv"               # class attribute — shared by all instances

    @classmethod
    def from_csv(cls, path):     # factory — returns an instance
        instance = cls()
        instance.data = pd.read_csv(path)
        return instance

    @classmethod
    def from_parquet(cls, path): # another factory
        instance = cls()
        instance.data = pd.read_parquet(path)
        return instance

    @staticmethod
    def validate(df: pd.DataFrame) -> bool:  # utility — no self, no cls
        return not df.empty and df.columns.duplicated().sum() == 0

loader = DataLoader.from_csv("data.csv")
```

- **@classmethod**: receives the class (not the instance). Used for factory methods.
- **@staticmethod**: receives nothing special. Used for utility functions related to the class.

### 3.4 Composition Over Inheritance

**Favor "has-a" over "is-a".** Instead of inheriting, build objects that contain other objects.

```python
# Inheritance approach (less flexible)
class PreprocessingPipeline(Pipeline):
    pass

# Composition approach (more flexible)
class PreprocessingPipeline:
    def __init__(self):
        self.scaler = StandardScaler()      # has-a scaler
        self.encoder = OneHotEncoder()      # has-an encoder
        self.selector = SelectKBest()       # has-a selector

    def fit_transform(self, X, y=None):
        X = self.scaler.fit_transform(X)
        X = self.encoder.fit_transform(X)
        X = self.selector.fit_transform(X, y)
        return X
```

Composition is more flexible: you can swap components, add steps, and test each piece independently. This principle is widely applied in [[Data Structures & Algorithms]], where complex structures are built from simpler ones.

---

## 4. OOP vs Functional Programming — When to Use Which

This is one of the most important distinctions to understand as a data scientist. Both paradigms are tools; choosing the right one depends on what you are building.

### 4.1 Use OOP When...

**1. You have state that changes over time**

```python
class ExperimentTracker:
    def __init__(self, name: str):
        self.name = name
        self.metrics = []
        self.start_time = None

    def begin(self):
        self.start_time = time.time()

    def log_metric(self, name: str, value: float):
        self.metrics.append({"name": name, "value": value, "time": time.time()})

    def summary(self) -> dict:
        return {
            "experiment": self.name,
            "duration": time.time() - self.start_time,
            "metrics": self.metrics,
        }
```

The tracker has **state** (metrics list, start time) that accumulates across method calls. This is natural in OOP, awkward in FP.

**2. You have multiple objects that share behavior but differ in specifics**

Inheritance and polymorphism shine here. A `RandomForestModel` and `XGBoostModel` both train and predict, but do so differently. The calling code does not care.

**3. You want to model real-world entities**

User, Dataset, Model, Experiment, Pipeline — these map naturally to objects with data and behavior.

**4. You need to enforce an interface**

Base classes with abstract methods ensure all subclasses implement the required methods. This prevents runtime surprises.

### 4.2 Use Functional Programming When...

**1. You are transforming data through a pipeline**

```python
def clean(df): return df.dropna()
def normalize(df): return (df - df.mean()) / df.std()
def encode(df): return pd.get_dummies(df)

pipeline = compose(encode, normalize, clean)  # clean → normalize → encode
result = pipeline(raw_data)
```

Each function takes data, returns transformed data, and has **no side effects**. This is the natural paradigm for data preprocessing.

**2. Operations are stateless and independent**

```python
def compute_accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()

def compute_precision(y_true, y_pred):
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0
```

These functions depend only on their inputs. They are easy to test (no setup needed), easy to parallelize (no shared state), and easy to reason about.

**3. You want to parallelize or distribute computation**

Stateless functions can be safely mapped across partitions, clusters, or GPUs. Spark, Dask, and Ray all leverage functional patterns.

```python
results = parallel_map(compute_metric, [(y1, p1), (y2, p2), ...])
```

**4. You are composing small, reusable operations**

Small functions like `scale`, `clip`, `log_transform` can be composed in any order. This is more natural with FP than OOP.

### 4.3 Mixing Both — The Data Science Sweet Spot

In practice, the best data science code uses **both**:

```python
class ModelPipeline:                          # OOP — manages state
    def __init__(self, preprocessors: list, model):
        self.transformers = preprocessors     # FP — stateless functions
        self.model = model

    def _apply_transformers(self, X):         # FP-style function composition
        for fn in self.transformers:
            X = fn(X)                         # each fn is pure
        return X

    def fit(self, X, y):
        X = self._apply_transformers(X)       # FP: data flows through
        self.model.fit(X, y)                  # OOP: model has state

    def predict(self, X):
        X = self._apply_transformers(X)
        return self.model.predict(X)
```

**Rule of thumb**:
- **OOP for the skeleton**: classes that hold state, manage resources, provide interfaces
- **FP for the internals**: stateless transformations, metric computations, data processing

scikit-learn itself follows this: `fit()` / `predict()` methods (OOP interface) with `transform()` methods that are often stateless (FP).

---

## 5. Common Mistakes

1. **Over-engineering**: not everything needs a class. If a function suffices, use a function. Premature abstraction adds complexity without value.

2. **Deep inheritance hierarchies**: "diamond problem," fragile base classes, hard to debug. Prefer composition over inheritance past 2 levels.

3. **Mutable shared state**: class variables modified by instances cause subtle bugs.

```python
class Bad:
    shared = []          # BAD: all instances share this list

a = Bad()
b = Bad()
a.shared.append(1)       # also affects b.shared!
```

4. **Using OOP when FP is simpler**: data transformations, metric calculations, and ETL pipelines rarely need classes.

---

## 6. Check Your Understanding

1. You are building a feature engineering library. Some transformations are stateless (log transform) and some require fitting (standard scaler). Would you use OOP, FP, or both? Why?
2. What is the difference between composition and inheritance? Give an example of when each is appropriate.
3. Why does Python not enforce private attributes? Is this a weakness?
4. You have a `@staticmethod` and a plain function outside the class — what is the difference? When would you use the static method?
5. A coworker defines every function inside a class, even pure functions like `add(a, b)`. What would you tell them?

---

## 7. Summary

OOP organizes code around objects with data and behavior. Its four pillars — encapsulation, inheritance, polymorphism, abstraction — make it ideal for complex, stateful systems. In data science, use OOP for models, pipelines, experiments, and anything that manages state. Use FP for transformations, metrics, and pure computations. The best code uses both, choosing the right tool for each job.

---

## 8. Where to Go Next

- [[Functional Programming]] — The complement to OOP
- [[Python Fundamentals]] — Basic building blocks
- [[Testing for Data Science]] — OOP structures make testing easier
- [[Python for Data Science]] — OOP in scikit-learn, PyTorch, and pandas
- [[Data Structures & Algorithms]] — OOP design patterns in practice
