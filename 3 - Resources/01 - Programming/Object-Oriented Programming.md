---
tags: [programming, oop, core]
status: growing
created: 2026-06-27
---

# Programación Orientada a Objetos

## 1. Escenario de aprendizaje

Tu equipo de ciencia de datos mantiene una docena de pipelines de ML diferentes, cada uno con preprocesamiento, entrenamiento y evaluación. Cada pipeline es similar pero con pequeños cambios. Sin una estructura común, el código se duplica y cada nuevos proyecto requiere empezar desde cero. La programación orientada a objetos (OOP) resuelve esto: defines una clase `Pipeline` base con métodos `fit()` y `predict()`, y cada caso concreto simplemente hereda y personaliza lo necesario. De repente, un nuevo proyecto es solo 20 líneas en lugar de 200.

En ciencia de datos, usas OOP cada vez que interactúas con un modelo, un DataFrame o un pipeline — todos son objetos. Entender OOP te permite:
- Construir pipelines de ML reutilizables y modelos personalizados
- Escribir código organizado, testeable y mantenible — consulta [[Code Quality]] para patrones de código limpio
- Entender cómo están diseñadas bibliotecas como scikit-learn y PyTorch — [[Python for Data Science]] explora su diseño OOP
- Elegir cuándo OOP es la herramienta correcta (y cuándo no)

---

## 2. Los cuatro pilares de OOP

### 2.1 Encapsulación

**Los datos y los métodos que operan sobre esos datos se agrupan juntos** en una clase. Los detalles internos están ocultos del exterior.

```python
class DataPipeline:
    def __init__(self, source: str):
        self.source = source
        self._data = None        # "protegido" por convención
        self.__status = "init"   # "privado" (name-mangled)

    def load(self) -> pd.DataFrame:
        """Public interface — hides how loading works."""
        self._data = pd.read_csv(self.source)
        self.__status = "loaded"
        return self._data
```

**Control de acceso en Python** (por convención):
- `attr`: público (úsalo libremente)
- `_attr`: protegido (uso interno, pero accesible)
- `__attr`: privado (name-mangled a `_ClassName__attr`, más difícil de acceder)

A diferencia de Java o C++, Python no impone el control de acceso — confía en que los desarrolladores respeten las convenciones.

### 2.2 Herencia

**Una clase puede heredar atributos y métodos de otra clase**, creando una relación "es-un".

```python
class BaseModel:
    def __init__(self, params: dict):
        self.params = params

    def train(self, X, y):
        raise NotImplementedError  # las subclases deben implementar

    def predict(self, X):
        raise NotImplementedError

class RandomForestModel(BaseModel):    # RandomForestModel ES-UN BaseModel
    def train(self, X, y):             # sobreescribe
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
```

**Cuándo usarla**: cuando tienes una jerarquía clara con comportamiento compartido y subclases especializadas. La clase base define la interfaz; las subclases implementan los detalles.

### 2.3 Polimorfismo

**La misma interfaz puede funcionar con diferentes tipos.** El que llama no necesita conocer el tipo específico.

```python
models = [RandomForestModel({}), XGBoostModel({"n_estimators": 100})]
for model in models:
    model.train(X, y)       # cada model.train() hace cosas diferentes
    predictions = model.predict(X_test)  # pero la interfaz es la misma
```

**Duck typing**: "Si camina como pato y suena como pato, es un pato." Python no requiere herencia de interfaz formal — cualquier objeto con un método `.train()` puede usarse donde se espera algo "entrenable".

### 2.4 Abstracción

**La complejidad se oculta detrás de una interfaz simple.** El usuario de una clase no necesita saber cómo funciona internamente.

```python
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=10)),
    ("classifier", RandomForestClassifier()),
])
pipeline.fit(X_train, y_train)  # No necesito saber cómo funciona cada paso
```

---

## 3. OOP en Python en profundidad

### 3.1 Métodos mágicos

Los métodos mágicos (métodos dunder) definen cómo se comportan los objetos con la sintaxis de Python.

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

### 3.2 Propiedades

Las propiedades permiten acceso controlado a atributos con lógica getter/setter:

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
print(model.is_trained)   # se lee como un atributo, pero usa el getter
```

### 3.3 Métodos de clase y métodos estáticos

```python
class DataLoader:
    format = "csv"               # atributo de clase — compartido por todas las instancias

    @classmethod
    def from_csv(cls, path):     # factory — retorna una instancia
        instance = cls()
        instance.data = pd.read_csv(path)
        return instance

    @classmethod
    def from_parquet(cls, path): # otro factory
        instance = cls()
        instance.data = pd.read_parquet(path)
        return instance

    @staticmethod
    def validate(df: pd.DataFrame) -> bool:  # utilidad — no self, no cls
        return not df.empty and df.columns.duplicated().sum() == 0

loader = DataLoader.from_csv("data.csv")
```

- **@classmethod**: recibe la clase (no la instancia). Se usa para métodos factory.
- **@staticmethod**: no recibe nada especial. Se usa para funciones de utilidad relacionadas con la clase.

### 3.4 Composición sobre herencia

**Prefiere "tiene-un" sobre "es-un".** En lugar de heredar, construye objetos que contienen otros objetos.

```python
# Enfoque de herencia (menos flexible)
class PreprocessingPipeline(Pipeline):
    pass

# Enfoque de composición (más flexible)
class PreprocessingPipeline:
    def __init__(self):
        self.scaler = StandardScaler()      # tiene-un scaler
        self.encoder = OneHotEncoder()      # tiene-un encoder
        self.selector = SelectKBest()       # tiene-un selector

    def fit_transform(self, X, y=None):
        X = self.scaler.fit_transform(X)
        X = self.encoder.fit_transform(X)
        X = self.selector.fit_transform(X, y)
        return X
```

La composición es más flexible: puedes intercambiar componentes, agregar pasos y probar cada pieza de forma independiente. Este principio se aplica ampliamente en [[Data Structures & Algorithms]], donde las estructuras complejas se construyen a partir de otras más simples.

---

## 4. OOP vs Programación Funcional — Cuándo usar cada una

Esta es una de las distinciones más importantes que debe entender un científico de datos. Ambos paradigmas son herramientas; elegir el correcto depende de lo que estés construyendo.

### 4.1 Usa OOP cuando...

**1. Tienes estado que cambia con el tiempo**

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

El tracker tiene **estado** (lista de métricas, tiempo de inicio) que se acumula a través de las llamadas a métodos. Esto es natural en OOP, incómodo en FP.

**2. Tienes múltiples objetos que comparten comportamiento pero difieren en detalles**

La herencia y el polimorfismo brillan aquí. Un `RandomForestModel` y un `XGBoostModel` ambos entrenan y predicen, pero lo hacen de manera diferente. El código que los llama no se preocupa por eso.

**3. Quieres modelar entidades del mundo real**

Usuario, Dataset, Modelo, Experimento, Pipeline — estos se mapean naturalmente a objetos con datos y comportamiento.

**4. Necesitas imponer una interfaz**

Las clases base con métodos abstractos aseguran que todas las subclases implementen los métodos requeridos. Esto evita sorpresas en tiempo de ejecución.

### 4.2 Usa Programación Funcional cuando...

**1. Estás transformando datos a través de un pipeline**

```python
def clean(df): return df.dropna()
def normalize(df): return (df - df.mean()) / df.std()
def encode(df): return pd.get_dummies(df)

pipeline = compose(encode, normalize, clean)  # clean → normalize → encode
result = pipeline(raw_data)
```

Cada función toma datos, retorna datos transformados y **no tiene efectos secundarios**. Este es el paradigma natural para el preprocesamiento de datos.

**2. Las operaciones no tienen estado y son independientes**

```python
def compute_accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()

def compute_precision(y_true, y_pred):
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0
```

Estas funciones dependen solo de sus entradas. Son fáciles de probar (sin configuración previa), fáciles de paralelizar (sin estado compartido) y fáciles de razonar.

**3. Quieres paralelizar o distribuir el cómputo**

Las funciones sin estado se pueden mapear de forma segura entre particiones, clusters o GPUs. Spark, Dask y Ray aprovechan patrones funcionales.

```python
results = parallel_map(compute_metric, [(y1, p1), (y2, p2), ...])
```

**4. Estás componiendo operaciones pequeñas y reutilizables**

Funciones pequeñas como `scale`, `clip`, `log_transform` se pueden componer en cualquier orden. Esto es más natural con FP que con OOP.

### 4.3 Combinando ambos — El punto óptimo en ciencia de datos

En la práctica, el mejor código de ciencia de datos usa **ambos**:

```python
class ModelPipeline:                          # OOP — gestiona estado
    def __init__(self, preprocessors: list, model):
        self.transformers = preprocessors     # FP — funciones sin estado
        self.model = model

    def _apply_transformers(self, X):         # Composición estilo FP
        for fn in self.transformers:
            X = fn(X)                         # cada fn es pura
        return X

    def fit(self, X, y):
        X = self._apply_transformers(X)       # FP: los datos fluyen
        self.model.fit(X, y)                  # OOP: el modelo tiene estado

    def predict(self, X):
        X = self._apply_transformers(X)
        return self.model.predict(X)
```

**Regla general**:
- **OOP para el esqueleto**: clases que mantienen estado, gestionan recursos, proveen interfaces
- **FP para el interior**: transformaciones sin estado, cálculos de métricas, procesamiento de datos

scikit-learn mismo sigue esto: métodos `fit()` / `predict()` (interfaz OOP) con métodos `transform()` que a menudo no tienen estado (FP).

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
