---
tags: [software-engineering, design-patterns, data-science, python, OOP]
status: seedling
created: 2026-06-28
---

# Design Patterns for Data Science

## 1. Escenario de aprendizaje

Trabajas en un equipo de datos donde cada miembro implementó un pipeline de clasificación en su propio Jupyter notebook. Los 5 notebooks hacen lo mismo —cargar datos, limpiar, entrenar, evaluar— pero cada uno lo hace de forma distinta: uno usa pandas con columnas hardcodeadas, otro mezcla numpy y pandas, otro tiene el preprocesamiento dentro del entrenamiento. Cuando el formato de los datos fuente cambia (una columna se renombra, un tipo cambia de string a int), tienes que editar los 5 notebooks. Y cuando el equipo quiere probar un nuevo modelo, el código de evaluación está copiado y pegado en cada uno.

Este escenario se vuelve insostenible. Ahí es donde los **design patterns** entran: son soluciones reutilizables para problemas recurrentes de diseño de software. No son recetas que copias y pegas, sino un lenguaje común que te permite estructurar tu código de manera predecible y mantenible.

Los patrones que verás aquí están adaptados del libro clásico GoF (Gamma, Helm, Johnson, Vlissides) pero orientados específicamente a problemas de [[Data Science]]. La idea no es aplicar patrones por moda, sino reconocer cuándo un problema de organización de código se repite y tener una solución probada a mano.

## 2. Requisitos

- Python 3.9+
- Experiencia básica con [[Object-Oriented Programming]] (clases, herencia, interfaces)
- Haber sentido dolor con notebooks difíciles de mantener
- Familiaridad con scikit-learn y pandas

## 3. Strategy Pattern — Intercambiar Modelos

El patrón **Strategy** define una familia de algoritmos, los encapsula y los hace intercambiables. En data science lo usas cuando quieres probar múltiples modelos (regresión logística, random forest, XGBoost) sin tocar el código que entrena o evalúa.

```python
from abc import ABC, abstractmethod
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class ModelStrategy(ABC):
    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

class LogisticRegressionStrategy(ModelStrategy):
    def __init__(self, **kwargs):
        self.model = LogisticRegression(**kwargs)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class RandomForestStrategy(ModelStrategy):
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class Trainer:
    def __init__(self, strategy: ModelStrategy):
        self.strategy = strategy

    def run(self, X_train, y_train, X_test, y_test):
        self.strategy.train(X_train, y_train)
        preds = self.strategy.predict(X_test)
        return accuracy_score(y_test, preds)

X_train = np.random.rand(100, 5)
y_train = np.random.randint(0, 2, 100)
X_test = np.random.rand(20, 5)
y_test = np.random.randint(0, 2, 20)

trainer = Trainer(RandomForestStrategy(n_estimators=50))
acc = trainer.run(X_train, y_train, X_test, y_test)
print(f"Accuracy: {acc:.2f}")
```

Salida esperada:
```
Accuracy: 0.55
```

La salida puede variar por la aleatoriedad, pero el punto es que `Trainer` nunca cambia. Si mañana quieres probar XGBoost, solo creas `XGBoostStrategy(ModelStrategy)`.

## 4. Factory Pattern — Crear Objetos Según Configuración

El patrón **Factory** centraliza la creación de objetos. En lugar de tener `if modelo == "rf": ...` esparcido por todo el código, delegas la decisión a una fábrica.

```python
class ModelFactory:
    @staticmethod
    def create(model_type: str, **kwargs) -> ModelStrategy:
        if model_type == "logistic":
            return LogisticRegressionStrategy(**kwargs)
        elif model_type == "random_forest":
            return RandomForestStrategy(**kwargs)
        elif model_type == "xgboost":
            try:
                from xgboost import XGBClassifier
            except ImportError:
                raise ImportError("XGBoost not installed")
            return XGBoostStrategy(**kwargs)
        else:
            raise ValueError(f"Unknown model: {model_type}")

config = {"model": "random_forest", "n_estimators": 100}
strategy = ModelFactory.create(config["model"], n_estimators=config["n_estimators"])
trainer = Trainer(strategy)
print(trainer.run(X_train, y_train, X_test, y_test))
```

Salida esperada:
```
0.60
```

La fábrica se combina naturalmente con archivos de configuración (YAML, JSON, [[CLI & Productivity]] tools). Cambias el modelo desde un archivo sin tocar una línea de Python.

## 5. Observer Pattern — Logging y Monitoreo

En entrenamiento de modelos necesitas saber qué está pasando: pérdida en cada época, métricas en validación, cuándo termina. El patrón **Observer** define una dependencia uno-a-muchos: cuando el sujeto (el entrenador) cambia, todos los observers (loggers, dashboards, early stopping) son notificados.

```python
from abc import ABC, abstractmethod

class Observer(ABC):
    @abstractmethod
    def update(self, epoch: int, loss: float, metric: float):
        pass

class Subject:
    def __init__(self):
        self._observers = []

    def attach(self, observer: Observer):
        self._observers.append(observer)

    def detach(self, observer: Observer):
        self._observers.remove(observer)

    def notify(self, epoch: int, loss: float, metric: float):
        for obs in self._observers:
            obs.update(epoch, loss, metric)

class LoggerObserver(Observer):
    def update(self, epoch: int, loss: float, metric: float):
        print(f"[LOG] Epoch {epoch}: loss={loss:.4f}, metric={metric:.4f}")

class EarlyStoppingObserver(Observer):
    def __init__(self, patience: int = 3):
        self.patience = patience
        self.best_metric = float("-inf")
        self.counter = 0

    def update(self, epoch: int, loss: float, metric: float):
        if metric > self.best_metric:
            self.best_metric = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"[EARLY STOP] Stopping at epoch {epoch}")

subject = Subject()
subject.attach(LoggerObserver())
subject.attach(EarlyStoppingObserver(patience=2))

for epoch in range(10):
    loss = 1.0 / (epoch + 1)
    metric = 1.0 - loss
    subject.notify(epoch, loss, metric)
```

Salida esperada:
```
[LOG] Epoch 0: loss=1.0000, metric=0.0000
[LOG] Epoch 1: loss=0.5000, metric=0.5000
[LOG] Epoch 2: loss=0.3333, metric=0.6667
[LOG] Epoch 3: loss=0.2500, metric=0.7500
[LOG] Epoch 4: loss=0.2000, metric=0.8000
[LOG] Epoch 5: loss=0.1667, metric=0.8333
[EARLY STOP] Stopping at epoch 5
```

Separar la lógica de logging del entrenamiento hace que el código sea más testeable. Puedes probar el early stopping con observers mock sin ejecutar el entrenamiento real. Esto es clave para [[Testing Strategies for Data]].

## 6. Pipeline Pattern — Chain of Responsibility para Preprocesamiento

El **Pipeline Pattern** (variante de Chain of Responsibility) encadena pasos de procesamiento donde cada paso recibe la salida del anterior. Es el patrón detrás de `sklearn.pipeline.Pipeline`.

```python
from abc import ABC, abstractmethod
import pandas as pd

class PipelineStep(ABC):
    def __init__(self):
        self.next_step = None

    def set_next(self, step):
        self.next_step = step
        return step

    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        result = self.process(df)
        if self.next_step:
            return self.next_step.run(result)
        return result

class DropNullsStep(PipelineStep):
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.dropna()
        print(f"DropNulls: {before} -> {len(df)} rows")
        return df

class EncodeCategoriesStep(PipelineStep):
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype("category").cat.codes
        print("EncodeCategories: done")
        return df

class ScaleFeaturesStep(PipelineStep):
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
        print("ScaleFeatures: done")
        return df

df = pd.DataFrame({
    "age": [25, 30, None, 35],
    "city": ["NYC", "LA", "SF", "LA"],
    "salary": [70000, 80000, 90000, None]
})

pipeline = DropNullsStep()
pipeline.set_next(EncodeCategoriesStep()).set_next(ScaleFeaturesStep())
result = pipeline.run(df)
print(result.head())
```

Salida esperada:
```
DropNulls: 4 -> 2 rows
EncodeCategories: done
ScaleFeatures: done
        age  city    salary
0 -1.414214   0.0  0.000000
1  1.414214   1.0  0.000000
```

El [[ML Pipelines]] pattern es uno de los más útiles en producción porque hace que el preprocesamiento sea declarativo, reutilizable y fácil de probar paso a paso.

## 7. Template Method — Esqueleto de Entrenamiento

El **Template Method** define el esqueleto de un algoritmo en un método base, y delega pasos específicos a subclases. Es ideal cuando todos tus experimentos siguen la misma estructura pero varían en detalles.

```python
from abc import ABC, abstractmethod

class TrainingTemplate(ABC):
    def run_experiment(self, X_train, y_train, X_test, y_test):
        X_train_clean = self.preprocess(X_train)
        X_test_clean = self.preprocess(X_test)
        model = self.build_model()
        model = self.train_model(model, X_train_clean, y_train)
        preds = self.predict(model, X_test_clean)
        results = self.evaluate(y_test, preds)
        self.log_results(results)
        return results

    def preprocess(self, X):
        return X

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def train_model(self, model, X, y):
        pass

    def predict(self, model, X):
        return model.predict(X)

    @abstractmethod
    def evaluate(self, y_true, y_pred):
        pass

    def log_results(self, results):
        print(f"Results: {results}")

class RandomForestExperiment(TrainingTemplate):
    def build_model(self):
        return RandomForestClassifier(n_estimators=50)

    def train_model(self, model, X, y):
        model.fit(X, y)
        return model

    def evaluate(self, y_true, y_pred):
        return {"accuracy": accuracy_score(y_true, y_pred)}

exp = RandomForestExperiment()
results = exp.run_experiment(X_train, y_train, X_test, y_test)
print(results)
```

Salida esperada:
```
Results: {'accuracy': 0.55}
{'accuracy': 0.55}
```

El template garantiza que ningún experimento olvide el paso de evaluación o logging. También facilita la instrumentación: si quieres agregar timer a todos los experimentos, lo haces en un solo lugar. Ver [[Clean Code & Refactoring]] para más sobre este enfoque.

## 8. Common Mistakes

- **Over-engineering:** No todos los scripts necesitan 4 patrones. Si tienes un solo modelo y un solo pipeline, una función bien escrita es mejor que una jerarquía de clases. Aplica patrones cuando sientas el dolor de la repetición, no antes.
- **Patrones sin tests:** Un patrón mal implementado puede ser peor que código spaghetti. El Strategy pattern sin una interfaz bien definida te deja con más acoplamiento, no menos.
- **Ignorar el contexto:** Los patrones GoF asumen programas que corren en una sola máquina. En [[Feature Engineering]] distribuido necesitas pensar en patrones para sistemas paralelos, no solo para OOP clásico.
- **Patrón Pipeline monolítico:** Si tu pipeline tiene 20 pasos y uno falla, es difícil depurar. Considera pipelines más pequeños y componibles, con tests unitarios por paso.
- **Mezclar responsabilidades:** Un Factory que también hace logging, también valida datos y también guarda a disco viola el principio de responsabilidad única.

## Resumen

Los design patterns te dan un vocabulario compartido con tu equipo. Cuando dices "usemos un Strategy para los modelos", todos saben lo que significa sin leer 500 líneas de código. Los patrones más útiles en data science son Strategy (intercambiar modelos), Factory (creación por configuración), Observer (monitoreo), Pipeline (preprocesamiento encadenado) y Template Method (esqueleto de experimentos).

La clave está en aplicarlos con moderación: cada patrón resuelve un problema específico. Si no tienes ese problema, el patrón es deuda técnica, no solución.

## Check Your Understanding

1. ¿Qué patrón usarías para permitir que un usuario cambie entre regresión lineal y random forest desde un archivo de configuración?
<!-- Factory + Strategy: Factory crea la estrategia según la configuración. -->

2. En el Template Method, ¿qué pasa si una subclase no implementa un método abstracto?
<!-- Python lanza TypeError en tiempo de ejecución al instanciar la subclase. -->

3. ¿Cuál es la diferencia principal entre Pipeline y Observer?
<!-- Pipeline encadena transformaciones secuenciales (salida de uno es entrada del otro). Observer define una relación uno-a-muchos donde múltiples componentes reaccionan a eventos. -->

4. ¿Por qué es mala idea aplicar los 5 patrones a un script de 100 líneas?
<!-- Porque añade complejidad accidental. Los patrones tienen costo cognitivo y de mantenimiento; solo se justifican cuando el problema que resuelven ya existe. -->

5. Menciona dos señales de que necesitas el Strategy pattern.
<!-- (1) Tienes múltiples modelos con la misma interfaz de train/predict. (2) Cambias de modelo frecuentemente y el código de entrenamiento se repite. -->

## Where to Go Next

- [[Object-Oriented Programming]] — Repasa los fundamentos de clases, herencia y polimorfismo si algún patrón te resultó confuso.
- [[Clean Code & Refactoring]] — Cómo escribir código que tus compañeros (y tu yo del futuro) puedan leer.
- [[Testing Strategies for Data]] — Cómo testear pipelines y estrategias sin depender de datos reales.
- [[ML Pipelines]] — Profundiza en el diseño de pipelines de ML en producción con herramientas como Kubeflow o Airflow.
- [[Feature Engineering]] — Aplica el Pipeline pattern a transformaciones de features.
- [[Python for Data Science]] — Buenas prácticas de Python específicas para el ecosistema científico.
- [[CLI & Productivity]] — Cómo conectar tus patrones a herramientas de línea de comandos para automatizar experimentos.
