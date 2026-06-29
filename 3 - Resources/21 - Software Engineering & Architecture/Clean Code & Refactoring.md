---
tags:
  - software-engineering
  - clean-code
  - refactoring
  - code-quality
status: seedling
created: 2026-06-28
---

# Clean Code & Refactoring

## 1. Escenario de aprendizaje

Tu notebook de ML funciona — los resultados son correctos, el accuracy es aceptable. Pero abres el archivo y sientes un escalofrío. Variables llamadas `df2`, `x`, `tmp`. Funciones de 200 líneas que cargan datos, transforman columnas, entrenan un modelo y grafican resultados, todo mezclado. Los imports están desperdigados en seis celdas distintas. La misma lógica de validación aparece copiada en tres lugares.

Un día necesitas cambiar el preprocessing porque los datos de origen cambiaron. Modificas una función. Otra parte del pipeline se rompe silenciosamente. Pasan dos semanas antes de que alguien descubra que las predicciones están degradadas. Clean Code no es estética — es ingeniería. Es la diferencia entre un sistema frágil que da miedo tocar y una base de código que puedes modificar con confianza.

Este escenario es real en prácticamente todos los equipos de datos que crecen rápido. El código de ML comienza como experimentación, y la experimentación prioriza velocidad sobre estructura. Pero cuando ese código experimental llega a producción, la deuda técnica acumulada empieza a cobrar intereses.

## 2. Requisitos

- Python 3.9+
- Experiencia escribiendo código de ML/DL en notebooks o scripts
- Familiaridad con pandas, scikit-learn
- Deseable: experiencia trabajando en equipo de datos

## 3. Naming

Elegir nombres es la habilidad más infravalorada en programación. Un buen nombre convierte código que hay que leer con lupa en código que se lee como prosa.

```python
# Mal: nombres que ocultan intención
def proc(d):
    d2 = d.copy()
    d2.dropna(inplace=True)
    x = d2[['age', 'income', 'education']]
    y = d2['default']
    m = LogisticRegression()
    m.fit(x, y)
    return m

# Bien: nombres que revelan intención
def train_default_risk_model(
    raw_data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str
) -> LogisticRegression:
    clean_data = raw_data.dropna()
    features = clean_data[feature_columns]
    target = clean_data[target_column]
    model = LogisticRegression()
    model.fit(features, target)
    return model
```

**Reglas prácticas:**
- Variables: sustantivos que describen el contenido (`customer_data`, no `d`)
- Funciones: verbos que describen la acción (`train_model`, no `proc`)
- Clases: sustantivos que describen el concepto (`DataProcessor`, no `DProc`)
- Booleanos: prefijos `is_`, `has_`, `should_` (`is_active`, no `flag`)
- Evitar abreviaturas: `transformer` vs `trfmr`, `normalizer` vs `norm`

```python
# Salida esperada:
# La versión "bien" permite entender el pipeline sin leer
# cada línea. Cualquier persona del equipo entiende qué hace
# train_default_risk_model con solo ver su firma.
# La versión "mal" requiere abrir la función y leer línea
# por línea para adivinar el propósito.
```

Los nombres deben reflejar el dominio del problema, no la implementación. En un sistema de recomendación, `user_item_matrix` es mejor que `sparse_mat`. `cold_start_strategy` es mejor que `method_3`.

## 4. Funciones pequeñas

Una función debe hacer una sola cosa y hacerla bien. El límite práctico es 20 líneas. Si una función tiene más de 20 líneas, probablemente está haciendo múltiples cosas.

```python
# Mal: una función que hace todo
def run_pipeline(data_path):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    data = pd.read_csv(data_path)
    data = data.dropna()
    data = data[data['amount'] > 0]
    data['log_amount'] = np.log(data['amount'] + 1)
    scaler = StandardScaler()
    data[['age', 'income']] = scaler.fit_transform(data[['age', 'income']])
    features = data.drop('fraud', axis=1)
    target = data['fraud']
    X_train, X_test, y_train, y_test = train_test_split(features, target)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, preds)}')
    return model

# Bien: funciones con una responsabilidad
def load_and_clean_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    data = data.dropna()
    data = data[data['amount'] > 0]
    return data

def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data['log_amount'] = np.log(data['amount'] + 1)
    return data

def train_model(
    features: pd.DataFrame, target: pd.Series
) -> RandomForestClassifier:
    X_train, X_test, y_train, y_test = train_test_split(features, target)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model
```

```python
# Salida esperada:
# Cada función individual es testeable, reutilizable y
# mantenible. Puedes cambiar RandomForestClassifier por
# XGBoost sin tocar load_and_clean_data ni engineer_features.
# Puedes testear engineer_features con datos sintéticos.
```

## 5. Principio DRY (Don't Repeat Yourself)

En pipelines de datos, la duplicación aparece en validaciones, transformaciones y feature engineering.

```python
# Mal: lógica duplicada
def validate_credit_data(data):
    assert data['age'].between(18, 100).all()
    assert data['income'].between(0, 1_000_000).all()

def validate_loan_data(data):
    assert data['age'].between(18, 100).all()
    assert data['income'].between(0, 1_000_000).all()
    assert data['loan_amount'].between(0, 1_000_000).all()

# Bien: validación reusable
def validate_column_range(
    data: pd.DataFrame,
    column: str,
    min_val: float,
    max_val: float
):
    assert data[column].between(min_val, max_val).all(), \
        f'{column} fuera de rango [{min_val}, {max_val}]'

def validate_credit_data(data):
    validate_column_range(data, 'age', 18, 100)
    validate_column_range(data, 'income', 0, 1_000_000)

def validate_loan_data(data):
    validate_column_range(data, 'age', 18, 100)
    validate_column_range(data, 'income', 0, 1_000_000)
    validate_column_range(data, 'loan_amount', 0, 1_000_000)
```

```python
# Salida esperada:
# Si el rango de income cambia, modificas un solo lugar.
# validate_loan_data gana claridad al delegar la lógica
# repetitiva a una función genérica.
```

## 6. SOLID en Data Science

SOLID son cinco principios de diseño orientado a objetos. Aquí nos enfocamos en los dos más relevantes para pipelines de datos.

**Single Responsibility Principle (SRP):** Cada clase/módulo debe tener una sola razón para cambiar. Separa carga de datos, transformación y entrenamiento en componentes distintos.

```python
# Mal: una clase que hace demasiado
class FraudPipeline:
    def load_data(self): ...
    def clean_data(self): ...
    def engineer_features(self): ...
    def train_model(self): ...
    def evaluate(self): ...
    def deploy(self): ...

# Bien: responsabilidades separadas
class DataLoader: ...
class DataCleaner: ...
class FeatureEngineer: ...
class ModelTrainer: ...
class ModelEvaluator: ...
```

**Dependency Injection:** Las dependencias (como el modelo) deben ser inyectadas, no instanciadas dentro de la función.

```python
# Mal: acoplado a una implementación específica
def train():
    model = RandomForestClassifier()
    model.fit(X, y)

# Bien: puedes inyectar cualquier modelo
def train(model: Any, X: pd.DataFrame, y: pd.Series):
    model.fit(X, y)
    return model
```

```python
# Salida esperada:
# Con DI puedes intercambiar modelos sin modificar la
# función train. Pruebas unitarias: inyectas un mock.
# Producción: inyectas el modelo real.
```

## 7. Refactoring patterns

**Extract Method:** Convierte un bloque de código en una función con nombre.

```python
# Antes
data['age_group'] = data['age'].apply(
    lambda x: 'young' if x < 30 else 'mid' if x < 60 else 'senior'
)

# Después
def categorize_age(age: int) -> str:
    if age < 30:
        return 'young'
    elif age < 60:
        return 'mid'
    return 'senior'

data['age_group'] = data['age'].apply(categorize_age)
```

**Introduce Parameter Object:** Agrupa parámetros que siempre viajan juntos.

```python
# Antes
def train_model(data, lr, max_depth, n_estimators, subsample):
    ...

train_model(data, 0.01, 5, 100, 0.8)

# Después
@dataclass
class ModelConfig:
    learning_rate: float
    max_depth: int
    n_estimators: int
    subsample: float

def train_model(data: pd.DataFrame, config: ModelConfig):
    ...

config = ModelConfig(learning_rate=0.01, max_depth=5, n_estimators=100, subsample=0.8)
train_model(data, config)
```

**Replace Magic Number:** Los números literales sin explicación son "magic numbers".

```python
# Mal: magic numbers
if user_age > 65:
    discount = price * 0.2

# Bien: constantes con nombre
SENIOR_AGE_THRESHOLD = 65
SENIOR_DISCOUNT_RATE = 0.2

if user_age > SENIOR_AGE_THRESHOLD:
    discount = price * SENIOR_DISCOUNT_RATE
```

## 8. Common Mistakes

**Refactorizar sin tests:** Cambiar la estructura del código sin una red de seguridad es la receta del desastre. Siempre escribe tests antes de refactorizar.

**Mejorar código que no se usa:** Es tentador aplicar Clean Code a todo. Prioriza el código que se ejecuta frecuentemente o que es crítico para el negocio.

**Perfeccionismo prematuro:** No refactorices hasta que entiendas el problema. A veces un notebook feo que se ejecuta una vez no merece una arquitectura hexagonal.

**Renombrar sin actualizar referencias:** Cambiar el nombre de una función y olvidar actualizar las llamadas rompe el pipeline. Usa herramientas de refactoring automáticas.

**Refactorizar y agregar funcionalidad en el mismo cambio:** Un commit debe ser o refactorización o nueva funcionalidad. Nunca ambas. Mezclarlos hace imposible el code review y el debugging.

```python
# Salida esperada (Common Mistakes):
# Si no tienes tests, no sabes si tu refactor cambió
# el comportamiento. Si mejoras código que nadie usa,
# desperdicias tiempo. Si refactorizas antes de entender
# el dominio, probablemente tendrás que refactorizar
# de nuevo.
```

## Resumen

Clean Code es una disciplina de ingeniería que transforma código frágil en código mantenible. Las prácticas fundamentales son: nombres que revelan intención, funciones pequeñas con una sola responsabilidad, eliminar duplicación con DRY, aplicar SOLID (especialmente SRP y Dependency Injection), y usar patrones de refactoring como Extract Method e Introduce Parameter Object. El objetivo no es la perfección estética sino la sostenibilidad del proyecto a largo plazo.

## Check Your Understanding

1. ¿Por qué es importante que una función tenga una sola responsabilidad? ¿Qué problemas previene?
<!-- Porque facilita testing, reutilización y debugging. Una función que hace varias cosas es difícil de testear, tiene múltiples razones para cambiar y oculta bugs. -->

2. Tu colega te dice: "No necesito Clean Code, mi notebook funciona bien." ¿Cómo respondes?
<!-- El código no solo comunica instrucciones a la computadora, también comunica intención a otros humanos (y a ti mismo en el futuro). Un notebook que "funciona" pero es ilegible se vuelve insostenible cuando crece. -->

3. ¿Cuándo está justificado violar DRY?
<!-- Cuando la duplicación es accidental y los fragmentos podrían evolucionar de forma independiente. Forzar DRY prematuramente puede introducir acoplamiento innecesario. La regla es "Rule of Three": si ves el mismo patrón tres veces, refactoriza. -->

4. ¿Qué principio SOLID permite intercambiar modelos sin modificar el código de entrenamiento?
<!-- Dependency Injection: el modelo se pasa como parámetro en lugar de instanciarse dentro de la función de entrenamiento. -->

5. Durante un refactor, ¿qué debes tener antes de cambiar cualquier línea de código?
<!-- Tests que verifiquen el comportamiento actual. Sin tests, refactorizar es como hacer cirugía sin anestesia. -->

## Where to Go Next

Especialízate en los patrones de diseño más usados en ciencia de datos con [[Design Patterns for Data Science]]. Para aplicar testing a tu código refactorizado, visita [[Testing Strategies for Data]]. Si quieres llevar tu código a producción, [[Production Recommenders]] cubre los desafíos reales de escalar modelos. La [[Object-Oriented Programming]] note profundiza en PILARES como encapsulación y polimorfismo. Para métricas objetivas de calidad de código, revisa [[Code Quality]]. Si buscas un paradigma complementario, [[Functional Programming]] ofrece alternativas inmutables y declarativas. Y para integrar todo en tu flujo de trabajo diario, [[CLI & Productivity]] te muestra cómo construir herramientas de línea de comandos para tus pipelines.
