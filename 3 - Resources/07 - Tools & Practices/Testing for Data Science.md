---
tags: [tools, testing, core]
status: growing
created: 2026-06-27
---

# Testing for Data Science

## 1. Escenario de aprendizaje

Trabajas en un modelo de clasificación para producción. Un día, el accuracy cae de 92 % a 70 % sin que salte ningún error en el código. El problema es que los datos de entrada cambiaron silenciosamente —una columna comenzó a llegar con 40 % de valores nulos— y ningún test lo detectó. En ciencia de datos, el código no testeado es una bomba de tiempo: errores silenciosos, datos corruptos, modelos que fallan en producción sin crashear.

El testing en ciencia de datos no es opcional —es la diferencia entre "funciona en mi máquina" y "funciona de forma confiable en producción".

---

## 2. La Pirámide de Testing para ML

```
    /\
   /  \          Tests E2E / smoke (pocos, costosos)
  /    \
 / Unit \        Tests unitarios (muchos, rápidos, baratos)
/________\
```

ML agrega dos capas extra: tests de datos y tests de modelo. Usa herramientas de [[CLI & Productivity]] para ejecutar tests eficientemente desde la terminal.

### 2.1 Tests Unitarios

Prueba funciones individuales de forma aislada (consulta [[Python Fundamentals]] para conceptos básicos de testing):

```python
# test_preprocessing.py
def test_clean_text():
    assert clean_text("  Hello!  ") == "hello"
    assert clean_text("") == ""
    assert clean_text(None) is None

def test_normalize():
    arr = [1, 2, 3, 4, 5]
    normalized = normalize(arr)
    assert abs(normalized.mean()) < 1e-10
    assert abs(normalized.std() - 1.0) < 1e-10
```

```python
# test_preprocessing.py
def test_clean_text():
    assert clean_text("  Hello!  ") == "hello"
    assert clean_text("") == ""
    assert clean_text(None) is None

def test_normalize():
    arr = [1, 2, 3, 4, 5]
    normalized = normalize(arr)
    assert abs(normalized.mean()) < 1e-10
    assert abs(normalized.std() - 1.0) < 1e-10
```

**Qué testear**:
- Casos borde: entrada vacía, NaN, None, valores extremos
- Condiciones límite: mínimo/máximo, un solo elemento
- Transformaciones esperadas: "la salida debe tener la propiedad X"

### 2.2 Tests de Datos

Prueba los datos mismos:

```python
# test_data.py
def test_no_negative_values():
    assert (df["age"] >= 0).all()

def test_schema():
    expected_columns = {"id", "age", "income", "target"}
    assert set(df.columns) == expected_columns

def test_unique_ids():
    assert df["id"].is_unique

def test_cardinality():
    valid_categories = {"A", "B", "C"}
    assert set(df["category"].unique()).issubset(valid_categories)
```

**Herramientas**: Great Expectations, Pandera.

### 2.3 Tests de Modelo

Prueba el comportamiento del modelo:

```python
# test_model.py
def test_model_improves_over_baseline():
    model = train_model(X_train, y_train)
    accuracy = evaluate(model, X_test, y_test)
    baseline = y_test.value_counts(normalize=True).max()
    assert accuracy > baseline

def test_expected_direction():
    # Si 'age' aumenta, 'risk' no debería disminuir
    inputs = df.copy()
    inputs["age"] = inputs["age"] * 1.1
    new_preds = model.predict(inputs)
    assert (new_preds >= original_preds).all()

def test_prediction_range():
    preds = model.predict(X_test)
    assert (preds >= 0).all()
    assert (preds <= 1).all()
```

Registra los resultados de tests con [[Experiment Tracking]] para monitorear el rendimiento del modelo a lo largo del tiempo.

### 2.4 Tests de Integración

Prueba el pipeline completo de extremo a extremo:

```python
# test_pipeline.py
def test_end_to_end():
    # Usar un conjunto pequeño de datos
    X_small = X_train[:100]
    y_small = y_train[:100]

    pipeline = build_pipeline()
    pipeline.fit(X_small, y_small)
    preds = pipeline.predict(X_test[:10])

    assert len(preds) == 10
    assert (preds >= 0).all()
```

---

## 3. Usando pytest

```python
# Ejecutar todos los tests
pytest

# Ejecutar un archivo específico
pytest tests/test_preprocessing.py

# Ejecutar tests que coincidan con un nombre
pytest -k "clean"

# Salida detallada
pytest -v

# Ver cobertura de código
pytest --cov=src tests/
```

---

## 4. CI/CD para ML

```yaml
# .github/workflows/test.yml
name: ML pipeline tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run unit tests
        run: pytest tests/
      - name: Run data validation
        run: great_expectations checkpoint run
```

---

## 5. Qué NO Testear

| No testear | Por qué |
|---|---|
| La salida exacta del modelo (ej.: "accuracy = 0.85") | Es estocástico — dará falsos fallos |
| Detalles internos de implementación | Los cambios rompen tests innecesariamente |
| Datos muy grandes en CI | Lento y costoso |

**Prueba propiedades, no valores exactos.**

---

## 6. Common Mistakes

1. **No tests at all**: the most common. Untested ML code will fail in production, and it will fail silently.

2. **Testing exact numerical outputs**: model training is stochastic. Test properties (range, direction, improvement over baseline) instead.

3. **Not separating unit and integration tests**: slow integration tests discourage running tests. Keep unit tests fast (< 1 second each).

4. **Testing on the full dataset**: use a small sample (100 rows) for fast, frequent testing. Test on full data only in scheduled pipelines.

5. **Not testing data quality**: if bad data enters the pipeline, no amount of code testing will save it.

---

## 7. Check Your Understanding

1. Why should you test properties of model outputs rather than exact accuracy values? (Training is stochastic — the exact number varies between runs. Properties are stable.)

2. A data validation test catches that a column is 40% null. What do you do? (Investigate the data source. Decide: impute, drop the column, or block the pipeline.)

3. Your integration test takes 30 minutes. What problem does this cause? (Developers will not run it locally. Split into fast unit tests + scheduled integration tests.)

4. What is the difference between a unit test and a data test? (Unit test: code correctness. Data test: data correctness. Both are needed.)

5. You train a model and run `assert accuracy > 0.9`. It fails. What could have changed? (Data distribution, code logic, or random seed.)

---

## 8. Summary

Testing in data science covers four areas: unit tests (code correctness), data tests (data quality), model tests (behavioral properties), and integration tests (pipeline correctness). Use pytest for unit tests, Great Expectations for data validation, and CI/CD to automate everything. Test properties, not exact values. The goal is catching problems before they reach production.

---

## 9. Where to Go Next

- [[Git]] — Version control for reproducible testing
- [[Code Quality]] — Linting, formatting, and type hints
- [[ML Pipelines]] — Building testable pipelines
- [[Virtual Environments]] — Testing across reproducible environments
