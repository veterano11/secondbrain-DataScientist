---
tags: [tools, testing, core]
status: growing
created: 2026-06-27
---

# Testing for Data Science

## 1. Why This Matters

ML code has a reputation for being hard to test. Models are stochastic, data is messy, and the final output depends on many intermediate steps. But the **consequences of untested ML code are severe**: wrong predictions, silent data corruption, models that fail in production without crashing.

Testing in data science is not optional — it is the difference between "works on my machine" and "works reliably in production."

---

## 2. The Testing Pyramid for ML

```
    /\
   /  \          E2E / smoke tests (few, expensive)
  /    \
 / Unit \         Unit tests (many, fast, cheap)
/________\
```

ML adds two extra layers: data tests and model tests. Use [[CLI & Productivity]] tools to run tests efficiently from the terminal.

### 2.1 Unit Tests

Test individual functions in isolation (see [[Python Fundamentals]] for testing basics):

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

**What to test**:
- Edge cases: empty input, NaN, None, extreme values
- Boundary conditions: min/max, single element
- Expected transformations: "output should have property X"

### 2.2 Data Tests

Test the data itself:

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

**Tools**: Great Expectations, Pandera.

### 2.3 Model Tests

Test model behavior:

```python
# test_model.py
def test_model_improves_over_baseline():
    model = train_model(X_train, y_train)
    accuracy = evaluate(model, X_test, y_test)
    baseline = y_test.value_counts(normalize=True).max()
    assert accuracy > baseline

def test_expected_direction():
    # If 'age' increases, 'risk' should not decrease
    inputs = df.copy()
    inputs["age"] = inputs["age"] * 1.1
    new_preds = model.predict(inputs)
    assert (new_preds >= original_preds).all()

def test_prediction_range():
    preds = model.predict(X_test)
    assert (preds >= 0).all()
    assert (preds <= 1).all()
```

Log test results with [[Experiment Tracking]] to monitor model performance over time.

### 2.4 Integration Tests

Test the full pipeline end-to-end:

```python
# test_pipeline.py
def test_end_to_end():
    # Use a small dataset
    X_small = X_train[:100]
    y_small = y_train[:100]

    pipeline = build_pipeline()
    pipeline.fit(X_small, y_small)
    preds = pipeline.predict(X_test[:10])

    assert len(preds) == 10
    assert (preds >= 0).all()
```

---

## 3. Using pytest

```python
# Run all tests
pytest

# Run a specific file
pytest tests/test_preprocessing.py

# Run tests matching a name
pytest -k "clean"

# Verbose output
pytest -v

# Check code coverage
pytest --cov=src tests/
```

---

## 4. CI/CD for ML

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

## 5. What NOT to Test

| Don't test | Why |
|---|---|
| Exact model output (e.g., "accuracy = 0.85") | Stochastic — will flake |
| Internal implementation details | Changes break tests unnecessarily |
| Very large datasets in CI | Slow, expensive |

**Test properties, not exact values.**

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
