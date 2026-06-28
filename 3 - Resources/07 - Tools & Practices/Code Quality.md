---
tags: [tools, code-quality, core]
status: growing
created: 2026-06-27
---

# Code Quality

## 1. Why This Matters

Data science code has a reputation for being messy. Jupyter notebooks with out-of-order cells. Scripts with no type hints. Functions named `process_data()` that do 15 different things. Undocumented configuration values.

This reputation exists because it is often true. But as ML systems move to production, code quality becomes critical. A bug in a model's feature engineering code produces wrong predictions silently. A typo in a config file wastes a week of training. See [[Python for Data Science]] for language-level conventions.

Code quality tools catch these problems automatically — before they cause damage.

---

## 2. Linting

Linters analyze code for potential errors, style violations, and anti-patterns without running it.

### 2.1 Ruff

Ruff is the modern Python linter (replaces flake8, isort, pyupgrade, and others in one tool).

```bash
ruff check .                    # Check all files
ruff check --fix .             # Auto-fix issues
ruff format .                  # Format code
```

```toml
# pyproject.toml
[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP", "B", "SIM"]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]  # unused imports allowed in init
```

**What it catches**:
- **E**: pycodestyle (PEP 8 violations)
- **F**: pyflakes (undefined variables, unused imports)
- **I**: isort (import ordering)
- **N**: naming conventions
- **UP**: pyupgrade (modern Python syntax)
- **B**: flake8-bugbear (common bugs)

### 2.2 Pre-commit Hooks

Run quality checks automatically before every commit:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.5.0
    hooks:
      - id: ruff
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
```

```bash
pre-commit install              # Install hooks
pre-commit run --all-files      # Run on all files
```

If a hook fails, the commit is blocked until the issue is fixed.

---

## 3. Type Hints

Type hints document what types a function expects and returns. They do not affect runtime behavior but enable static analysis.

```python
from typing import Optional, List, Tuple, Dict

def preprocess(
    df: pd.DataFrame,
    columns: List[str],
    drop_na: bool = True,
    fill_value: Optional[float] = None,
) -> pd.DataFrame:
    """Preprocess a DataFrame."""
    if drop_na:
        df = df.dropna()
    if fill_value is not None:
        df = df.fillna(fill_value)
    return df
```

**Benefits**:
- **Documentation**: clear interface without reading the implementation
- **Catch errors**: mypy catches mismatched argument types
- **IDE support**: autocomplete and inline documentation
- **Refactoring**: changing a type reveals all callers

### myPy

```bash
mypy src/  # checks all files in src/
```

Example catch:
```python
def get_age(person: dict) -> int:
    return person["age"]

# mypy: error: Returning Any from function declared to return "int"
# (because person is not typed)
```

**Fix**:
```python
from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int

def get_age(person: Person) -> int:
    return person["age"]  # mypy knows this is int
```

---

## 4. Formatting

Formatters automatically fix code style (indentation, quotes, line length, spacing).

| Tool | Style | Opinionated? |
|---|---|---|
| **Black** | PEP 8 with modifications | Very (few options) |
| **Ruff format** | Compatible with Black | Same as Black |
| **Autopep8** | Pure PEP 8 | Less opinionated |

**Why use a formatter?** It eliminates all style debates. The team agrees: the formatter decides.

```bash
black src/          # Format all files
ruff format src/    # Same as Black, but faster
```

---

## 5. Documentation

### 5.1 Docstrings

```python
def train_model(
    X: np.ndarray,
    y: np.ndarray,
    params: Optional[dict] = None,
) -> RandomForestClassifier:
    """Train a Random Forest classifier.

    Args:
        X: Training features. Shape (n_samples, n_features).
        y: Training labels. Shape (n_samples,).
        params: Optional hyperparameters. Defaults to
            {'n_estimators': 100, 'max_depth': 5}.

    Returns:
        Trained RandomForestClassifier.

    Raises:
        ValueError: If X and y have mismatched lengths.
    """
```

### 5.2 README

Every project should have a README that answers:
- What does this project do?
- How do I set it up? (see [[Virtual Environments]])
- How do I run it?

---

## 6. Code Review Checklist

- [ ] Does the code do what it says?
- [ ] Are there edge cases not handled?
- [ ] Are there tests for new functions?
- [ ] Are type hints correct and complete?
- [ ] Are variable/function names descriptive?
- [ ] Are there no hardcoded values (use constants)?
- [ ] Are experiment parameters tracked via [[Experiment Tracking]]?
- [ ] Is the logic easy to follow?

---

## 7. Common Mistakes

1. **No type hints**: "I know what this function does" — six months later, no one does.

2. **Not using a linter**: simple bugs (undefined variable, unused import) go to runtime. Catch them at write time.

3. **Ignoring linter warnings**: "it is just a style warning" — warnings compound into unmaintainable code. Fix them.

4. **Committing failing code quality checks**: disable hooks or bypass linting → quality degrades. Fix issues before committing.

5. **Writing code for the computer, not for the reader**: code is read far more often than it is written. Optimize for readability.
6. **Not treating pipeline code as production code**: ML pipeline code should follow the same [[ML Pipelines]] quality standards as any production service.

---

## 8. Check Your Understanding

1. You run `ruff check` and get 50 warnings. What do you do? (Fix them all — most are auto-fixable. For persistent ones, configure ruff to allow them.)

2. What does a type hint tell you that a variable name does not? (The exact type — "age" could be int, float, or str. `age: int` is unambiguous.)

3. Why use a formatter like Black over manual formatting? (Eliminates all style discussions. Everyone uses the same style automatically.)

4. A pre-commit hook blocks your commit because of a trailing whitespace. Is this helpful or annoying? (Helpful — prevents whitespace from cluttering future diffs.)

5. You review a PR with no type hints. What do you say? (Request type hints, especially on public functions.)

---

## 9. Summary

Code quality tools catch bugs before they reach production. Ruff lints and formats. myPy enforces type hints. Pre-commit runs checks automatically before every commit. Good documentation (docstrings, README) makes code maintainable. The investment in code quality pays for itself in fewer bugs, faster development, and happier teammates.

---

## 10. Where to Go Next

- [[Testing for Data Science]] — Complementing static analysis with tests
- [[Git]] — Version control workflow
- [[CLI & Productivity]] — Running quality checks from the CLI
