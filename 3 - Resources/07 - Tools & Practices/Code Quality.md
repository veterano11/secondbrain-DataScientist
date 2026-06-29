---
tags: [tools, code-quality, core]
status: growing
created: 2026-06-27
---

# Code Quality

## 1. Escenario de aprendizaje

Tu equipo heredó un código de ciencia de datos con funciones como `procesar()` que hace 15 cosas distintas, sin type hints, y con variables llamadas `tmp` y `datos2`. Un bug en el preprocesamiento pasó desapercibido por semanas y generó predicciones incorrectas en producción. El responsable sonríe y dice "funciona en mi máquina".

Las herramientas de calidad de código detectan estos problemas automáticamente —antes de que causen daño. Un linter encuentra variables sin usar, errores de tipo y malas prácticas al instante.

---

## 2. Linting

Los linters analizan el código en busca de errores potenciales, violaciones de estilo y antipatrones sin ejecutarlo.

### 2.1 Ruff

Ruff es el linter moderno de Python (reemplaza a flake8, isort, pyupgrade y otros en una sola herramienta).

```bash
ruff check .                    # Revisar todos los archivos
ruff check --fix .             # Corregir problemas automáticamente
ruff format .                  # Formatear código
```

```toml
# pyproject.toml
[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP", "B", "SIM"]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]  # imports sin uso permitidos en init
```

**Qué detecta**:
- **E**: pycodestyle (violaciones de PEP 8)
- **F**: pyflakes (variables no definidas, imports sin usar)
- **I**: isort (orden de imports)
- **N**: convenciones de nombres
- **UP**: pyupgrade (sintaxis moderna de Python)
- **B**: flake8-bugbear (errores comunes)

### 2.2 Pre-commit Hooks

Ejecuta controles de calidad automáticamente antes de cada commit:

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
pre-commit install              # Instalar hooks
pre-commit run --all-files      # Ejecutar en todos los archivos
```

Si un hook falla, el commit se bloquea hasta que el problema se solucione.

---

## 3. Type Hints

Los type hints documentan qué tipos espera y devuelve una función. No afectan el comportamiento en tiempo de ejecución, pero permiten análisis estático.

```python
from typing import Optional, List, Tuple, Dict

def preprocess(
    df: pd.DataFrame,
    columns: List[str],
    drop_na: bool = True,
    fill_value: Optional[float] = None,
) -> pd.DataFrame:
    """Preprocesa un DataFrame."""
    if drop_na:
        df = df.dropna()
    if fill_value is not None:
        df = df.fillna(fill_value)
    return df
```

**Beneficios**:
- **Documentación**: interfaz clara sin leer la implementación
- **Detección de errores**: mypy detecta tipos de argumentos incorrectos
- **Soporte del IDE**: autocompletado y documentación en línea
- **Refactorización**: cambiar un tipo revela todos los puntos de uso

### myPy

```bash
mypy src/  # revisa todos los archivos en src/
```

Ejemplo de detección:
```python
def get_age(person: dict) -> int:
    return person["age"]

# mypy: error: Returning Any from function declared to return "int"
# (porque person no está tipado)
```

**Solución**:
```python
from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int

def get_age(person: Person) -> int:
    return person["age"]  # mypy sabe que es int
```

---

## 4. Formateo

Los formateadores corrigen el estilo del código automáticamente (indentación, comillas, longitud de línea, espaciado).

| Herramienta | Estilo | ¿Opinado? |
|---|---|---|
| **Black** | PEP 8 con modificaciones | Muy (pocas opciones) |
| **Ruff format** | Compatible con Black | Igual que Black |
| **Autopep8** | PEP 8 puro | Menos opinado |

**¿Por qué usar un formateador?** Elimina todos los debates de estilo. El equipo acuerda: el formateador decide.

```bash
black src/          # Formatear todos los archivos
ruff format src/    # Igual que Black, pero más rápido
```

---

## 5. Documentación

### 5.1 Docstrings

```python
def train_model(
    X: np.ndarray,
    y: np.ndarray,
    params: Optional[dict] = None,
) -> RandomForestClassifier:
    """Entrena un clasificador Random Forest.

    Args:
        X: Características de entrenamiento. Forma (n_muestras, n_características).
        y: Etiquetas de entrenamiento. Forma (n_muestras,).
        params: Hiperparámetros opcionales. Por defecto
            {'n_estimators': 100, 'max_depth': 5}.

    Returns:
        RandomForestClassifier entrenado.

    Raises:
        ValueError: Si X y y tienen longitudes distintas.
    """
```

### 5.2 README

Todo proyecto debería tener un README que responda:
- ¿Qué hace este proyecto?
- ¿Cómo se configura? (consulta [[Virtual Environments]])
- ¿Cómo se ejecuta?

---

## 6. Lista de Verificación para Code Review

- [ ] ¿El código hace lo que dice?
- [ ] ¿Hay casos borde no manejados?
- [ ] ¿Hay tests para las funciones nuevas?
- [ ] ¿Los type hints son correctos y completos?
- [ ] ¿Los nombres de variables y funciones son descriptivos?
- [ ] ¿No hay valores hardcodeados (usar constantes)?
- [ ] ¿Los parámetros de experimentos se registran mediante [[Experiment Tracking]]?
- [ ] ¿La lógica es fácil de seguir?

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

## 9. Resumen

Code quality tools catch bugs before they reach production. Ruff lints and formats. myPy enforces type hints. Pre-commit runs checks automatically before every commit. Good documentation (docstrings, README) makes code maintainable. The investment in code quality pays for itself in fewer bugs, faster development, and happier teammates.

---

## 10. Where to Go Next

- [[Testing for Data Science]] — Complementing static analysis with tests
- [[Git]] — Version control workflow
- [[CLI & Productivity]] — Running quality checks from the CLI
