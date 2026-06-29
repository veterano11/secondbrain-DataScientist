---
tags:
  - software-engineering
  - testing
  - data-quality
  - pytest
status: seedling
created: 2026-06-28
---

# Testing Strategies for Data

## 1. Escenario de aprendizaje

Modificaste una función de limpieza de datos para manejar un nuevo formato de fecha. Los tests unitarios pasaron. El pipeline completo se ejecutó sin errores. Pero dos semanas después, el equipo de producto reporta que las ventas del último mes cayeron 15%. Investigando, descubres que tu cambio silenciosamente eliminó filas donde la fecha era `NaT` — filas que antes se conservaban con una imputación por defecto.

Ningún test detectó el problema porque los tests solo verificaban que la función no lanzara excepciones, no que produjera el resultado correcto. El bug estuvo en producción 14 días. El costo: decisiones de negocio basadas en datos incorrectos.

Testear código de datos es diferente de testear una aplicación web. Los datos cambian, los esquemas evolucionan, y un pipeline que funciona hoy puede fallar mañana porque los datos de origen tienen una distribución diferente. Este note cubre las estrategias específicas para testing en el contexto de data science e ingeniería de datos.

## 2. Requisitos

- Python 3.9+
- pytest
- pandas
- Experiencia con pipelines de datos (ETL, transformaciones)
- Deseable: conocimiento de schemas de datos

## 3. Unit tests para funciones de transformación

Las funciones de transformación son el lugar ideal para unit tests: reciben un input conocido y deben producir un output esperado.

```python
import pandas as pd
import pytest

# Código a testear
def impute_missing_age(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    median_age = result['age'].median()
    result['age'].fillna(median_age, inplace=True)
    return result

# Unit test
def test_impute_missing_age_replaces_nan_with_median():
    data = pd.DataFrame({'age': [25, None, 35, None, 40]})
    # mediana de [25, 35, 40] = 35 (None se excluye)
    result = impute_missing_age(data)
    assert result['age'].iloc[1] == 35
    assert result['age'].iloc[3] == 35
    assert result['age'].isna().sum() == 0

def test_impute_missing_age_preserves_existing_values():
    data = pd.DataFrame({'age': [25, 30, 35]})
    result = impute_missing_age(data)
    assert result['age'].tolist() == [25, 30, 35]
```

```python
# Salida esperada:
# pytest test_transforms.py -v
# test_transforms.py::test_impute_missing_age_replaces_nan_with_median PASSED
# test_transforms.py::test_impute_missing_age_preserves_existing_values PASSED
```

## 4. Integration tests para pipelines

Los unit tests verifican componentes individuales. Los integration tests verifican que el pipeline completo funciona correctamente de principio a fin.

```python
# Pipeline completo
def run_pipeline(raw_path: str, output_path: str):
    data = pd.read_csv(raw_path)
    data = clean_data(data)
    data = engineer_features(data)
    data.to_parquet(output_path)
    return data

# Integration test con datos sintéticos
@pytest.fixture
def sample_raw_data(tmp_path):
    data = pd.DataFrame({
        'user_id': [1, 2, 3],
        'age': [25, None, 35],
        'amount': [100.0, 200.0, 300.0],
        'date': ['2024-01-01', '2024-01-02', None]
    })
    input_path = tmp_path / 'raw_data.csv'
    data.to_csv(input_path, index=False)
    return input_path, tmp_path

def test_pipeline_output_schema(sample_raw_data):
    input_path, tmp_path = sample_raw_data
    output_path = tmp_path / 'processed.parquet'

    result = run_pipeline(input_path, output_path)

    expected_columns = {'user_id', 'age', 'amount', 'date', 'age_group', 'log_amount'}
    assert set(result.columns) == expected_columns
    assert len(result) == 3
    assert output_path.exists()
```

```python
# Salida esperada:
# pytest test_pipeline.py -v
# test_pipeline.py::test_pipeline_output_schema PASSED
# Los integration tests son más lentos que unit tests,
# pero detectan bugs que surgen de la interacción entre
# componentes.
```

## 5. Data quality tests con pandera

Pandera permite definir schemas para DataFrames y validar que los datos cumplen con restricciones.

```python
import pandera as pa
from pandera.typing import DataFrame

class CustomerSchema(pa.DataFrameModel):
    customer_id: int = pa.Field(ge=0)
    age: int = pa.Field(ge=18, le=120, nullable=True)
    income: float = pa.Field(ge=0.0)
    signup_date: datetime = pa.Field()
    is_active: bool = pa.Field()

@pa.check_types
def process_customers(data: DataFrame[CustomerSchema]) -> DataFrame[CustomerSchema]:
    return data

# Si los datos no cumplen el schema, pandera lanza SchemaError
# con un mensaje claro indicando qué filas y columnas fallaron.
```

```python
# Salida esperada:
# SchemaError: <Schema Column 'age'> expected series 'age'
# to be greater than or equal to 18, but encountered values:
# [15, 12]
```

Pandera se integra con pytest: puedes escribir tests que verifiquen que el schema se cumple, y ejecutarlos como parte de tu suite de tests.

## 6. Golden datasets para regression testing

Un golden dataset es un conjunto de datos de referencia con outputs esperados conocidos. Cada vez que modificas el pipeline, ejecutas el pipeline contra el golden dataset y comparas los outputs.

```python
import pandas as pd
import pytest
from pathlib import Path

GOLDEN_DATA_DIR = Path('tests/golden_datasets')

def test_pipeline_against_golden_dataset():
    # Cargar golden input
    golden_input = pd.read_parquet(GOLDEN_DATA_DIR / 'input.parquet')
    golden_output = pd.read_parquet(GOLDEN_DATA_DIR / 'output.parquet')

    # Ejecutar pipeline actual
    current_output = run_pipeline(golden_input)

    # Comparar
    pd.testing.assert_frame_equal(current_output, golden_output)
```

```python
# Salida esperada:
# Si el pipeline cambia intencionalmente el output (ej: nueva
# feature), actualizas el golden dataset. Si cambia sin
# intención, el test falla y alerta antes de llegar a producción.
```

**Cuándo actualizar el golden dataset:**
- Cuando el cambio en el output es intencional y verificado
- Cuando agregas nuevas features (el dataset debe incluir las columnas nuevas)
- Cuando cambian las reglas de negocio

## 7. Property-based testing con Hypothesis

Los tests tradicionales verifican casos específicos. Property-based testing genera automáticamente múltiples casos de entrada y verifica invariantes.

```python
from hypothesis import given, strategies as st
import pandas as pd

def sort_by_date(events: pd.DataFrame) -> pd.DataFrame:
    return events.sort_values('timestamp').reset_index(drop=True)

@given(st.lists(
    st.datetimes(),
    min_size=0,
    max_size=50
))
def test_sort_by_date_is_monotonic(timestamps):
    data = pd.DataFrame({'timestamp': timestamps})
    result = sort_by_date(data)
    # Invariante: los timestamps deben estar ordenados
    assert result['timestamp'].is_monotonic_increasing

@given(st.lists(
    st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=1_000_000),
    min_size=1,
    max_size=100
))
def test_normalize_keeps_values_in_range(values):
    series = pd.Series(values)
    normalized = (series - series.min()) / (series.max() - series.min())
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0
```

```python
# Salida esperada:
# Hypothesis ejecuta cientos de ejemplos automáticos.
# Si encuentra un caso donde falla, muestra el input
# exacto que reproduce el error y lo guarda como
# "falsifying example" para debugging.
```

Property-based testing es especialmente útil para detectar edge cases que no anticipaste: listas vacías, todos los valores iguales, valores extremos.

## 8. Common Mistakes

**Testear sobre datos reales en lugar de datos controlados:** Los datos reales cambian, tienen distribuciones variables y pueden contener sorpresas. Un test que usa datos reales no es determinista. Usa datos sintéticos o fixtures.

**No testear edge cases:** Los pipelines de datos encuentran más nulls, DataFrames vacíos, tipos mixtos y outliers de los que imaginas. Cada función debe testearse con:
- DataFrame vacío
- Todas las filas con nulls en una columna
- Tipos de datos incorrectos
- Valores extremos (outliers)

**Tests que solo verifican que no hay errores:** Un test que llama a una función y no verifica el resultado es un "test óptico" — te hace sentir que testeas pero no atrapa bugs. Siempre verifica el output.

**Ignorar la determinismo de los tests:** Si tu pipeline usa random forest, la semilla aleatoria debe fijarse en los tests. Si usas train_test_split sin random_state, los tests fallarán intermitentemente.

```python
# Mal: test no determinista
def test_split_data():
    data = pd.DataFrame({'value': range(100)})
    train, test = split_data(data, test_size=0.2)
    assert len(train) == 80  # Puede fallar si split es aleatorio

# Bien: semilla fija
def test_split_data():
    data = pd.DataFrame({'value': range(100)})
    train, test = split_data(data, test_size=0.2, random_state=42)
    assert len(train) == 80  # Determinista
```

```python
# Salida esperada (Common Mistakes):
# Datos reales en tests = tests que fallan aleatoriamente.
# Sin tests de edge cases = bugs que aparecen en producción.
# Tests sin assertions = falsa sensación de seguridad.
```

## Resumen

Testear código de datos requiere un enfoque específico: unit tests para transformaciones individuales, integration tests para pipelines completos, data quality tests con pandera para validar schemas, golden datasets para regression testing, y property-based testing para descubrir edge cases automáticamente. La clave es usar datos controlados y deterministas, testear edge cases explícitamente, y verificar outputs, no solo la ausencia de errores.

## Check Your Understanding

1. ¿Cuál es la diferencia fundamental entre unit tests e integration tests para pipelines de datos?
<!-- Unit tests verifican componentes individuales con datos controlados. Integration tests verifican que el pipeline completo funciona de extremo a extremo. Los unit tests son rápidos y aíslan el fallo; los integration tests detectan problemas de interacción entre componentes. -->

2. ¿Por qué es problemático usar datos reales en tests?
<!-- Los datos reales cambian con el tiempo. Un test que pasa hoy puede fallar mañana porque los datos de origen cambiaron de distribución. Los tests deben ser deterministas, y eso requiere datos controlados (sintéticos o fixtures). -->

3. ¿Qué tipo de bug detecta un golden dataset que un unit test no detectaría?
<!-- Un golden dataset detecta cambios no intencionales en el output del pipeline completo. Un unit test pasa si la función individual funciona, pero no detecta que el pipeline ahora produce resultados diferentes. -->

4. ¿Cuándo usarías property-based testing en lugar de tests con ejemplos fijos?
<!-- Cuando quieres descubrir edge cases que no anticipaste. Hypothesis genera cientos de entradas aleatorias y verifica invariantes. Es ideal para funciones donde hay muchas combinaciones posibles de entrada. -->

5. Tu pipeline tiene un paso de random forest. ¿Qué debes hacer para que el integration test sea determinista?
<!-- Fijar random_state en todos los componentes aleatorios (RandomForestClassifier, train_test_split, etc.) y documentar que los tests dependen de esa semilla. -->

## Where to Go Next

Profundiza en [[Testing for Data Science]] para un tratamiento más extenso de estrategias de testing en el ciclo de vida de datos. Aplica [[Clean Code & Refactoring]] para que tu código sea más testeable desde el diseño. [[Data Quality & Testing]] cubre herramientas de validación continua de datos en producción. Si trabajas con [[Feature Engineering]], esta guía te ayudará a testear transformaciones correctamente. Para pipelines más complejos, [[Data Pipelines & ETL]] muestra cómo integrar tests en orquestadores como Airflow. Revisa [[Code Quality]] para métricas que complementan tu estrategia de testing. Y [[Python for Data Science]] consolida las bases del ecosistema Python para datos.
