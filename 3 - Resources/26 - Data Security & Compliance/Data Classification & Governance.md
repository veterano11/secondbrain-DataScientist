---
tags:
  - data-security
  - data-governance
  - data-classification
  - pii
  - data-catalog
  - rbac
  - compliance
status: seedling
created: 2026-06-28
---

## 1. Escenario de aprendizaje

Tu empresa tiene 500 tablas en el data warehouse. Nadie sabe cuáles contienen PII (nombres, emails, RUTs). Un analista expone datos sensibles en un dashboard público porque no había controles. La gerencia quiere implementar un programa de data classification y governance para que esto no vuelva a ocurrir.

## 2. Requisitos previos

- SQL básico (consultas `SELECT`, `JOIN`, `WHERE`)
- Python: scripting, regex, pandas
- Familiaridad con data warehouses (Snowflake, Redshift, BigQuery)
- Conceptos generales de seguridad (usuarios, roles, permisos)

## 3. Data classification

Clasificar datos significa etiquetarlos según su nivel de sensibilidad. Es el primer paso para saber qué proteger y cómo.

### 3.1 Niveles comunes de clasificación

| Nivel | Descripción | Ejemplo |
|-------|-------------|---------|
| Pública | Sin riesgo si se expone | Precios de productos, nombres de campañas |
| Interna | Daño menor si se filtra | Organigrama, KPIs internos |
| Confidencial | Daño significativo | Datos de clientes, finanzas |
| Restringida | Daño crítico/legal | RUT, salud, tarjetas de crédito |

### 3.2 Implementación con metadatos

```sql
-- Agregar tags de clasificación en Snowflake
ALTER TABLE customers SET TAG classification = 'restricted';
ALTER TABLE products   SET TAG classification = 'public';
ALTER TABLE employees  SET TAG classification = 'confidential';

-- Consultar clasificación de tablas
SELECT table_name, TAG_VALUE('classification', table_name) AS level
FROM information_schema.tables
WHERE table_schema = 'PUBLIC';
```

**Salida esperada:**
```
TABLE_NAME    | LEVEL
--------------+-------------
customers     | restricted
products      | public
employees     | confidential
```

## 4. PII detection

Detectar PII (Personally Identifiable Information) es un problema de [[Data Quality & Testing]] porque requiere identificar patrones en datos no estructurados.

### 4.1 Detección con regex

```python
import re
import pandas as pd

def detect_pii_column(series: pd.Series) -> list[str]:
    """
    Analiza una columna y devuelve los tipos de PII detectados.
    """
    patterns = {
        "email": r"[\w\.-]+@[\w\.-]+\.\w+",
        "phone": r"\+?\d{1,3}[\s-]?\d{3,4}[\s-]?\d{4}",
        "rut":   r"^\d{1,2}\.\d{3}\.\d{3}[-][0-9kK]$",
        "credit_card": r"\b(?:\d{4}[-\s]?){3,4}\d{4}\b",
    }

    sample = series.dropna().astype(str).head(100)
    detected = []
    for pii_name, pattern in patterns.items():
        matches = sample.str.contains(pattern, regex=True, na=False)
        if matches.sum() >= 5:  # umbral: 5% de la muestra
            detected.append(pii_name)
    return detected

# Ejemplo de uso
df = pd.DataFrame({
    "nombre": ["Juan Pérez", "María García"],
    "email":   ["juan@example.com", "maria@corp.cl"],
})
for col in df.columns:
    print(f"{col}: {detect_pii_column(df[col])}")
```

**Salida esperada:**
```
nombre: []
email: ['email']
```

### 4.2 Heurísticas + ML

Las heurísticas complementan al regex:

- **Columna "email"** con formato conocido → email
- **Columna "salario"** con valores numéricos > 0 → posible dato financiero
- **Cardinalidad**: si una columna tiene valores únicos por fila, podría ser identificador

Para casos complejos se usan modelos como [[ML Governance & Regulation]] sugiere: transformers especializados (Presidio, Microsoft PIICon) que clasifican texto libre.

```python
from presidio_analyzer import AnalyzerEngine

analyzer = AnalyzerEngine()
results = analyzer.analyze(text="Mi RUT es 12.345.678-9 y mi email es juan@mail.com",
                           language="es")
for r in results:
    print(f"Tipo: {r.entity_type}, Score: {r.score}, Inicio: {r.start}, Fin: {r.end}")
```

**Salida esperada:**
```
Tipo: RUT_CL, Score: 0.85, Inicio: 8, Fin: 20
Tipo: EMAIL, Score: 1.0, Inicio: 37, Fin: 51
```

## 5. Data catalog

Un [[Data Pipelines & ETL]] no debería operar sin catálogo. El catalog es el sistema de registro para todo el inventario de datos.

### 5.1 Componentes

| Componente | Descripción |
|------------|-------------|
| Metadatos técnicos | Schema, tipos, particiones, estadísticas |
| Lineage | Origen y transformaciones de cada columna |
| Ownership | Quién es responsable del dataset |
| Business glossary | Definiciones de negocio (ej: "cliente activo") |
| Classification tags | Nivel de sensibilidad (sección 3) |

### 5.2 Implementación con dbt + Datahub

```yaml
# schema.yml en dbt
version: 2

models:
  - name: customers
    description: "Tabla maestra de clientes"
    config:
      tags: ["restricted", "pii"]
    columns:
      - name: email
        description: "Email del cliente"
        data_type: string
        tests:
          - not_null
        meta:
          classification: restricted
          pii_type: email
      - name: rut
        description: "RUT chileno"
        meta:
          classification: restricted
          pii_type: rut
```

```bash
# Publicar metadatos a Datahub
datahub ingest -c dbt_to_datahub.yml
```

**Salida esperada:**
```
[2026-06-28 10:00:01] ✅  Successfully ingested 45 entities to Datahub
[2026-06-28 10:00:02] 🔗  Lineage: customers → daily_revenue → dashboard_revenue
```

## 6. Data contracts

Un [[dbt & Data Transformation]] data contract es un acuerdo formal entre el productor y el consumidor de datos. Garantiza schema, calidad y clasificación.

```python
# contract.yaml
dataset: customers
version: 1.0
owner: equipo-datos@empresa.cl
classification: restricted

schema:
  email:
    type: string
    nullable: false
    pii: true
    format: email
  rut:
    type: string
    nullable: false
    pii: true
    format: rut_cl

sla:
  freshness: 1h
  completeness: 0.99

breach_actions:
  - notify_owner
  - block_downstream
```

```bash
# Validar contract en pipeline CI/CD
datacontract validate contract.yaml --server snowflake
```

**Salida esperada:**
```
✅ Schema validation passed (10 fields)
✅ Freshness: 0 violations in last 24h
⚠️ Completeness: 0.97 < 0.99 SLA → notified owner
```

## 7. Access control policies

Una vez clasificado, aplicas [[Encryption & Access Control]] con políticas RBAC.

```sql
-- RBAC en Snowflake basado en clasificación
CREATE ROLE data_analyst;
CREATE ROLE data_scientist;
CREATE ROLE data_steward;

-- Los analistas solo ven datos públicos e internos
GRANT SELECT ON ALL TABLES IN SCHEMA public TO ROLE data_analyst;
GRANT SELECT ON ALL TABLES CLASSIFIED AS 'internal' TO ROLE data_analyst;

-- Los científicos ven confidencial pero no restringido
GRANT SELECT ON ALL TABLES CLASSIFIED AS 'confidential' TO ROLE data_scientist;

-- El data steward accede a todo
GRANT SELECT ON ALL TABLES CLASSIFIED AS 'restricted' TO ROLE data_steward;
```

```bash
# Flujo de aprobación con OpenMetadata
curl -X POST https://openmetadata.empresa.cl/api/v1/policies \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "restricted_access",
    "rules": [
      {
        "resources": ["table:classification=restricted"],
        "roles": ["data_steward"],
        "condition": "request.access_reason == \'business_need\' AND approver.reviewed == true"
      }
    ]
  }'
```

## 8. Common Mistakes

1. **Clasificar todo como "confidencial"**: Si todo es urgente, nada lo es. Los analistas ignoran las etiquetas y encuentran workarounds peligrosos. Sé específico.
2. **Governance sin automation**: Llenar Excel con clasificaciones que nadie actualiza. La gobernanza debe estar embebida en el pipeline (CI/CD checks, tags automáticos en el DW).
3. **Solo mirar columnas obvias**: El campo `notes` de un CRM puede contener PII en texto libre aunque la columna se llame "comentarios". Escanea contenido, no solo nombres.
4. **Ownership sin accountability**: Poner un "owner" en el catálogo pero sin que esa persona sepa que es responsable. El ownership debe tener revisión trimestral.
5. **Ignorar datos en reposo no estructurado**: Archivos CSV en S3, logs de aplicación, correos. El data warehouse es solo una parte del problema.

## Resumen

Data classification y governance son la base de cualquier programa de seguridad de datos. Clasificar (pública, interna, confidencial, restringida) permite aplicar políticas de acceso proporcionadas. La detección de PII combina regex, heurísticas y ML para identificar datos sensibles automáticamente. Un data catalog con metadatos, lineage y ownership centraliza el inventario, mientras que los data contracts garantizan acuerdos formales entre productores y consumidores. Sin automation, la gobernanza muere. Sin clasificación precisa, los controles no funcionan.

## Check Your Understanding

1. ¿Cuál es la diferencia entre un dato "confidencial" y uno "restringido"?
2. ¿Por qué clasificar todo como "confidencial" es contraproducente?
3. Menciona dos técnicas para detectar PII en columnas de texto.
4. ¿Qué componente del data catalog responde a "¿de dónde viene este dato?"?
5. ¿Qué es un data contract y qué problema resuelve?

<!--
1. Confidencial: daño significativo si se filtra (clientes, finanzas). Restringido: daño crítico/legal (RUT, salud, tarjetas). La diferencia es el nivel de impacto.
2. Si todo es confidencial, los equipos ignoran las etiquetas por fatiga de alertas. Además, no se pueden priorizar controles. Se pierde granularidad para RBAC.
3. Regex para patrones conocidos (emails, RUT) y modelos de ML (Presidio, transformers) para contexto. También heurísticas como nombre de columna + cardinalidad.
4. Lineage: rastrea el origen y las transformaciones de cada columna a través del pipeline.
5. Un data contract es un acuerdo formal entre productor y consumidor que garantiza schema, calidad y clasificación del dataset. Resuelve la falta de comunicación y los breaking changes silenciosos.
-->

## Where to Go Next

- [[Data Quality & Testing]]
- [[Encryption & Access Control]]
- [[Privacy Regulations]]
- [[Data Warehousing & Lakehouse]]
- [[ML Governance & Regulation]]
- [[dbt & Data Transformation]]
- [[Data Pipelines & ETL]]
