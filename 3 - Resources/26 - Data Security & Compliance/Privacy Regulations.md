---
tags:
  - gdpr
  - ccpa
  - privacy
  - data-protection
  - eu-ai-act
  - d sar
  - right-to-explanation
  - compliance
status: seedling
created: 2026-06-28
---

## 1. Escenario de aprendizaje

Un usuario escribe: "Por favor borren todos mis datos. Ya no quiero que usen mi información." La empresa opera en Europa y California. GDPR Art. 17 (right to erasure) y CCPA exigen que puedas encontrar, exportar y eliminar datos personales en 30 días. Si no puedes demostrar que lo hiciste, las multas alcanzan el 4% de los ingresos globales anuales.

## 2. Requisitos previos

- SQL: consultas sobre múltiples bases de datos
- Data catalog: saber qué sistemas contienen datos de personas
- Familiaridad con data pipelines y [[Data Pipelines & ETL]]
- Conocimiento básico de [[Data Classification & Governance]]

## 3. GDPR

El Reglamento General de Protección de Datos (GDPR) es la regulación europea que protege los datos personales de los ciudadanos de la UE.

### 3.1 Principios fundamentales

| Principio | Significado |
|-----------|-------------|
| Licitud, lealtad, transparencia | Procesar datos con base legal clara |
| Limitación de la finalidad | Solo usar los datos para el propósito informado |
| Minimización de datos | Recoger solo lo estrictamente necesario |
| Exactitud | Datos correctos y actualizados |
| Limitación del plazo | No guardar datos más tiempo del necesario |
| Integridad y confidencialidad | Seguridad técnica y organizativa |
| Responsabilidad proactiva | Demostrar cumplimiento |

### 3.2 Lawful basis

Debes tener una base legal para procesar datos: consentimiento, contrato, obligación legal, interés vital, interés público, o interés legítimo.

### 3.3 Data Subject Rights

```sql
-- Encontrar todos los datos de un usuario en el data warehouse
SELECT 'customers' AS source, customer_id, email, name, created_at
FROM customers WHERE email = 'usuario@example.com'
UNION ALL
SELECT 'orders', user_id, email, product_name, order_date
FROM orders WHERE email = 'usuario@example.com'
UNION ALL
SELECT 'logs', user_id, email, action, timestamp
FROM access_logs WHERE email = 'usuario@example.com';
```

**Salida esperada:**
```
source    | customer_id | email               | name        | created_at
----------+-------------+---------------------+-------------+--------------------
customers | 45231       | usuario@example.com | Juan Pérez  | 2024-03-15 10:30:00
orders    | 45231       | usuario@example.com | Laptop Pro  | 2025-07-22 14:00:00
logs      | 45231       | usuario@example.com | LOGIN       | 2026-06-27 09:15:00
```

## 4. CCPA/CPRA

La California Consumer Privacy Act (CCPA) y su enmienda CPRA dan derechos similares al GDPR pero con diferencias importantes.

### 4.1 Diferencias clave

| Aspecto | GDPR | CCPA/CPRA |
|---------|------|-----------|
| Ámbito | Ciudadanos UE | Residentes de California |
| Base legal requerida | Sí (6 bases) | No (opt-out) |
| Derecho a eliminar | Sí (Art. 17) | Sí, con excepciones |
| Multa máxima | 4% revenue global o €20M | $7,500 por violación intencional |
| Derecho a saber | Sí | Sí, + categorías de fuentes |
| Opt-out de venta | No explícito | Sí (venta de datos) |
| Sensitive data | Categoría especial | Categoría separada con más protecciones |

```bash
# Simular un DSAR request desde API
curl -X POST https://privacy-api.empresa.cl/v1/dsar \
  -H "Authorization: Bearer $PRIVACY_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "user_email": "usuario@example.com",
    "request_type": "access",
    "jurisdiction": "gdpr"
  }' | jq '.'
```

**Salida esperada:**
```json
{
  "request_id": "DSAR-2026-06-28-0042",
  "status": "in_progress",
  "systems_searched": 12,
  "records_found": 145,
  "estimated_completion": "2026-07-12T00:00:00Z"
}
```

## 5. EU AI Act

La [[ML Governance & Regulation]] incluye el EU AI Act, que clasifica sistemas de IA por nivel de riesgo.

### 5.1 Categorías de riesgo

| Categoría | Ejemplos | Requisitos |
|-----------|----------|------------|
| Prohibido | Social scoring, manipulación conductual | Prohibido completamente |
| Alto riesgo | Crédito, salud, contratación, policía | Evaluación de conformidad, documentación, supervisión humana |
| Riesgo limitado | Chatbots, deepfakes | Transparencia (informar que es IA) |
| Riesgo mínimo | Spam filters, juegos | Sin obligaciones |

### 5.2 Transparencia para modelos

```python
def classify_ai_risk(model_card: dict) -> str:
    """
    Determina la categoría de riesgo de un modelo según EU AI Act.
    """
    high_risk_sectors = {"credit", "healthcare", "hiring", "law_enforcement"}
    domain = model_card.get("domain", "")

    if "manipulation" in model_card.get("techniques", []):
        return "prohibited"
    if domain in high_risk_sectors:
        return "high_risk"
    if model_card.get("interacts_with_humans"):
        return "limited_risk"
    return "minimal_risk"

card = {
    "domain": "healthcare",
    "techniques": ["classification"],
    "interacts_with_humans": False,
}
print(classify_ai_risk(card))
```

**Salida esperada:**
```
high_risk
```

## 6. Data Subject Access Request (DSAR)

El DSAR es el proceso para encontrar todos los datos de una persona. [[Secure ML & Incident Response]] comparte principios de detección y respuesta que aplican aquí.

### 6.1 Identificación de sistemas

```yaml
# systems-registry.yaml
systems:
  - name: postgres_customers
    type: database
    pii_fields: [email, name, address, phone]
    retention_days: 2555  # 7 años
    deletion_method: hard_delete
  - name: mongo_logs
    type: database
    pii_fields: [email, ip_address]
    retention_days: 365
    deletion_method: anonymize
  - name: s3_data_lake
    type: object_storage
    pii_fields: [any]  # datos no estructurados
    retention_days: 730
    deletion_method: lifecycle_policy
  - name: redshift_analytics
    type: data_warehouse
    pii_fields: [email, customer_id]
    retention_days: 1095
    deletion_method: delete + vacuum
```

### 6.2 Automatización del DSAR

```python
import yaml
import psycopg2

def execute_dsar(user_email: str, systems_file: str) -> dict:
    """
    Busca todos los registros de un usuario en todos los sistemas.
    """
    with open(systems_file) as f:
        registry = yaml.safe_load(f)

    results = {}
    for system in registry["systems"]:
        conn = psycopg2.connect(f"dbname={system['name']}")
        for field in system["pii_fields"]:
            if field == "any":
                continue
            query = f"SELECT * FROM information_schema.tables WHERE table_schema = 'public'"
            # lógica de búsqueda por sistema
            cur = conn.cursor()
            cur.execute(f"SELECT * FROM all_tables WHERE {field} = %s", (user_email,))
            rows = cur.fetchall()
            if rows:
                results[system["name"]] = rows
    return results

# dsar_result = execute_dsar("usuario@example.com", "systems-registry.yaml")
```

## 7. Right to explanation

Cuando un modelo de ML toma una decisión automatizada (rechazar un crédito, diagnosticar una enfermedad), el usuario tiene derecho a saber por qué. [[Interpretability (SHAP-LIME)]] proporciona las herramientas para implementarlo.

```python
import shap
import xgboost as xgb

def explain_decision(model, instance: dict) -> dict:
    """
    Genera una explicación SHAP para cumplir con el right to explanation.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(instance)
    feature_importance = dict(
        zip(instance.columns, shap_values[0])
    )
    # Ordenar por impacto absoluto
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    return {
        "decision": model.predict(instance)[0],
        "top_factors": [
            {"feature": f, "impact": round(v, 4)}
            for f, v in sorted_features[:5]
        ],
        "model_version": model.get_xgb_params().get("n_estimators"),
    }

# response = explain_decision(credit_model, applicant_data)
```

**Salida esperada:**
```json
{
  "decision": "rejected",
  "top_factors": [
    {"feature": "income", "impact": 0.452},
    {"feature": "debt_ratio", "impact": -0.321},
    {"feature": "credit_history", "impact": 0.189}
  ],
  "model_version": 200
}
```

## 8. Common Mistakes

1. **Asumir que GDPR no aplica a tu empresa**: Si tienes un solo usuario en la UE o California, te aplica. No importa dónde esté tu empresa. El GDPR tiene alcance extraterritorial.
2. **No tener proceso documentado**: Decir "borramos los datos cuando nos piden" sin un proceso reproducible. Necesitas un DSAR playbook con SLA de 30 días, aprobado y auditado.
3. **Borrar datos sin backup**: Eliminas al usuario de producción, pero los datos siguen en backups, data lake, logs, y cachés. La eliminación debe cubrir todos los sistemas, incluyendo snapshots y archivos muertos.
4. **Ignorar datos no estructurados**: El nombre del usuario en un email, en un PDF adjunto, en un chat de soporte. El DSAR no está completo hasta que buscas en todos los sistemas.
5. **No distinguir entre eliminación y anonimización**: A veces no puedes eliminar (obligación legal de retención). En ese caso, debes anonimizar irreversiblemente para que los datos no sean atribuibles.
6. **Confundir "consentimiento" con "interés legítimo"**: Si usas interés legítimo como base, el usuario puede objetar y debes demostrar que tu interés prevalece. No es un comodín.

## Resumen

GDPR y CCPA/CPRA exigen que las empresas puedan encontrar, exportar y eliminar datos personales en plazos definidos (30 días generalmente). El GDPR se basa en siete principios y requiere una base legal para procesar datos. El EU AI Act añade una capa de regulación para sistemas de IA según su nivel de riesgo. Un DSAR efectivo requiere un inventario completo de sistemas, un proceso automatizado de búsqueda, y cobertura de datos estructurados y no estructurados. El right to explanation demanda que las decisiones automatizadas sean interpretables con herramientas como SHAP.

## Check Your Understanding

1. ¿Cuál es el plazo máximo para responder a un DSAR según GDPR?
2. Diferencia entre "right to erasure" (GDPR Art. 17) y "opt-out" (CCPA).
3. ¿Qué categoría de riesgo del EU AI Act aplica a un sistema de aprobación de créditos?
4. ¿Por qué es importante buscar en datos no estructurados durante un DSAR?
5. ¿Qué diferencia hay entre eliminar datos y anonimizarlos?

<!--
1. 30 días, prorrogables 60 días adicionales en casos complejos (notificando al usuario). El CCPA da 45 días.
2. Right to erasure: el usuario puede solicitar que borres sus datos si ya no hay base legal. Opt-out (CCPA): el usuario puede impedir que vendas sus datos, pero no necesariamente que los proceses para otros fines.
3. Alto riesgo (high-risk). Los sistemas de crédito están explícitamente listados como high-risk en el EU AI Act y requieren evaluación de conformidad.
4. Porque los datos personales pueden estar en emails, PDFs, logs de chats, tickets de soporte, etc. Si no buscas ahí, el DSAR está incompleto y podrías estar incumpliendo la regulación.
5. Eliminar = borrar irreversiblemente el registro. Anonimizar = transformar los datos para que no puedan asociarse a una persona (irreversible). La anonimización se usa cuando hay obligación legal de retener los datos pero no de mantenerlos identificables.
-->

## Where to Go Next

- [[ML Governance & Regulation]]
- [[Data Classification & Governance]]
- [[Secure ML & Incident Response]]
- [[Interpretability (SHAP-LIME)]]
- [[Bias & Fairness]]
- [[Data Quality & Testing]]
- [[Experiment Tracking]]
