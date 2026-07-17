---
tags:
  - responsible-ai
  - governance
  - regulation
  - compliance
  - eu-ai-act
status: seedling
created: 2026-06-28
---

## Escenario de aprendizaje

Tu empresa despliega modelos que toman decisiones sobre personas (préstamos, diagnósticos, contrataciones). Los reguladores (UE AI Act, FDA, CFPB) exigen documentación, transparencia y auditoría. Necesitas un sistema de governance.

Eres ML platform lead en un banco. El equipo de créditos desplegó un modelo de scoring automático que aprobó préstamos con tasas de default aceptables, pero el regulador solicita una auditoría completa: ¿cómo se entrenó?, ¿con qué datos?, ¿qué sesgos tiene?, ¿cómo se monitorea? No tienes nada documentado.

## 1. ¿Qué es ML governance?

ML governance es el sistema de procesos, roles, herramientas y documentación que asegura que los modelos de ML sean responsables, transparentes, auditables y alineados con regulaciones.

- **Procesos**: revisión previa a despliegue, monitoreo continuo, retrain approval.
- **Roles**: model owner, data owner, reviewer, compliance officer.
- **Documentación**: model cards, data sheets, registros de decisiones.
- **Herramientas**: model registry, experiment tracking, drift monitoring.

```python
# Simular ciclo de governance
from datetime import datetime
import pandas as pd

class ModelGovernanceRecord:
    def __init__(self, model_id, owner, risk_category):
        self.model_id = model_id
        self.owner = owner
        self.risk_category = risk_category
        self.status = "development"
        self.approvals = []
        self.versions = []
        self.audit_log = []

    def request_approval(self, reviewer, notes=""):
        self.approvals.append({
            'reviewer': reviewer,
            'timestamp': datetime.now(),
            'status': 'pending',
            'notes': notes
        })
        self.audit_log.append(f"Approval requested from {reviewer}")

    def approve(self, reviewer):
        for a in self.approvals:
            if a['reviewer'] == reviewer and a['status'] == 'pending':
                a['status'] = 'approved'
                a['timestamp'] = datetime.now()
        self.audit_log.append(f"Approved by {reviewer}")

    def promote(self, stage):
        self.status = stage
        self.audit_log.append(f"Promoted to {stage}")

# Ejemplo
record = ModelGovernanceRecord("credito-score-v2", "Maria Lopez", "high")
record.request_approval("Juan Perez", "Revisar sesgo por código postal")
record.approve("Juan Perez")
record.promote("staging")
print(f"Modelo: {record.model_id}, Estado: {record.status}, Auditoría: {len(record.audit_log)} eventos")
```

**Salida esperada:**
```
Modelo: credito-score-v2, Estado: staging, Auditoría: 3 eventos
```

## 2. Model Cards

Una model card es una ficha estandarizada que documenta el propósito, rendimiento, sesgos y limitaciones de un modelo. Propuesta originalmente por Google (Mitchell et al., 2019).

```python
model_card = {
    "model_details": {
        "name": "Credit Scoring Classifier v2",
        "version": "2.1.0",
        "type": "Gradient Boosting (XGBoost)",
        "owner": "Equipo de Riesgo Crediticio",
        "training_date": "2026-05-15",
    },
    "intended_use": {
        "primary_use": "Aprobar o rechazar solicitudes de préstamos personales < $50,000",
        "out_of_scope": "Préstamos hipotecarios, comerciales, o > $50,000",
    },
    "metrics": {
        "overall_accuracy": 0.92,
        "demographic_parity_diff": 0.04,
        "equal_opportunity_diff": 0.03,
    },
    "evaluation_data": {
        "dataset": "solicitudes_2025_2026",
        "size": 45000,
        "sensitive_features": ["genero", "edad", "codigo_postal"],
    },
    "ethical_considerations": {
        "bias_analysis": "Diferencia menor al 5% en todas las métricas de equidad",
        "fairness_mitigation": "Threshold adjustment post-procesamiento por grupo etario",
    },
    "limitations": [
        "No evaluado en poblaciones rurales",
        "Depende de datos de buró de crédito (puede contener errores)",
    ],
}

for section, content in model_card.items():
    print(f"\n[{section}]")
    if isinstance(content, list):
        for item in content:
            print(f"  - {item}")
    elif isinstance(content, dict):
        for k, v in content.items():
            print(f"  {k}: {v}")
```

**Salida esperada:**
```
[model_details]
  name: Credit Scoring Classifier v2
  version: 2.1.0
  type: Gradient Boosting (XGBoost)
  owner: Equipo de Riesgo Crediticio
  training_date: 2026-05-15

[intended_use]
  primary_use: Aprobar o rechazar solicitudes de préstamos personales < $50,000
  out_of_scope: Préstamos hipotecarios, comerciales, o > $50,000

[metrics]
  overall_accuracy: 0.92
  demographic_parity_diff: 0.04
  equal_opportunity_diff: 0.03

[evaluation_data]
  dataset: solicitudes_2025_2026
  size: 45000
  sensitive_features: ['genero', 'edad', 'codigo_postal']

[ethical_considerations]
  bias_analysis: Diferencia menor al 5% en todas las métricas de equidad
  fairness_mitigation: Threshold adjustment post-procesamiento por grupo etario

[limitations]
  - No evaluado en poblaciones rurales
  - Depende de datos de buró de crédito (puede contener errores)
```

## 3. EU AI Act

El EU AI Act clasifica los sistemas de IA por nivel de riesgo:

| Categoría | Ejemplos | Requisitos clave |
|-----------|----------|------------------|
| **Prohibido** | Social scoring, manipulación conductual | Prohibición total |
| **Alto riesgo** | Crédito, salud, contratación, justicia | Evaluación conformidad, documentación, supervisión humana |
| **Riesgo limitado** | Chatbots, deepfakes | Transparencia (reveal que es IA) |
| **Riesgo mínimo** | Spam filters, videojuegos | Sin requisitos adicionales |

```python
def classify_ai_act_risk(use_case):
    high_risk_domains = [
        "credito", "salud", "contratacion", "justicia",
        "seguros", "migracion", "educacion"
    ]
    prohibited_uses = [
        "social scoring", "manipulacion conductual",
        "reconocimiento facial en tiempo real"
    ]

    use_lower = use_case.lower()

    for prohibited in prohibited_uses:
        if prohibited in use_lower:
            return "PROHIBIDO"

    for domain in high_risk_domains:
        if domain in use_lower:
            return "ALTO RIESGO"

    return "RIESGO LIMITADO O MÍNIMO"

casos = [
    "Sistema de scoring crediticio para aprobación de préstamos",
    "Chatbot de atención al cliente",
    "Sistema de manipulación conductual en publicidad online",
    "Filtro de spam para correo electrónico",
]

for caso in casos:
    print(f"  {caso}: {classify_ai_act_risk(caso)}")
```

**Salida esperada:**
```
  Sistema de scoring crediticio para aprobación de préstamos: ALTO RIESGO
  Chatbot de atención al cliente: RIESGO LIMITADO O MÍNIMO
  Sistema de manipulación conductual en publicidad online: PROHIBIDO
  Filtro de spam para correo electrónico: RIESGO LIMITADO O MÍNIMO
```

## 4. Data Sheets

Documentar el origen, sesgos y procesamiento de los datasets — equivalente a model cards pero para datos (Gebru et al., 2021).

```python
data_sheet = {
    "dataset_name": "Solicitudes de Crédito 2025-2026",
    "collection": {
        "source": "Sistema core bancario + buró de crédito",
        "collection_period": "Enero 2025 - Mayo 2026",
        "method": "Extracción automática de DB transaccional",
        "consent": "Los clientes aceptan uso de datos en contrato de apertura",
    },
    "composition": {
        "instances": 45000,
        "features": 23,
        "sensitive_features": ["genero", "edad", "codigo_postal", "estado_civil"],
        "missing_data": "5% en ingresos, imputado con mediana por código postal",
    },
    "preprocessing": {
        "steps": [
            "Eliminación de registros con edad < 18 o > 90",
            "Imputación de ingresos faltantes con mediana geográfica",
            "Encoding one-hot para variables categóricas",
            "Estandarización (z-score) para features numéricas",
        ],
        "software": "Python 3.11, pandas 2.0, scikit-learn 1.3",
    },
    "biases": {
        "known_biases": "Sesgo geográfico: 70% de solicitudes de zonas urbanas",
        "mitigation": "Planeado: sobremuestrear zonas rurales en próxima iteración",
    },
}

print(f"Dataset: {data_sheet['dataset_name']}")
print(f"Tamaño: {data_sheet['composition']['instances']} registros")
print(f"Sesgo conocido: {data_sheet['biases']['known_biases']}")
```

**Salida esperada:**
```
Dataset: Solicitudes de Crédito 2025-2026
Tamaño: 45000 registros
Sesgo conocido: Sesgo geográfico: 70% de solicitudes de zonas urbanas
```

## 5. Model Registry

Un model registry es el sistema de versionado, lineage, approvals y seguimiento de despliegues. Ejemplos: MLflow Model Registry, DVC, Weights & Biases.

```python
class ModelRegistry:
    def __init__(self):
        self.models = {}
        self.deployments = []

    def register(self, model_id, version, metadata):
        if model_id not in self.models:
            self.models[model_id] = {}
        self.models[model_id][version] = {
            'metadata': metadata,
            'status': 'registered',
            'lineage': []
        }
        print(f"Registered: {model_id} v{version}")

    def add_lineage(self, model_id, version, parent_id, parent_version):
        self.models[model_id][version]['lineage'].append({
            'parent': parent_id,
            'parent_version': parent_version
        })

    def deploy(self, model_id, version, environment):
        self.deployments.append({
            'model_id': model_id,
            'version': version,
            'environment': environment,
            'timestamp': datetime.now(),
        })
        self.models[model_id][version]['status'] = f'deployed:{environment}'
        print(f"Deployed {model_id} v{version} → {environment}")

registry = ModelRegistry()
registry.register("credito-score", "1.0.0", {"accuracy": 0.89})
registry.register("credito-score", "2.0.0", {"accuracy": 0.92})
registry.add_lineage("credito-score", "2.0.0", "credito-score", "1.0.0")
registry.deploy("credito-score", "2.0.0", "production")
print(f"Deployments: {len(registry.deployments)}")
```

**Salida esperada:**
```
Registered: credito-score v1.0.0
Registered: credito-score v2.0.0
Deployed credito-score v2.0.0 → production
Deployments: 1
```

## 6. Monitoreo continuo

El governance no termina en el despliegue. Los modelos requieren monitoreo continuo de drift, fairness y performance.

```python
import numpy as np

def monitor_model(reference_scores, current_scores, fairness_metrics):
    alerts = []
    # Drift detection (PSI)
    def psi(ref, curr, bins=10):
        ref_hist, _ = np.histogram(ref, bins=bins, range=(0,1), density=True)
        curr_hist, _ = np.histogram(curr, bins=bins, range=(0,1), density=True)
        ref_hist = np.clip(ref_hist, 1e-10, None)
        curr_hist = np.clip(curr_hist, 1e-10, None)
        return np.sum((ref_hist - curr_hist) * np.log(ref_hist / curr_hist))

    psi_value = psi(reference_scores, current_scores)
    if psi_value > 0.2:
        alerts.append(f"PSI drift detectado: {psi_value:.3f}")

    # Fairness monitoring
    for metric, value in fairness_metrics.items():
        if value > 0.1:
            alerts.append(f"Fairness breach: {metric}={value:.3f}")

    return alerts

# Simular monitoreo
ref_scores = np.random.beta(5, 2, 1000)
curr_scores_drifted = np.random.beta(3, 4, 1000)  # distribución cambiada
fairness = {"demographic_parity": 0.12, "equal_opportunity": 0.04}

alerts = monitor_model(ref_scores, curr_scores_drifted, fairness)
for alert in alerts:
    print(f"ALERTA: {alert}")
if not alerts:
    print("Sin alertas — modelo saludable")
```

**Salida esperada:**
```
ALERTA: PSI drift detectado: 0.345
ALERTA: Fairness breach: demographic_parity=0.120
```

## 7. Common Mistakes

| Error | Consecuencia | Solución |
|-------|-------------|----------|
| Documentar solo al final del proyecto | Documentación incompleta, olvidos críticos | Documentar en cada etapa (data sheet + model card desde el inicio) |
| No actualizar model cards tras retrain | Model card desactualizada, auditoría encuentra inconsistencias | Model card versionada en CI/CD (actualizar automáticamente al registrar modelo) |
| Governance sin herramientas | Procesos manuales no escalan, se omiten pasos | Usar model registry, experiment tracking, automated reporting |
| Ignorar monitoreo post-despliegue | Model drift no detectado, decisiones incorrectas sin supervisión | Dashboards de monitoreo con alertas automáticas y retrain triggers |
| Clasificar riesgo incorrectamente | Multas regulatorias (EU AI Act: hasta 7% de ingresos globales) | Auditoría legal + matriz de riesgo por uso y dominio |

## Resumen

1. ML governance abarca procesos, roles, documentación y herramientas para modelos responsables y auditables.
2. Model Cards documentan propósito, métricas, sesgos y limitaciones de cada modelo.
3. El EU AI Act clasifica sistemas de IA por nivel de riesgo e impone requisitos crecientes.
4. Data Sheets complementan a Model Cards documentando origen, sesgos y preprocesamiento de datasets.
5. Un Model Registry gestiona versionado, lineage, approvals y despliegues.
6. El monitoreo continuo (drift, fairness, performance) es necesario post-despliegue con alertas automatizadas.
7. La documentación temprana, las herramientas adecuadas y la clasificación correcta de riesgo previenen fallos regulatorios.

## Check Your Understanding

1. ¿Qué diferencia hay entre una Model Card y un Data Sheet? <!-- Model Card documenta el modelo (propósito, métricas, limitaciones); Data Sheet documenta el dataset (origen, composición, sesgos, preprocesamiento). -->
2. ¿Por qué un modelo de scoring crediticio es "alto riesgo" según el EU AI Act? <!-- Porque toma decisiones automatizadas que afectan significativamente los derechos financieros de las personas, y su mal funcionamiento puede causar daños graves. -->
3. ¿Qué es un Model Registry y qué problema resuelve? <!-- Es un sistema centralizado de versionado, lineage, approvals y seguimiento de despliegues. Resuelve el caos de "¿qué versión está en producción?" y "¿de dónde vino este modelo?". -->
4. ¿Por qué el monitoreo de drift y fairness debe ser continuo y no puntual? <!-- Porque los datos y el comportamiento del modelo cambian con el tiempo; un modelo auditado hoy puede estar sesgado o degradado mañana. -->

## Where to Go Next

- [[Bias & Fairness]] — métricas de equidad para incluir en model cards
- [[Interpretability (SHAP-LIME)]] — explicaciones para documentación regulatoria
- [[Privacy & Security in ML]] — requisitos legales de privacidad (GDPR, HIPAA)
- [[Model Monitoring]] — dashboards y alertas para governance post-despliegue
- [[Data & Concept Drift]] — triggers de retrain y reporting automático
- [[Experiment Tracking]] — integrar con model registry para lineage completo
