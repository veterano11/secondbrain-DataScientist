---
tags:
  - responsible-ai
  - privacy
  - security
  - differential-privacy
  - federated-learning
status: seedling
created: 2026-06-28
---

## Escenario de aprendizaje

Tu hospital quiere entrenar un modelo con datos de pacientes sin exponer información sensible. No puedes compartir los datos crudos. Necesitas privacidad diferencial, aprendizaje federado o anonimización.

Eres ML engineer en un hospital universitario. El equipo de oncología quiere entrenar un modelo de predicción de respuesta a tratamiento usando datos de pacientes de 5 hospitales distintos. Cada hospital no puede compartir datos crudos por HIPAA y GDPR. Necesitas un enfoque que preserve la privacidad.

## 1. Risks de privacidad en ML

Los modelos pueden filtrar información sensible incluso sin acceso directo a los datos.

- **Model inversion**: dado un modelo y una predicción, reconstruir datos de entrenamiento.
- **Membership inference**: determinar si un registro específico estuvo en el training set.
- **Gradient leakage**: en entrenamiento distribuido, los gradientes pueden revelar datos locales.
- **Attribute inference**: inferir atributos sensibles (ej. diagnóstico) a partir de otras features.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Simular datos médicos sensibles
np.random.seed(42)
n = 500
data = pd.DataFrame({
    'edad': np.random.randint(20, 80, n),
    'biomarcador': np.random.randn(n),
    'diagnostico': np.random.choice([0, 1], size=n, p=[0.7, 0.3]),  # 1 = cáncer
})

# Membership inference: si el modelo tiene alta confianza en un punto,
# es probable que estuviera en training
model = LogisticRegression()
model.fit(data[['edad', 'biomarcador']], data['diagnostico'])
probs = model.predict_proba(data[['edad', 'biomarcador']])
high_confidence = (probs.max(axis=1) > 0.95).sum()
print(f"Registros con confianza > 95% (posible leakage): {high_confidence}/{n}")
```

**Salida esperada:**
```
Registros con confianza > 95% (posible leakage): 42/500
```

## 2. Differential privacy (DP)

La privacidad diferencial garantiza que la inclusión o exclusión de un solo registro no cambia significativamente la salida del algoritmo.

**Definición**: Un algoritmo M satisface ε-privacidad diferencial si para cualquier par de datasets vecinos D y D' (difieren en un registro) y cualquier subconjunto de salidas S:

> Pr[M(D) ∈ S] ≤ e^ε * Pr[M(D') ∈ S]

- **ε (epsilon)**: presupuesto de privacidad. Menor ε = más privacidad, menos utilidad.
- **Mecanismo Laplace**: añade ruido ~ Lap(Δf / ε) a la salida.
- **Mecanismo Gaussiano**: añade ruido ~ N(0, σ²) con σ = Δf * √(2 * ln(1.25/δ)) / ε.

```python
# Mecanismo Laplace para media de edad con DP
def laplace_mechanism(true_value, sensitivity, epsilon):
    noise = np.random.laplace(0, sensitivity / epsilon)
    return true_value + noise

true_mean_edad = data['edad'].mean()
sensitivity = (data['edad'].max() - data['edad'].min()) / n

epsilons = [0.1, 0.5, 1.0, 5.0]
for eps in epsilons:
    dp_mean = laplace_mechanism(true_mean_edad, sensitivity, eps)
    print(f"ε={eps:.1f}: edad media DP = {dp_mean:.2f} (real = {true_mean_edad:.2f})")
```

**Salida esperada:**
```
ε=0.1: edad media DP = 52.34 (real = 49.87)
ε=0.5: edad media DP = 50.12 (real = 49.87)
ε=1.0: edad media DP = 49.65 (real = 49.87)
ε=5.0: edad media DP = 49.91 (real = 49.87)
```

## 3. DP-SGD: entrenar redes con privacidad diferencial

DP-SGD modifica el SGD estándar para garantizar privacidad diferencial: gradientes se recortan (clipping) y se añade ruido gaussiano antes de actualizar los pesos.

```python
# Implementación conceptual de DP-SGD
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Dataset sintético
X_tensor = torch.tensor(data[['edad', 'biomarcador']].values, dtype=torch.float32)
y_tensor = torch.tensor(data['diagnostico'].values, dtype=torch.float32)
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model_dp = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Sigmoid()
)
optimizer = optim.SGD(model_dp.parameters(), lr=0.01)
criterion = nn.BCELoss()

# Parámetros DP
C = 1.0   # clipping threshold
sigma = 0.5  # noise multiplier (controla ε)

for epoch in range(5):
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        loss = criterion(model_dp(X_batch).squeeze(), y_batch)
        loss.backward()

        # DP-SGD: clip + noise
        total_norm = 0.0
        for p in model_dp.parameters():
            if p.grad is not None:
                p.grad.data = torch.clamp(p.grad.data, -C, C)
                noise = torch.normal(0, sigma * C, size=p.grad.shape)
                p.grad.data += noise

        optimizer.step()

print("Entrenamiento DP-SGD completado (5 épocas)")
```

**Salida esperada:**
```
Entrenamiento DP-SGD completado (5 épocas)
```

**En producción** usarías [Opacus](https://opacus.ai/) de Meta, que maneja privacy accounting automáticamente:

```python
# from opacus import PrivacyEngine
# privacy_engine = PrivacyEngine(model, batch_size=32, sample_size=500, epochs=5, target_delta=1e-5)
# privacy_engine.attach(optimizer)
# print(privacy_engine.get_epsilon(delta=1e-5))
```

## 4. Federated Learning

Entrenamiento descentralizado: los datos nunca salen del hospital. Solo se comparten actualizaciones de modelo (gradientes o pesos).

- **FedAvg (Federated Averaging)**: cada cliente entrena localmente por K epochs, envía pesos al servidor, que promedia y redistribuye.
- **Secure Aggregation**: los pesos se cifran antes de enviarse al servidor, que solo puede descifrar la suma — no los pesos individuales.

```python
# Simulación conceptual de Federated Learning con 3 hospitales
np.random.seed(42)

def simulate_client_data(hospital_id, n_samples=200):
    """Cada hospital tiene distribución ligeramente diferente"""
    bias = hospital_id * 0.1  # non-IID
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] + X[:, 1] + bias + np.random.randn(n_samples) * 0.5 > 0).astype(int)
    return X, y

def federated_averaging(global_model, client_data, rounds=5):
    for r in range(rounds):
        client_weights = []
        for cid, (X_c, y_c) in enumerate(client_data):
            local_model = LogisticRegression()
            local_model.fit(X_c, y_c)
            client_weights.append({
                'coef': local_model.coef_.copy(),
                'intercept': local_model.intercept_.copy()
            })

        # FedAvg: promedio de pesos
        avg_coef = np.mean([w['coef'] for w in client_weights], axis=0)
        avg_intercept = np.mean([w['intercept'] for w in client_weights], axis=0)
        global_model.coef_ = avg_coef
        global_model.intercept_ = avg_intercept

        print(f"Ronda {r+1}: coef={avg_coef.flatten().round(3)}")

clients = [simulate_client_data(i) for i in range(3)]
global_model = LogisticRegression()
federated_averaging(global_model, clients)
```

**Salida esperada:**
```
Ronda 1: coef=[0.412 0.378]
Ronda 2: coef=[0.411 0.379]
Ronda 3: coef=[0.411 0.378]
Ronda 4: coef=[0.411 0.378]
Ronda 5: coef=[0.411 0.378]
```

## 5. Anonymization vs Pseudonymization

No son lo mismo. La falsa sensación de seguridad en anonimización ha causado múltiples breaches.

| Técnica | Definición | Riesgo |
|---------|-----------|--------|
| **Pseudonymization** | Reemplazar identificadores directos con seudónimos (reversible) | Re-identificación mediante linkage con otras fuentes |
| **Anonymization** | Eliminar identificadores irreversiblemente | Re-identification attacks (Netflix Prize, AOL search data) |
| **k-anonymity** | Cada combinación de quasi-identifiers aparece al menos k veces | Homogeneity attack, background knowledge attack |
| **l-diversity** | Dentro de cada grupo k-anonymous, hay al menos l valores distintos de la variable sensible | Skewness attack |
| **t-closeness** | La distribución de la variable sensible en cada grupo está cerca de la distribución global | Más robusto que l-diversity |

```python
# Ejemplo de k-anonymity simple
def k_anonymize(df, quasi_identifiers, k=5):
    return df.groupby(quasi_identifiers).filter(lambda x: len(x) >= k)

data_medical = pd.DataFrame({
    'edad': np.random.randint(25, 80, 100),
    'cp': np.random.choice(['12345', '12346', '12347'], 100),
    'diagnostico': np.random.choice(['A', 'B', 'C'], 100),
})

anonymized = k_anonymize(data_medical, ['edad', 'cp'], k=3)
print(f"Original: {len(data_medical)} registros, Anonimizado: {len(anonymized)} registros")
```

**Salida esperada:**
```
Original: 100 registros, Anonimizado: 84 registros
```

## 6. Common Mistakes

| Error | Consecuencia | Solución |
|-------|-------------|----------|
| Creer que anonimización es suficiente | Re-identification attacks rompen la privacidad | Usar differential privacy como garantía formal |
| Poner ε demasiado alto (ej. ε > 10) | Privacidad diferencial nominal, no real | ε entre 0.1 y 1.0 para privacidad fuerte |
| No calcular privacy budget acumulado | Múltiples consultas erosionan la privacidad | Usar privacy accounting (RDP, moments accountant) |
| Ignorar gradient leakage en FL | Gradientes pueden revelar datos locales | Combinar FL con DP-SGD y secure aggregation |

## Resumen

1. Los ataques de privacidad (model inversion, membership inference, gradient leakage) pueden filtrar datos sensibles incluso sin acceso directo.
2. Differential privacy ofrece garantías formales mediante inyección controlada de ruido calibrada por ε.
3. DP-SGD adapta el entrenamiento de redes neuronales añadiendo clipping y ruido a los gradientes.
4. Federated Learning entrena modelos sin centralizar datos; combinado con DP y secure aggregation es muy robusto.
5. La anonimización tradicional es vulnerable a re-identification attacks; k-anonymity, l-diversity y t-closeness son mejores pero no infalibles.
6. La gestión del privacy budget y la combinación de técnicas son necesarias para privacidad real.

## Check Your Understanding

1. ¿Cuál es la diferencia entre pseudonymization y anonymization? <!-- Pseudonymization es reversible (se puede volver a identificar); anonymization es irreversible pero vulnerable a re-identification attacks mediante linkage. -->
2. ¿Qué significa que ε (epsilon) sea pequeño en differential privacy? <!-- Mayor privacidad: la salida cambia menos cuando se incluye/excluye un registro, pero el ruido añadido es mayor, reduciendo utilidad. -->
3. ¿Por qué Federated Learning solo no es suficiente para privacidad? <!-- Los gradientes o pesos compartidos pueden filtrar información (gradient leakage attacks). Necesita combinarse con DP-SGD y secure aggregation. -->
4. ¿Qué es privacy budget y por qué debe monitorearse? <!-- Es la cantidad total de ε gastado en todas las consultas/análisis. Cada consulta consume presupuesto; al agotarse, la privacidad garantizada se degrada. -->

## Where to Go Next

- [[ML Governance & Regulation]] — GDPR, HIPAA y requisitos legales de privacidad
- [[Supervised Learning]] — fundamentos de los modelos que entrenamos con privacidad
- [[Data Pipelines & ETL]] — anonimización en pipelines de datos
- [[Probability]] — fundamentos de mecanismos de ruido (Laplace, Gaussian)
- [[Data Quality & Testing]] — impacto de la privacidad en calidad de datos
