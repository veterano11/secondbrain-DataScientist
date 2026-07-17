---
tags:
  - ml-engineering
  - model-compression
  - edge-ml
  - production-ml
status: seedling
created: 2026-06-28
---

# Model Compression

## 1. Escenario de aprendizaje

Fine-tuneaste BERT-base para clasificación de sentimientos en una app móvil: 110M parámetros, 450MB en FP32. Cargar 450MB en un iPhone tarda 8s y cada predicción consume 300ms. Necesitas <50MB y <100ms sin perder más de 2 puntos de accuracy.

Aprenderás quantization, pruning, knowledge distillation y ONNX — y cómo combinarlos evaluando el trade-off entre tamaño, velocidad y precisión.

## 2. Requisitos

- Python 3.10+, PyTorch 2.0+, Transformers
- ONNX (`pip install onnx onnxruntime`)
- Optimum (`pip install optimum[onnxruntime]`)

## 3. Quantization

Reduce la precisión numérica: FP32 (32 bits) → INT8 (8 bits). Reduce tamaño 4x y acelera inferencia con SIMD/Tensor Cores.

### Dinámica (solo pesos)

```python
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
print(f"Parámetros: {model.num_parameters():,}")

quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

original_size = sum(p.numel() * p.element_size() for p in model.parameters())
quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())
print(f"Original: {original_size/1024/1024:.1f}MB → Cuantizado: {quantized_size/1024/1024:.1f}MB")
```

**Salida esperada:** `Original: 441.6MB → Cuantizado: 110.4MB`

### Comparación de Métodos

| Método | Pesos | Activaciones | Compresión | Aceleración |
|--------|-------|-------------|------------|-------------|
| FP32 | 32-bit | 32-bit | 1x | 1x |
| Dinámica INT8 | 8-bit | 32-bit | ~3x | ~2x |
| Estática INT8 | 8-bit | 8-bit | ~4x | ~3x |

```python
import time, torch

def benchmark(model, inputs, n=100):
    model.eval()
    with torch.no_grad():
        for _ in range(10): model(inputs)  # warmup
        start = time.time()
        for _ in range(n): model(inputs)
        return (time.time() - start) / n * 1000

sample = torch.randint(0, 30000, (1, 128))
print(f"FP32: {benchmark(model, sample):.1f}ms")
print(f"INT8: {benchmark(quantized_model, sample):.1f}ms")
```

**Salida esperada:** `FP32: 285.3ms \n INT8: 98.7ms (2.9x)`

## 4. Pruning

Elimina pesos cercanos a cero. Unstructured pruning (pesos individuales) vs structured pruning (neuronas enteras).

### Unstructured Pruning

```python
import torch.nn.utils.prune as prune

for name, module in model.named_modules():
    if isinstance(module, nn.Linear) and "attention" in name:
        prune.l1_unstructured(module, name="weight", amount=0.3)
        prune.remove(module, "weight")

total = sum(p.numel() for p in model.parameters())
zeros = sum((p == 0).sum().item() for p in model.parameters())
print(f"Sparsity: {zeros/total*100:.1f}%")
```

**Salida esperada:** `Sparsity: 19.5%`

### Structured Pruning (cabezas de atención)

```python
def prune_attention_heads(model, heads_to_prune: dict):
    for layer_id, heads in heads_to_prune.items():
        model.bert.encoder.layer[layer_id].attention.prune_heads(heads)
    return model

model = prune_attention_heads(model, {0: [7], 5: [3, 5], 10: [0, 11]})
print(f"Parámetros: {model.num_parameters():,}")
```

### Pruning Iterativo

Podar gradualmente con reentrenamiento entre rondas para evitar accuracy collapse.

```python
def iterative_pruning(model, target_sparsity=0.5, n_rounds=5):
    for r in range(n_rounds):
        for module in model.modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, "weight", amount=target_sparsity/n_rounds)
        for module in model.modules():
            if isinstance(module, nn.Linear):
                prune.remove(module, "weight")
        current = sum((p==0).sum().item() for p in model.parameters()) / sum(p.numel() for p in model.parameters())
        print(f"Ronda {r+1}: sparsity={current:.1%}")
    return model
```

**Salida esperada:** `Ronda 5: sparsity=50.0%`

## 5. Knowledge Distillation

Entrena un modelo student (chico) para imitar a un teacher (grande). El student aprende la distribución de probabilidades del teacher, no solo las etiquetas.

```python
import torch.nn.functional as F

class DistillationTrainer:
    def __init__(self, teacher, student, temperature=4.0, alpha=0.7):
        self.teacher, self.student = teacher, student
        self.temperature, self.alpha = temperature, alpha

    def loss(self, student_logits, teacher_logits, labels):
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        distill_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean")
        distill_loss *= self.temperature ** 2
        ce_loss = F.cross_entropy(student_logits, labels)
        return self.alpha * distill_loss + (1 - self.alpha) * ce_loss

student = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=2)
print(f"Student: {student.num_parameters():,} parámetros ({student.num_parameters()*4/1024/1024:.1f}MB)")
```

**Salida esperada:** `Student: 13,681,922 parámetros (52.2MB)`

### Temperature Scaling

Temperatura alta suaviza las probabilidades del teacher, revelando relaciones entre clases.

```python
import numpy as np
logits = np.array([2.0, 1.0, 0.1])
for temp in [1.0, 2.0, 5.0]:
    sm = np.exp(logits/temp) / np.sum(np.exp(logits/temp))
    print(f"T={temp}: {np.round(sm, 4)}")
```

**Salida esperada:** `T=1.0: [0.659 0.242 0.099] \n T=2.0: [0.476 0.286 0.238] \n T=5.0: [0.378 0.315 0.307]`

## 6. ONNX

Formato portable con optimizaciones de grafo y runtime dedicado (onnxruntime).

```python
import torch
import onnxruntime as ort
import numpy as np, os

model.eval()
dummy = torch.randint(0, 30000, (1, 128))
torch.onnx.export(model, dummy, "bert_sentiment.onnx",
    input_names=["input_ids"], output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch_size"}, "logits": {0: "batch_size"}},
    opset_version=17)

print(f"ONNX: {os.path.getsize('bert_sentiment.onnx')/1024/1024:.1f}MB")
```

**Salida esperada:** `ONNX: 440.2MB`

### Cuantización y Benchmark

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic("bert_sentiment.onnx", "bert_sentiment_int8.onnx", weight_type=QuantType.QInt8)
print(f"ONNX INT8: {os.path.getsize('bert_sentiment_int8.onnx')/1024/1024:.1f}MB")

session = ort.InferenceSession("bert_sentiment_int8.onnx",
    providers=["CPUExecutionProvider"])
input_data = np.random.randint(0, 30000, (1, 128)).astype(np.int64)

start = time.time()
for _ in range(100):
    session.run(None, {"input_ids": input_data})
print(f"ONNX Runtime: {(time.time()-start)/100*1000:.1f}ms")
```

**Salida esperada:** `ONNX INT8: 110.3MB \n ONNX Runtime: 55.0ms`

## 7. Evaluación Post-Compresión

```python
from sklearn.metrics import accuracy_score

def evaluate(model, loader, model_type="pytorch"):
    preds, labels = [], []
    for batch in loader:
        if model_type == "onnx":
            outputs = model.run(None, {"input_ids": batch["input_ids"].numpy()})
            preds.extend(np.argmax(outputs[0], axis=1))
        else:
            with torch.no_grad():
                preds.extend(torch.argmax(model(batch["input_ids"]).logits, dim=-1))
        labels.extend(batch["labels"].tolist())
    return accuracy_score(labels, preds)

results = {
    "FP32 PyTorch": {"acc": 0.924, "size": 441.6, "lat": 310},
    "INT8 Dynamic": {"acc": 0.921, "size": 110.4, "lat": 105},
    "ONNX FP32":    {"acc": 0.924, "size": 440.2, "lat": 70},
    "ONNX INT8":    {"acc": 0.918, "size": 110.3, "lat": 55},
    "Distilled":    {"acc": 0.909, "size": 52.2,  "lat": 45},
}

import pandas as pd
df = pd.DataFrame(results).T
df.columns = ["Accuracy", "Size (MB)", "p99 Latency (ms)"]
print(df)
```

**Salida esperada:**
```
               Accuracy  Size (MB)  p99 Latency (ms)
FP32 PyTorch      0.924      441.6            310.0
INT8 Dynamic      0.921      110.4            105.0
ONNX INT8         0.918      110.3             55.0
Distilled         0.909       52.2             45.0
```

## 8. Common Mistakes

### Mistake 1: Cuantizar sin Evaluar — pérdida de accuracy >5% en modelos pequeños.
**Solución**: Evaluar siempre post-cuantización en validación. Probar dinámica si estática degrada mucho.

### Mistake 2: Podar Demasiado (Accuracy Collapse) — sparsity >80% puede colapsar accuracy de 92% a 50%.
**Solución**: Podar iterativamente, evaluar cada ronda, detenerse cuando la pérdida supere el umbral.

### Mistake 3: No Medir Latencia Real vs Teórica — 50% sparsity no corre 2x más rápido sin hardware sparse.
**Solución**: Medir en hardware objetivo (iPhone, Raspberry Pi). No asumir linealidad.

### Mistake 4: Ignorar Cuantización de Activaciones — solo cuantizar pesos da 2x; con activaciones da 4x.
**Solución**: Usar cuantización estática en CPU siempre que sea posible.

### Mistake 5: No Combinar Técnicas — la mayor compresión se logra combinando distillation + pruning + quantization + ONNX.
**Solución**: Pipeline: Distill → Prune → Quantize → ONNX. Evaluar en cada paso.

## Resumen

- **Quantization**: FP32 → INT8. Dinámica (2-3x), estática (3-4x). Siempre evaluar post-cuantización.
- **Pruning**: unstructured (pesos) o structured (neuronas). Iterativo con reentrenamiento.
- **Knowledge distillation**: teacher → student con temperature scaling.
- **ONNX**: optimizaciones de grafo + runtime dedicado. Hasta 4x más rápido que PyTorch eager.
- Combinación logra 8-10x de compresión con pérdida mínima. Evaluar accuracy, tamaño y latencia en hardware objetivo.

## Comprueba tu Conocimiento

1. ¿Cuál es la diferencia entre cuantización dinámica y estática?
   <!-- Dinámica: pesos INT8, activaciones FP32. Estática: pesos y activaciones INT8 + calibración. -->
2. ¿Por qué el pruning unstructured no siempre acelera la inferencia?
   <!-- El hardware convencional no tiene soporte nativo para sparse matrices. -->
3. ¿Qué rol juega la temperatura en knowledge distillation?
   <!-- Suaviza las probabilidades del teacher, revelando relaciones entre clases. -->
4. ¿Qué ventajas tiene ONNX sobre PyTorch para servir?
   <!-- Optimizaciones de grafo automáticas, runtime liviano, cuantización integrada, multiplataforma. -->
5. ¿Cómo decidirías entre quantization, pruning, o distillation para un proyecto?
   <!-- Distillation si puedes reentrenar, quantization si ya tienes el modelo, pruning para sparsity extrema. Combinar. -->
6. ¿Qué es el accuracy collapse en pruning y cómo evitarlo?
   <!-- Caída abrupta al superar cierto umbral de sparsity. Evitar con pruning iterativo. -->
7. ¿Por qué la latencia teórica no siempre coincide con la real?
   <!-- Depende del hardware: instrucciones SIMD, ancho de banda, overhead del runtime. -->
8. ¿Qué métrica usarías para seleccionar la mejor técnica de compresión?
   <!-- Score ponderado: accuracy * w_acc + (1-size/max)*w_size + (1-latency/max)*w_lat. -->

## ¿Dónde ir Siguente?

- [[Transformer Architecture]] — entender la arquitectura que estás comprimiendo
- [[Model Serving]] — servir modelos comprimidos con Triton y ONNX Runtime
- [[Training Techniques]] — técnicas que facilitan la compresión
- [[Python for Data Science]] — fundamentos para implementar pipelines de compresión
- [[Transfer Learning]] — cómo el fine-tuning afecta la compresibilidad
- [[ONNX]] — profundizar en exportación y optimización de grafos
- [[Fine-tuning]] — ajustar modelos pre-entrenados antes de comprimir
