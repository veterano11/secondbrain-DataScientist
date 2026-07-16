---
tags: [deep-learning, transfer-learning, advanced]
status: growing
created: 2026-06-27
---

# Transfer Learning

## 1. Escenario de aprendizaje

Tienes 500 imágenes de rayos X para clasificar entre neumonía y normal. Entrenar una red neuronal profunda desde cero requeriría millones de imágenes etiquetadas, días de GPU y experiencia significativa. Transfer learning — comenzar desde un modelo ya entrenado en un gran conjunto de datos genérico (como ImageNet) — reduce esto a cientos de ejemplos y horas de fine-tuning en una sola GPU. Prácticamente toda aplicación práctica de deep learning usa transfer learning. No es una técnica especializada; es la práctica estándar.

---

## 2. La Idea Central

**Pre-entrenamiento**: entrenar un modelo en un conjunto de datos grande y genérico (ImageNet para visión, Wikipedia + libros para NLP).

**Fine-tuning**: adaptar el modelo pre-entrenado a tu tarea específica con un conjunto de datos más pequeño.

**Por qué funciona**: las primeras capas aprenden características generales (bordes, texturas en visión; sintaxis, gramática en lenguaje). Solo las últimas capas necesitan especializarse en la tarea objetivo.

---

## 3. Estrategias de Transfer Learning

### 3.1 Extracción de Características

- Congelar el backbone pre-entrenado (todos los pesos se mantienen fijos)
- Reemplazar la(s) capa(s) de clasificación final(es)
- Entrenar solo las capas nuevas

**Cuándo usar**: conjunto de datos objetivo muy pequeño (< 1000 ejemplos), dominio similar a los datos de pre-entrenamiento.

```python
import torchvision.models as models

# Cargar ResNet pre-entrenado
backbone = models.resnet50(weights="IMAGENET1K_V2")

# Congelar todas las capas
for param in backbone.parameters():
    param.requires_grad = False

# Reemplazar clasificador
backbone.fc = nn.Linear(2048, num_classes)

# Solo la capa del clasificador nueva es entrenable
```

### 3.2 Fine-Tuning Completo

- Descongelar todo el modelo
- Entrenar todo con una tasa de aprendizaje baja

**Cuándo usar**: conjunto de datos objetivo moderado (1000-10000 ejemplos), o dominio diferente al de pre-entrenamiento.

### 3.3 Descongelamiento Progresivo

- Comenzar con solo el clasificador nuevo entrenable
- Descongelar gradualmente capas de arriba hacia abajo
- Cada paso de descongelamiento usa una tasa de aprendizaje más baja

**Por qué**: las capas tempranas aprenden características muy generales que deberían cambiar menos. Exponerlas gradualmente a los datos objetivo previene el olvido catastrófico.

### 3.4 Fine-Tuning Eficiente en Parámetros

En lugar de actualizar todos los parámetros, insertar módulos pequeños entrenables:

| Método | Qué se entrena | Parámetros | Usado en |
|---|---|---|---|
| **LoRA** | Actualizaciones de pesos de bajo rango | ~0.1-1% | LLMs |
| **Adapters** | Capas bottleneck pequeñas | ~1-5% | NLP |
| **Prefix Tuning** | Embeddings de tokens virtuales | ~0.1% | NLG |

---

## 4. Modelos Pre-entrenados Comunes

### 4.1 Visión

| Modelo | Datos de Pre-entrenamiento | Mejor Para |
|---|---|---|
| **ResNet** (18/50/101) | ImageNet (1.3M imágenes) | Clasificación, detección — ver [[CNNs]] |
| **EfficientNet** | ImageNet + student ruidoso | Eficiencia |
| **ViT** (Vision Transformer) | ImageNet-21k / JFT-300M | Visión (basado en transformer) — ver [[Transformers]] |
| **CLIP** (OpenAI) | 400M pares imagen-texto | Zero-shot, multi-modal |

### 4.2 NLP / LLMs

| Modelo | Parámetros | Datos de Pre-entrenamiento | Mejor Para |
|---|---|---|---|
| **BERT** | 110M-340M | Libros + Wikipedia | Comprensión (clasificación, NER) |
| **RoBERTa** | 125M-355M | 160GB texto | Comprensión (BERT mejorado) |
| **GPT-2** | 124M-1.5B | WebText | Generación |
| **Llama 2/3** | 7B-70B | 2T-15T tokens | Propósito general |
| **Mistral** | 7B | Varios | Propósito general (eficiente) |
| **Gemma** | 2B-7B | 6T tokens | Investigación |

---

## 5. Guías Prácticas

### 5.1 Elegir una Estrategia

| Datos Objetivo | Similitud de Dominio | Estrategia Recomendada |
|---|---|---|
| < 1000 | Similar | Extracción de características |
| < 1000 | Diferente | Fine-tuning de capas superiores |
| 1K-10K | Similar | Fine-tuning completo con LR baja |
| 1K-10K | Diferente | Fine-tuning completo desde pre-entrenado |
| > 10K | Cualquiera | Considerar entrenamiento desde cero |

### 5.2 Mejores Prácticas de Fine-Tuning

- **Tasa de aprendizaje**: 10-100× más baja que entrenar desde cero. Para fine-tuning completo: 2e-5 a 5e-5. Para LoRA: 1e-4 a 5e-4.
- **Optimizador**: AdamW es estándar. SGD con momentum funciona para visión.
- **Épocas**: menos que entrenar desde cero. Monitorear pérdida de validación — el fine-tuning puede hacer overfitting rápidamente.
- **Tamaño de lote**: tan grande como la memoria permita (pero no demasiado — lotes pequeños actúan como regularizador).
- **Aumento de datos**: aún más importante con conjuntos de datos pequeños.

### 5.3 Cuándo NO Usar Transfer Learning

- El dominio objetivo es fundamentalmente diferente al dominio de pre-entrenamiento (ej: imágenes médicas de una modalidad de imagenado novedosa)
- Tienes un conjunto de datos objetivo muy grande (millones de ejemplos)
- Las restricciones de latencia requieren un modelo mucho más pequeño
- Quieres entender la arquitectura desde cero (investigación)

---

## 6. Errores Comunes

1. **Tasa de aprendizaje demasiado alta**: el error de fine-tuning más común. Los pesos pre-entrenados ya son buenos — actualizaciones grandes destruyen características aprendidas.

2. **No congelar estadísticas de batch norm**: la media/varianza en ejecución de batchnorm debe congelarse al hacer fine-tuning con lotes pequeños. Actualizarlos solo si se entrena con lotes grandes.

3. **Overfitting a datos pequeños**: fine-tuning con cientos de ejemplos puede hacer overfitting. Usar [[Regularization]] más fuerte, early stopping y aumento de datos.

4. **Olvido catastrófico**: el modelo puede "olvidar" las características generales aprendidas durante el pre-entrenamiento. Usar descongelamiento progresivo o replay de datos de pre-entrenamiento.

5. **No ajustar el tamaño de entrada**: los modelos pre-entrenados esperan tamaños de entrada específicos (224×224 para ResNet, 512/1024 tokens para BERT). Redimensionar tus datos en consecuencia.

---

## 7. Comprueba tu Conocimiento

1. Tienes 500 imágenes de rayos X médicas etiquetadas. ¿Deberías entrenar ResNet desde cero, usar extracción de características, o hacer fine-tuning del modelo completo? ¿Por qué?

2. ¿Por qué la tasa de aprendizaje para fine-tuning típicamente es 100× más baja que entrenar desde cero?

3. LoRA entrena 1% de los parámetros pero logra un rendimiento similar al fine-tuning completo en LLMs. ¿Cómo?

4. Haces fine-tuning de un modelo BERT en un conjunto de datos personalizado. La pérdida de entrenamiento disminuye pero la pérdida de validación aumenta después de 2 épocas. ¿Qué haces?

5. ¿Cuándo NO usarías transfer learning?

---

## 8. Resumen

El transfer learning es la práctica estándar en deep learning. Comenzar desde un modelo pre-entrenado, adaptarlo a tu tarea con una pequeña cantidad de datos y una tasa de aprendizaje baja. Elegir la estrategia (extracción de características, fine-tuning o PEFT) según el tamaño de tu conjunto de datos y la similitud de dominio. La clave es no destruir las características pre-entrenadas con actualizaciones agresivas.

---

## 9. ¿Dónde ir Siguente?

- [[Neural Networks]] — Qué se está transfiriendo
- [[Fine-tuning]] — Transfer learning específico para LLMs (LoRA, QLoRA)
- [[Training Techniques]] — Optimizadores y programaciones para fine-tuning
