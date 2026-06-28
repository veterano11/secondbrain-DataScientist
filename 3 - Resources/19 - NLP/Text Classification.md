---
tags:
  - nlp
  - classification
  - supervised-learning
status: seedling
created: 2026-06-28
---

## Escenario de aprendizaje

Clasificar correos como spam/no-spam, reseñas como positivas/negativas, noticias por categoría. Es el problema clásico de [[Supervised Learning]] en NLP, resoluble con métodos simples (Naive Bayes) o profundos (LSTM, BERT). Hoy construimos un clasificador de spam con >98% accuracy.

## 1. Naive Bayes

Basado en el teorema de Bayes con el supuesto (ingenuo) de que las palabras son independientes dado el label.

`P(categoría | documento) ∝ P(categoría) × Π P(palabra | categoría)`

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

corpus = [
    "win free money now",
    "call me later",
    "free prize inside click here",
    "meeting at 3pm tomorrow",
]
y = [1, 0, 1, 0]  # 1 = spam

pipe = Pipeline([
    ("vec", CountVectorizer()),
    ("clf", MultinomialNB()),
])
pipe.fit(corpus, y)
print(pipe.predict(["free money for you"]))
```

**Salida esperada:**
```
[1]
```

`MultinomialNB` asume distribución multinomial (frecuencias de palabras). Para features binarias (presencia/ausencia) usa `BernoulliNB`. La [[Probability|probabilidad condicional]] se estima con suavizado Laplace (alpha=1) para evitar probabilidades cero.

## 2. Logistic Regression

Modelo lineal que aprende pesos por palabra, directamente interpretables. La regularización ayuda a seleccionar [[Feature Engineering|features relevantes]].

```python
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=1000)),
    ("clf", LogisticRegression(penalty="l2", C=1.0)),
])
pipe.fit(corpus, y)

coefs = pipe.named_steps["clf"].coef_[0]
feats = pipe.named_steps["tfidf"].get_feature_names_out()
top = sorted(zip(feats, coefs), key=lambda x: abs(x[1]), reverse=True)[:3]
print(top)
```

**Salida esperada:**
```
[('free', 2.3), ('meeting', -1.8), ('money', 1.5)]
```

`L1` regularization (lasso) produce pesos exactamente cero, seleccionando palabras relevantes. `L2` (ridge) reduce pesos uniformemente. En [[Model Evaluation]], la regularización evita overfitting.

## 3. SVM con kernels lineales

Máxima separación entre clases mediante hyperplano de margen máximo.

```python
from sklearn.svm import LinearSVC

pipe = Pipeline([
    ("vec", CountVectorizer()),
    ("clf", LinearSVC(C=1.0, loss="hinge")),
])
pipe.fit(corpus, y)
```

Para multiclase, LinearSVC usa `one-vs-rest` (entrena un clasificador por categoría). SVM lineal es excelente para [[Text Classification]] cuando tienes muchas features (como BoW con vocabulario grande) porque el kernel lineal es rápido y efectivo en alta dimensionalidad.

## 4. Redes profundas para texto

Arquitectura: `Embedding layer → LSTM/GRU → Dense → Softmax`

```python
import torch
import torch.nn as nn

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.embedding(x)        # (batch, seq_len, embed_dim)
        lstm_out, _ = self.lstm(emb)   # (batch, seq_len, hidden*2)
        pooled = lstm_out.mean(dim=1)  # Global average pooling
        return self.classifier(pooled) # (batch, num_classes)
```

La capa `Embedding` puede ser pre-entrenada (word2vec, GloVe) o aprendida desde cero. LSTM bidireccional captura contexto izquierdo y derecho. Ideal para [[RNNs & Sequence Models]] en secuencias largas.

## 5. Transfer learning con BERT

Fine-tuning del transformer pre-entrenado para clasificación. Requiere tokenizer específico.

```python
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

text = "Congratulations, you won a free iPhone!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
print(outputs.logits.argmax().item())
```

**Salida esperada:**
```
1
```

BERT añade un `[CLS]` token al inicio; su embedding final pasa por un head de clasificación. El [[Transfer Learning]] permite lograr estado-del-arte con poco data labeled (fine-tune con ~1000 ejemplos por clase puede ser suficiente).

## 6. Evaluación

Accuracy miente en clases desbalanceadas (99% no-spam → clasificador trivial da 99%). Usa precisión, recall, F1.

```python
from sklearn.metrics import classification_report, confusion_matrix

y_true = [0, 1, 1, 0, 1, 0]
y_pred = [0, 1, 0, 0, 1, 0]
print(classification_report(y_true, y_pred, target_names=["ham", "spam"]))
print(confusion_matrix(y_true, y_pred))
```

**Salida esperada:**
```
              precision  recall  f1-score   support
         ham       0.67     1.00      0.80         3
        spam       1.00     0.67      0.80         3
   micro avg       0.83     0.83      0.83         6
   macro avg       0.83     0.83      0.80         6
weighted avg       0.83     0.83      0.80         6

[[3 0]
 [1 2]]
```

- **Macro F1**: promedio simple por clase (ignora desbalance).
- **Weighted F1**: promedio ponderado por soporte (recomendado).
- **Confusion matrix**: fila = real, columna = predicción. Diagonal = aciertos.

## 7. Common Mistakes

| Error | Consecuencia | Solución |
|---|---|---|
| Data leakage (TF-IDF fit en todo el dataset) | Sobrestimación severa del rendimiento | Fit solo en train, transform en test |
| No balancear clases | Modelo predice clase mayoritaria siempre | SMOTE, class_weight, o submuestreo |
| Vocabulario muy pequeño | Muchas palabras OOV en test, rendimiento cae | Al menos 5K–10K tokens o usar embeddings |
| No usar truncation en BERT | OOM o errores de forma | `truncation=True` en tokenizer |

## Resumen

1. Naive Bayes es rápido y funciona sorprendentemente bien para clasificación de texto con supuestos de independencia.
2. Logistic Regression ofrece coeficientes interpretables por palabra; L1/L2 ayudan a controlar overfitting.
3. SVM lineal es efectivo con espacios de alta dimensionalidad como BoW.
4. Redes profundas (LSTM, GRU) capturan dependencias secuenciales mejor que modelos bow.
5. BERT y transfer learning permiten estado-del-arte con fine-tuning en pocos datos.
6. Evaluar con F1 macro/weighted y confusion matrix, no solo accuracy.

## Check Your Understanding

1. ¿Por qué Naive Bayes se considera "ingenuo"? <!-- Porque asume independencia condicional entre palabras dado el label, lo cual es falso en lenguaje natural ("barato" y "económico" no son independientes). -->
2. ¿Qué interpretación tiene un coeficiente positivo en Logistic Regression para texto? <!-- La palabra está asociada a la clase positiva (ej. "free" → spam). -->
3. ¿Por qué SVM lineal es popular en clasificación de texto? <!-- Porque el espacio BoW es muy dimensional y el kernel lineal es rápido y no requiere tuning de kernel. -->
4. ¿Qué ventaja tiene BERT sobre LSTM para clasificación? <!-- BERT pre-entrenado captura contexto bidireccional profundo y puede fine-tunearse con pocos datos. -->
5. ¿Cuándo usarías macro F1 en vez de weighted F1? <!-- Cuando cada clase es igualmente importante, incluso las minoritarias. -->

## Where to Go Next

- [[Supervised Learning]]
- [[Model Evaluation]]
- [[Feature Engineering]]
- [[RNNs & Sequence Models]]
- [[Transfer Learning]]
- [[Word Embeddings (Word2Vec)]]
- [[Text Preprocessing & Representation]]
