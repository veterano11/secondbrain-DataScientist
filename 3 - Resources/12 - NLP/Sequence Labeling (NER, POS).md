---
tags:
  - nlp
  - sequence-labeling
  - ner
  - pos
status: seedling
created: 2026-06-28
---

## Escenario de aprendizaje

En un texto médico, necesitas extraer: "El paciente [PERSONA] fue diagnosticado con [ENFERMEDAD] y recetó [MEDICAMENTO]". Esto es Named Entity Recognition (NER). Similar a POS tagging pero con entidades del dominio. Trabajamos con artículos médicos para extraer enfermedades, medicamentos y síntomas usando [[Supervised Learning]].

## 1. POS Tagging

Partes del habla (Part-of-Speech): etiquetar cada palabra como sustantivo, verbo, adjetivo, etc. El tagset estándar es el Penn Treebank (~45 etiquetas).

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The patient was diagnosed with pneumonia")

for token in doc:
    print(f"{token.text:12} {token.pos_:6} {token.tag_:6} {token.dep_:10}")
```

**Salida esperada:**
```
The          DET    DT     det
patient      NOUN   NN     nsubjpass
was          AUX    VBD    auxpass
diagnosed    VERB   VBN    ROOT
with         ADP    IN     prep
pneumonia    PROPN  NNP    pobj
```

POS tagging es un paso previo común para [[Text Preprocessing & Representation|extraer features]] antes de NER, y mejora el rendimiento cuando se usa como feature en CRF.

## 2. Named Entity Recognition (NER)

Identifica personas, organizaciones, locaciones, fechas, cantidades monetarias, etc. spaCy incluye un modelo pre-entrenado para 18 categorías.

```python
doc = nlp("Apple Inc. was founded by Steve Jobs in Cupertino on April 1, 1976.")

for ent in doc.ents:
    print(f"{ent.text:25} {ent.label_:10} {ent.start_char}-{ent.end_char}")
```

**Salida esperada:**
```
Apple Inc.                 ORG        0-10
Steve Jobs                 PERSON     25-35
Cupertino                  GPE        39-48
April 1, 1976              DATE       52-66
```

Cada entidad tiene un label, un inicio y un fin en el texto original. Para dominios específicos (medicina, leyes) se necesitan entidades personalizadas o fine-tuning con [[Transfer Learning]].

## 3. CRF (Conditional Random Fields)

Modelo probabilístico que considera dependencias entre etiquetas vecinas. A diferencia de clasificar cada token independientemente, CRF modela la secuencia completa.

```python
import sklearn_crfsuite
from sklearn_crfsuite import metrics

# Features: palabra, POS, sufijo, prefijo, mayúscula
def word2features(sent, i):
    word = sent[i][0]
    return {
        "word": word,
        "word.lower": word.lower(),
        "is_upper": word[0].isupper(),
        "is_digit": word.isdigit(),
        "suffix_3": word[-3:],
        "prefix_2": word[:2],
    }

# CRF entiende que "New" (B-LOC) → "York" (I-LOC) es más probable que "New" (B-LOC) → "York" (O)
crf = sklearn_crfsuite.CRF(
    algorithm="lbfgs",
    c1=0.1,  # L1 regularization
    c2=0.1,  # L2 regularization
    max_iterations=100,
)
```

CRF fue estado-del-arte para NER antes de las redes profundas. Las features de contexto (palabra anterior, siguiente, POS) son críticas. La regularización evita overfitting en [[Model Evaluation|corpus pequeños]].

## 4. BiLSTM-CRF

Combina un LSTM bidireccional (aprende representaciones contextuales) con una capa CRF (modela dependencias entre etiquetas).

```python
import torch
import torch.nn as nn
from torchcrf import CRF

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embed_dim=100, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, x, tags=None):
        emb = self.embedding(x)
        lstm_out, _ = self.bilstm(emb)
        emissions = self.hidden2tag(lstm_out)
        if tags is not None:
            return -self.crf(emissions, tags)  # Negative log-likelihood
        return self.crf.decode(emissions)
```

El BiLSTM aprende representaciones contextuales de cada token mirando toda la secuencia. La capa CRF fuerza transiciones válidas (B-LOC no puede seguir a I-PER). Esta arquitectura fue el estándar pre-BERT en [[RNNs & Sequence Models]].

## 5. spaCy: agregar entidades personalizadas

spaCy permite añadir entidades nuevas al pipeline sin reentrenar todo.

```python
from spacy.tokens import Span

nlp = spacy.load("en_core_web_sm")
doc = nlp("The patient was given ibuprofen for arthritis")

# Agregar entidad personalizada
span = Span(doc, 4, 5, label="MEDICINE")  # ibuprofen
doc.set_ents([span], default="unmodified")

for ent in doc.ents:
    print(f"{ent.text:15} {ent.label_}")
```

**Salida esperada:**
```
ibuprofen       MEDICINE
arthritis       DISEASE (si spaCy ya lo reconoce)
```

Para dominios no cubiertos por modelos pre-entrenados, puedes entrenar un pipeline NER propio con `spacy train` usando datos anotados en formato [[Word Embeddings (Word2Vec)|JSONL o spaCy binario]].

## 6. Evaluación

NER se evalúa a nivel de span (no de token): una entidad es correcta solo si coincide exactamente el texto y el label.

```python
from seqeval.metrics import classification_report, f1_score

y_true = [["O", "B-PER", "I-PER", "O", "B-LOC", "O"]]
y_pred = [["O", "B-PER", "I-PER", "O", "B-LOC", "O"]]

print(classification_report(y_true, y_pred))
```

**Salida esperada:**
```
              precision    recall  f1-score   support
         PER       1.00      1.00      1.00         1
         LOC       1.00      1.00      1.00         1
```

**BIO tagging**: B-begin, I-inside, O-outside. Alternativa: BILOU (B-begin, I-inside, L-last, O-outside, U-unit). BILOU es más granular y suele dar mejor rendimiento. En la evaluación se usa `seqeval`, que compara spans completos (no tokens individuales).

## 7. Common Mistakes

| Error | Consecuencia | Solución |
|---|---|---|
| Context window insuficiente | El modelo no ve palabras clave lejanas | CRF con features amplias o LSTM bidireccional |
| Ignorar etiqueta O (outside) | Desbalance extremo (~90% tokens son O) | Submuestreo o weighted loss |
| Evaluar token-level en vez de span-level | Una entidad parcial cuenta como acierto | Usar seqeval (span-based F1) |
| No normalizar entidades | "Dr. Smith" y "Smith" como entidades distintas | Post-procesamiento o merging de spans |

## Resumen

1. POS tagging etiqueta cada palabra con su categoría gramatical (sustantivo, verbo, etc.) y sirve como feature para NER.
2. NER identifica entidades nombradas (personas, orgs, locaciones) y se puede extender a dominios personalizados.
3. CRF modela dependencias entre etiquetas adyacentes; BiLSTM-CRF añade representaciones contextuales profundas.
4. spaCy ofrece un pipeline NER pre-entrenado y entrenable para datos propios.
5. La evaluación span-based (seqeval) es más estricta que token-level y es el estándar en la literatura.
6. BIO/BILOU son esquemas de etiquetado secuencial para marcar límites de entidades.

## Comprueba tu Conocimiento

1. ¿Qué diferencia hay entre POS tagging y NER? <!-- POS etiqueta categorías gramaticales (sustantivo, verbo); NER identifica entidades del mundo real (personas, lugares). -->
2. ¿Por qué CRF es mejor que clasificar cada token independientemente? <!-- Porque modela dependencias entre etiquetas vecinas: "New York" es una entidad, no dos independientes. -->
3. ¿Qué significan B, I, O en BIO tagging? <!-- B = begin (inicio de entidad), I = inside (dentro de entidad), O = outside (fuera de entidad). -->
4. ¿Cómo se evalúa NER correctamente? <!-- Con span-based F1 (seqeval): una entidad es correcta solo si coinciden texto exacto y label. -->
5. ¿Qué ventaja tiene BILOU sobre BIO? <!-- BILOU añade L (last) y U (unit) para entidades de un solo token, dando más información al modelo. -->

## ¿Dónde ir Siguente?

- [[RNNs & Sequence Models]]
- [[Text Preprocessing & Representation]]
- [[Supervised Learning]]
- [[Model Evaluation]]
- [[Word Embeddings (Word2Vec)]]
- [[Transfer Learning]]
