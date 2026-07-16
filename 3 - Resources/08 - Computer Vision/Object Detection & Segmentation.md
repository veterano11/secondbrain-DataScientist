---
tags: [computer-vision, detection, segmentation]
status: growing
created: 2026-06-27
---

# Object Detection & Segmentation

## 1. Escenario de aprendizaje

Tenés una foto de una calle con autos, personas, y semáforos. Clasificación diría "esto es una calle". Pero necesitás saber **dónde** está cada objeto (detection) y **qué píxeles** ocupa (segmentation).

Detection y segmentation son la base de autos autónomos, diagnóstico médico por imágenes, robots, y búsqueda visual.

## 2. Classification vs Detection vs Segmentation

| Tarea | Entrada | Salida |
|-------|---------|--------|
| Classification | Imagen | "Perro" (una etiqueta) |
| Detection | Imagen | "Perro en (100,200,300,400)" (bbox) |
| Semantic seg. | Imagen | Cada píxel etiquetado (perro, fondo, césped) |
| Instance seg. | Imagen | Cada perro tiene su propia máscara |

## 3. Object Detection

### 3.1 Two-Stage Detectors (R-CNN family)

Primero proponen regiones candidatas, luego clasifican cada una.

```text
R-CNN       → Propuesta → Warp → Clasificar (lento: ~50s por imagen)
Fast R-CNN  → CNN una vez → RoI Pooling → Clasificar (rápido)
Faster R-CNN→ RPN aprende a proponer regiones (end-to-end)
Mask R-CNN  → + Rama de segmentación (máscara por objeto)
```

**Faster R-CNN** es el más usado. La **RPN (Region Proposal Network)** desliza ventanas sobre el feature map y predice si cada ancla contiene un objeto y cómo ajustar su bounding box.

```python
# Conceptual: una RPN predice objetos vs fondo
class RegionProposalNetwork(nn.Module):
    def __init__(self):
        self.conv = nn.Conv2d(256, 512, 3, padding=1)
        self.cls_logits = nn.Conv2d(512, 9 * 2, 1)   # 9 anchors × 2 clases
        self.bbox_pred = nn.Conv2d(512, 9 * 4, 1)     # 9 anchors × 4 coords

    def forward(self, x):
        t = F.relu(self.conv(x))
        return self.cls_logits(t), self.bbox_pred(t)
```

**Anchor boxes**: cajas de referencia de diferentes tamaños y aspect ratios. En cada posición del feature map, el modelo predice si cada ancla contiene un objeto.

### 3.2 One-Stage Detectors (YOLO)

Sin etapa de propuestas. Dividen la imagen en una grilla $S \times S$. Cada celda predice $B$ bounding boxes + confianza + $C$ probabilidades de clase.

```text
YOLOv3:  S=13, B=3, C=80 → output 13×13×255
         Una sola CNN. Un solo forward. Listo.
```

**Ventajas**: velocidad (30-200 FPS). **Desventaja**: ligeramente menor precisión en objetos pequeños.

**Innovaciones clave**:
- **Anchor boxes** con clustering (k-means sobre los bboxes del dataset)
- **FPN** (Feature Pyramid Network): predicciones en múltiples escalas
- **Focal Loss**: reduce el peso de ejemplos bien clasificados (combate desbalance entre foreground/background)

```python
# Focal Loss: penaliza menos los ejemplos fáciles
FL(p_t) = -α_t (1-p_t)^γ log(p_t)
# γ=0 → cross-entropy normal
# γ=2 → focus en ejemplos difíciles
```

### 3.3 DETR (Detection Transformer)

Transforma detection en un problema de **predicción de conjuntos**. Un encoder-decoder Transformer predice un conjunto fijo de N objetos. Hungarian Matching asigna cada predicción al ground truth.

```text
Sin anchors, sin NMS, sin RPN.
```

**Problema**: converge lento (500 epochs). Deformable DETR acelera con atención escasa.

### 3.4 Evaluación

**IoU (Intersection over Union)**:

$$\text{IoU} = \frac{\text{área de intersección}}{\text{área de unión}}$$

- $IoU \geq 0.5$: detección correcta
- $mAP@0.5$: mean Average Precision con IoU ≥ 0.5
- $mAP@0.5:0.95$: promedio de IoUs de 0.5 a 0.95 (COCO standard)

## 4. Segmentation

### 4.1 Semantic Segmentation

Cada píxel → clase. No distingue instancias.

**U-Net**: encoder (downsampling) + decoder (upsampling) con skip connections. Diseñado para datos biomédicos chicos. Funciona increíble con ~100 imágenes etiquetadas.

```python
# Skip connection en U-Net
class UNet(nn.Module):
    def forward(self, x):
        x1 = self.enc1(x)           # (B, 64, H, W)
        x2 = self.enc2(self.pool(x1)) # (B, 128, H/2, W/2)
        x3 = self.enc3(self.pool(x2)) # (B, 256, H/4, W/4)
        x = self.dec1(x3)           # (B, 128, H/2, W/2)
        x = torch.cat([x, x2], dim=1) # skip connection
        x = self.dec2(x)            # (B, 64, H, W)
        x = torch.cat([x, x1], dim=1) # skip connection
        return self.dec3(x)         # (B, n_classes, H, W)
```

**DeepLab**: usa convoluciones dilatadas (atrous) para aumentar el receptive field sin perder resolución.

### 4.2 Instance Segmentation

Cada objeto individual → su propia máscara.

**Mask R-CNN**: Faster R-CNN + rama paralela de máscara. Para cada RoI, predice una máscara binaria de $28 \times 28$.

**SAM (Segment Anything)**: modelo fundacional de Meta. Click en un punto → máscara del objeto. Cero-shot, sin fine-tuning.

```python
# SAM: prompt-based segmentation
predictor.set_image(imagen)
mascara, _, _ = predictor.predict(
    point_coords=np.array([[100, 200]]),
    point_labels=np.array([1])  # foreground
)
# Devuelve la máscara del objeto que contiene el punto (100, 200)
```

## 5. Pipeline completo: YOLO inference

```python
import cv2

model = YOLO('yolov8n.pt')
resultados = model('calle.jpg')

for r in resultados:
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = box.conf[0]
        cls = int(box.cls[0])
        print(f"{model.names[cls]}: {conf:.2f}, bbox=({int(x1)},{int(y1)},{int(x2)},{int(y2)})")
```

**Salida esperada:**
```
car: 0.95, bbox=(100,200,300,400)
person: 0.89, bbox=(400,500,450,600)
traffic light: 0.76, bbox=(600,100,620,150)
```

## 6. Errores Comunes

1. **Anchor boxes mal dimensionadas**: si tu dataset tiene objetos alargados y las anchors son cuadradas, las detecciones serán malas. Clusteriza los bboxes del dataset para redefinir anchors.
2. **NMS mal configurado**: threshold muy bajo → cajas duplicadas. Muy alto → objetos perdidos. 0.5 es típico.
3. **Objetos pequeños**: detectan muy mal sin FPN. Usá YOLOv8 o DETR con multi-scale features.
4. **Dataset desbalanceado**: la mayoría de las celdas en YOLO son fondo. Focal Loss ayuda.

## 7. Verifica tu Comprensión

1. ¿Por qué YOLO es más rápido que Faster R-CNN? (Una sola pasada vs propuesta + clasificación)
2. ¿Qué ventaja tiene DETR sobre YOLO? (No necesita anchors ni NMS; es más simple conceptualmente)
3. ¿Cuándo usarías SAM en vez de entrenar tu propio segmentador? (Cuando no tenés datos etiquetados y necesitas segmentación general)
4. ¿Qué problema resuelve FPN? (Objetos pequeños se detectan mal sin multi-scale features)

## 8. Resumen

Detection localiza objetos con bounding boxes (YOLO para velocidad, Faster R-CNN para precisión, DETR para simplicidad). Segmentation etiqueta píxeles (semántica: U-Net, DeepLab; instancias: Mask R-CNN, SAM). La evaluación usa mAP con IoU. Las herramientas modernas (YOLOv8, SAM) hacen que detection y segmentation sean accesibles desde pocas líneas de código.

## 9. Dónde Ir Ahora

- [[Image Processing Fundamentals]] — la base clásica
- [[Vision Transformers]] — DETR y arquitecturas attention-based
- [[Self-Supervised & Multimodal Vision]] — SAM y modelos fundacionales
- [[Generative Models for Vision]] — generación de máscaras y layouts
- [[CNNs]] — backbone de detectores como Faster R-CNN y YOLO
- [[Transfer Learning]] — fine-tuning de backbones pre-entrenados
- [[Training Techniques]] — entrenamiento de detectores y segmentadores
