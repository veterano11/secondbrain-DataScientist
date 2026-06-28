---
tags: [computer-vision, generative, gans, diffusion]
status: growing
created: 2026-06-27
---

# Generative Models for Vision

## 1. Escenario de aprendizaje

Querés generar imágenes de rostros realistas para un dataset sintético. O querés tomar una foto borrosa y hacerla nítida. O querés crear una escena 3D desde fotos 2D.

Los modelos generativos aprenden la **distribución** de las imágenes, no solo una función de mapeo. Esto permite crear imágenes nuevas, completar imágenes dañadas, o generar vistas 3D desde cualquier ángulo.

## 2. ¿Qué significa "generar"?

Un clasificador aprende $P(y|x)$: dada una imagen, ¿qué etiqueta tiene?

Un generador aprende $P(x)$ o $P(x|y)$: la probabilidad de que una imagen exista. Una vez aprendida, podemos **muestrear** de esa distribución para crear imágenes nuevas.

### 2.1 Tres familias principales

| Familia | Cómo genera | Dónde está hoy |
|---------|-------------|----------------|
| **VAE** | Codifica imagen a latente, decodifica latente a imagen | Base para Stable Diffusion (latent space) |
| **GAN** | Dos redes compiten: generador vs discriminador | Obsoleto para texto→imagen; útil para estilo |
| **Diffusion** | Agrega ruido gradual, aprende a revertirlo | Estado del arte (DALL-E, Stable Diffusion, Midjourney) |

## 3. Generative Adversarial Networks (GANs)

Dos redes en competencia:

- **Generador** $G(z)$: recibe ruido aleatorio $z$, produce imagen
- **Discriminador** $D(x)$: predice si $x$ es real ($\approx 1$) o generado ($\approx 0$)

```text
Ruido z → Generador → Imagen fake → Discriminador → ¿real o fake?
           ↑                           ↑
        aprende a engañar           aprende a detectar
```

### 3.1 La función de pérdida

$$\min_G \max_D \mathbb{E}_{x \sim \text{datos}}[\log D(x)] + \mathbb{E}_{z \sim \text{ruido}}[\log(1 - D(G(z)))]$$

$D$ maximiza (mejor detectando fakes). $G$ minimiza (mejor engañando a $D$).

### 3.2 En la práctica: DCGAN

```python
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0),  # 4×4
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),          # 8×8
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),          # 16×16
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 3, 4, 2, 1),            # 32×32
            nn.Tanh(),
        )

    def forward(self, z):
        z = z.view(-1, 100, 1, 1)
        return self.net(z)
```

**Problemas clásicos de GANs:**
- **Mode collapse**: el generador descubre una imagen que engaña al discriminador y solo genera variaciones mínimas de esa misma imagen
- **Inestabilidad**: el equilibrio G-D es frágil; un paso malo puede colapsar el entrenamiento
- **Medición**: no hay una métrica objetiva como log-likelihood

## 4. Diffusion Models

### 4.1 Intuición

Imaginá que tenés una foto nítida. Agregás ruido Gaussiano una vez, dos veces, ... hasta que es ruido puro (100 steps aprox). El modelo diffusion aprende a **revertir** ese proceso: parte de ruido puro y va eliminando el ruido paso a paso hasta obtener una imagen limpia.

### 4.2 Forward process (agregar ruido)

$$x_t = \sqrt{1-\beta_t} x_{t-1} + \sqrt{\beta_t} \varepsilon_{t-1}$$

Después de $T$ pasos, $x_T \sim \mathcal{N}(0, I)$ — ruido puro.

### 4.3 Reverse process (generar)

Aprendemos una red $\varepsilon_\theta(x_t, t)$ que predice el ruido agregado en el paso $t$.

$$\mathcal{L} = \mathbb{E}_{t, x_0, \varepsilon} \left[ \|\varepsilon - \varepsilon_\theta(x_t, t)\|^2 \right]$$

```python
def sample(model, T=1000, img_size=64):
    # Empezar con ruido puro
    x = torch.randn(1, 3, img_size, img_size)

    for t in reversed(range(T)):
        epsilon_pred = model(x, t)           # predecir ruido
        noise = torch.randn_like(x) if t > 0 else 0
        x = denoise(x, epsilon_pred, t, noise)

    return x  # imagen generada
```

### 4.4 Stable Diffusion (Latent Diffusion)

El problema de diffusion en pixel space: es lentísimo para imágenes grandes (256×256 × 3 canales). Stable Diffusion corre el proceso de diffusion en el **espacio latente** de un VAE pre-entrenado.

```text
Imagen (512×512×3)
    ↓ VAE Encoder
Latente (64×64×4)    ← aquí corre la diffusion (64× más chico)
    ↓ VAE Decoder
Imagen (512×512×3)
```

Esto hace que training e inference sean órdenes de magnitud más rápidos.

## 5. NeRF: De fotos 2D a escenas 3D

NeRF representa una escena 3D como una red neuronal: dadas coordenadas 3D $(x,y,z)$ y una dirección de vista $(\theta, \phi)$, predice color RGB y densidad $\sigma$.

```python
def nerf_forward(x, y, z, theta, phi):
    # Positional encoding (mapear a frecuencias altas)
    pos_enc = positional_encoding(x, y, z)
    view_enc = positional_encoding(theta, phi)

    # MLP: densidad desde posición
    h = mlp_density(pos_enc)
    density = h[0]

    # Color condicionado a dirección de vista
    color = mlp_color(torch.cat([h, view_enc]))

    return color, density  # (R,G,B) y qué tan sólido es este punto
```

**Limitación**: entrenar un NeRF por escena toma horas. No generaliza a escenas no vistas. Gaussian Splatting (2023) es más rápido y mejor calidad.

## 6. VAEs (Variational Autoencoders)

El VAE aprende un espacio latente estructurado: codifica la imagen a una distribución (media y varianza), muestrea un punto de esa distribución, y decodifica a imagen.

$$\mathcal{L} = \underbrace{-\mathbb{E}_{z \sim q_\phi}[\log p_\theta(x|z)]}_{\text{reconstrucción}} + \underbrace{D_{KL}(q_\phi(z|x) \| p(z))}_{\text{regularización}}$$

**VAE vs GAN vs Diffusion**:

| Propiedad | VAE | GAN | Diffusion |
|-----------|-----|-----|-----------|
| Calidad de imagen | Borrosa | Buena | Excelente |
| Diversidad | Buena | Mode collapse | Buena |
| Velocidad de generación | Rápida | Rápida | Lenta (muchos pasos) |
| Log-likelihood trazable | Sí | No | Sí |
| Complejidad de entrenamiento | Baja | Alta | Media |

## 7. Common Mistakes

1. **Mode collapse en GANs**: el generador produce solo 1-2 tipos de imágenes. Soluciones: minibatch discrimination, spectral normalization, o cambiarse a diffusion.
2. **Color shifting en diffusion**: errores acumulados durante sampling producen imágenes decoloradas. Usar classifier-free guidance con scale bien calibrado (7.5 para Stable Diffusion).
3. **NeRF muy lento**: NeRF vanilla tarda minutos por vista. Para aplicaciones interactivas usá Instant NGP (hash grids) o Gaussian Splatting.
4. **VAE blur**: la reconstrucción borrosa es intrínseca al modelo (el término KL empuja a latentes suaves). Para imágenes nítidas, diffusion es mejor.

## 8. Check Your Understanding

1. En una GAN, si el discriminador se vuelve perfecto muy rápido, ¿qué pasa con el generador? (El gradiente se vuelve 0 — el generador deja de aprender)
2. Diffusion requiere 50-1000 pasos para generar una imagen. ¿Cómo hace Stable Diffusion para generar en segundos? (Corre la diffusion en el espacio latente, que es 64× más chico)
3. ¿Por qué NeRF necesita positional encoding de alta frecuencia? (Las redes profundas tienden a aprender frecuencias bajas; el encoding fuerza a la red a aprender detalles finos)
4. ¿En qué caso usarías un VAE en vez de diffusion? (Cuando necesitás un espacio latente estructurado para interpolación o exploración controlada)

## 9. Summary

Tres familias de modelos generativos compiten hoy. GANs (dos redes compitiendo) fueron el estándar 2018-2022 pero sufren mode collapse. Diffusion models (agregar y remover ruido gradualmente) son el estado del arte para generación de imágenes. VAEs ofrecen un espacio latente estructurado pero imágenes borrosas. NeRF extiende la generación a 3D. Stable Diffusion combina VAE + Diffusion para generar imágenes de alta calidad en segundos.

## 10. Where to Go Next

- [[Self-Supervised & Multimodal Vision]] — CLIP guidance para texto→imagen
- [[CNNs]] — backbone de discriminadores GAN y encoder-decoder
- [[Transfer Learning]] — fine-tuning de modelos generativos pre-entrenados
- [[Image Processing Fundamentals]] — pre y post-procesamiento de imágenes generadas
- [[Vision Transformers]] — ViT como backbone en diffusion models
- [[Unsupervised Learning]] — modelos generativos como paradigma no supervisado
- [[Training Techniques]] — estabilización de entrenamiento GAN/diffusion
