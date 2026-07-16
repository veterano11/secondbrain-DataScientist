---
tags: [mathematics, statistics, foundational]
status: growing
created: 2026-06-27
---


# Estadística

## 1. Escenario de aprendizaje

Entrenas un modelo y obtienes un 92% de precisión. ¿Es bueno? ¿Cómo sabes que no es solo suerte? Pruebas dos modelos diferentes y uno rinde 1% mejor. ¿Es una mejora real? Despliegas un modelo y al mes siguiente la precisión cae al 88%. ¿Es una degradación real o solo ruido aleatorio?

Sin herramientas estadísticas, estas preguntas son imposibles de responder. Necesitas intervalos de confianza, pruebas de hipótesis y comprensión del sesgo-varianza para separar la señal del ruido. Si la probabilidad es la matemática de la incertidumbre, la estadística es la práctica de **tomar decisiones con datos a pesar de la incertidumbre**.

La estadística te da las herramientas para responder estas preguntas. Separa la señal del ruido. Sin ella, estás adivinando.

---

## 2. Estadística descriptiva — Resumiendo datos

Antes de hacer inferencias, necesitas **describir** lo que ves.

### 2.1 Medidas de tendencia central

**Media** (promedio): $\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$
- Sensible a valores atípicos. Un solo multimillonario en una sala de 100 personas eleva drásticamente el ingreso medio.

**Mediana**: el valor del medio cuando los datos están ordenados.
- Robusta a valores atípicos. La mediana del ingreso apenas cambia si agregas un multimillonario.

**Moda**: el valor más frecuente.
- Útil para datos categóricos. "La categoría de producto más común es Electrónicos."

**¿Cuándo usar cada una?**
- Distribución normal → media (estimador más eficiente)
- Datos sesgados (ingresos, precios de casas) → mediana
- Datos categóricos → moda

### 2.2 Medidas de dispersión

**Varianza**: $\sigma^2 = \frac{1}{n}\sum (x_i - \bar{x})^2$
- Distancia cuadrática promedio desde la media.

**Desviación estándar**: $\sigma = \sqrt{\sigma^2}$
- En las mismas unidades que los datos. Si la altura tiene media 170cm y desviación estándar 10cm, la mayoría de las personas están entre 160-180cm.

**Rango Intercuartil (IQR)**: Q3 - Q1 (percentil 75 - percentil 25)
- Robusto a valores atípicos. Contiene el 50% central de los datos.

**Por qué importa la dispersión**: dos conjuntos de datos pueden tener la misma media pero dispersiones muy diferentes. Un modelo entrenado con datos de alta varianza puede necesitar más regularización.

### 2.3 Forma

- **Asimetría (Skewness)**: falta de simetría. Sesgo positivo → cola larga a la derecha (como los ingresos). Sesgo negativo → cola larga a la izquierda.
- **Curtosis**: pesadez de las colas. Curtosis alta → más valores atípicos. Los rendimientos financieros tienen alta curtosis (colas gruesas).

---

## 3. Estadística inferencial — Extrayendo conclusiones de muestras

Nunca tienes todos los datos (la población). Tienes una **muestra**. La estadística te dice qué puedes inferir sobre la población a partir de esa muestra.

### 3.1 Estimación

**Estimación puntual**: una mejor estimación única para un parámetro poblacional.
- Ejemplo: la media muestral $\bar{x}$ es una estimación puntual de la media poblacional $\mu$.

**Intervalo de confianza**: un rango que probablemente contiene el valor verdadero.

$$IC = \hat{\theta} \pm z_{\alpha/2} \cdot EE$$

- $\hat{\theta}$: estimación puntual
- $EE$: error estándar (desviación estándar de la distribución muestral)
- $z_{\alpha/2}$: valor crítico (1.96 para 95% de confianza)

**Interpretación**: "Si repitiéramos este experimento muchas veces, el 95% de los intervalos de confianza contendrían el verdadero parámetro poblacional."

Esto NO significa "hay un 95% de probabilidad de que el valor verdadero esté en este intervalo." El valor verdadero está o no está en el intervalo. El 95% se refiere al **procedimiento**, no al intervalo específico.

### 3.2 Pruebas de hipótesis

**El marco**:

1. **$H_0$ (hipótesis nula)**: la suposición predeterminada (sin efecto, sin diferencia)
2. **$H_1$ (hipótesis alternativa)**: lo que quieres probar (hay un efecto)

**Ejemplo**:
- $H_0$: el nuevo modelo tiene la misma precisión que el modelo anterior
- $H_1$: el nuevo modelo tiene mayor precisión

3. **Elige el nivel de significancia $\alpha$** (típicamente 0.05)
4. **Calcula un estadístico de prueba** y su **valor p**
5. **Decisión**: si $p < \alpha$, rechaza $H_0$ (tienes evidencia para $H_1$)

### 3.3 El valor p

**Definición**: la probabilidad de observar datos tan extremos como los tuyos (o más extremos) **asumiendo que $H_0$ es cierta**.

**Qué significa p < 0.05**: "Si verdaderamente no hubiera efecto, veríamos datos tan extremos como estos menos del 5% de las veces."

**Interpretaciones erróneas comunes**:
- Incorrecto: "Hay un 95% de probabilidad de que el efecto sea real"
- Incorrecto: "Los valores p te dicen el tamaño del efecto"
- Correcto: "Los valores p te dicen qué tan sorprendentes serían tus datos bajo $H_0$"

### 3.4 Errores Tipo I y Tipo II

| Decisión | $H_0$ es cierta | $H_0$ es falsa |
|---|---|---|
| **Rechazar $H_0$** | Error Tipo I (falso positivo) | ¡Correcto! |
| **No rechazar $H_0$** | ¡Correcto! | Error Tipo II (falso negativo) |

- **Tasa de Tipo I** = $\alpha$ (tú controlas esto — usualmente 0.05)
- **Tasa de Tipo II** = $\beta$
- **Poder** = $1 - \beta$ (probabilidad de detectar un efecto real)

En términos de ML:
- Falso positivo: desplegar un modelo que en realidad no mejora
- Falso negativo: NO desplegar un modelo que SÍ habría mejorado
- Poder: qué tan probable es que tu prueba A/B detecte una mejora real

### 3.5 Pruebas estadísticas comunes

| Prueba | Qué compara | Cuándo usarla |
|---|---|---|
| **t-test** | Medias de dos grupos | Prueba A/B, datos aproximadamente normales |
| **ANOVA** | Medias de 3+ grupos | Comparación de múltiples modelos |
| **Chi-cuadrado** | Distribuciones categóricas | Prueba de independencia de características |
| **Mann-Whitney U** | Medianas de dos grupos | Datos no normales, muestras pequeñas |
| **KS test** | Distribuciones completas | Detección de deriva de datos |

---

## 4. El equilibrio Sesgo-Varianza

Este es el **concepto central** que conecta la estadística con [[Supervised Learning]].

### 4.1 Definiciones

- **Sesgo (Bias)**: error por asumir que la forma del modelo es más simple que la realidad. Un modelo lineal en datos no lineales tiene alto sesgo.
- **Varianza (Variance)**: error por sensibilidad a fluctuaciones en los datos de entrenamiento. Un árbol de decisión profundo tiene alta varianza.

$$E[(y - \hat{f}(x))^2] = \text{Sesgo}^2 + \text{Varianza} + \text{Error irreducible}$$

### 4.2 El equilibrio

- **Modelo simple** (regresión lineal): alto sesgo, baja varianza
- **Modelo complejo** (árbol profundo): bajo sesgo, alta varianza
- **El objetivo**: encontrar el punto óptimo donde el error total se minimiza

**Descripción visual**: imagina disparar flechas a un blanco.
- Alto sesgo: todas las flechas agrupadas en el lugar equivocado (fuera del centro)
- Alta varianza: todas las flechas dispersas aleatoriamente (algunas cerca, otras lejos)
- Ideal: todas las flechas agrupadas en el centro (bajo sesgo, baja varianza)

### 4.3 Consecuencias en la práctica

- **Subajuste (Underfitting)** (alto sesgo): el modelo es demasiado simple. El error de entrenamiento es alto. Solución: más características, modelo más complejo.
- **Sobreajuste (Overfitting)** (alta varianza): el modelo memorizó los datos de entrenamiento. El error de entrenamiento es bajo pero el error de validación es alto. Solución: regularización, más datos, modelo más simple.

La curva de aprendizaje te dice qué problema tienes:
- Si las curvas de entrenamiento y validación convergen pero ambas son altas → alto sesgo (necesitas un modelo más expresivo)
- Si hay una gran brecha entre entrenamiento y validación → alta varianza (necesitas regularización o más datos)

---

## 5. Correlación y Causalidad

### 5.1 Correlación

$$\rho_{X,Y} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

- Varía de -1 a 1
- 0 significa que no hay relación lineal
- ±1 significa relación lineal perfecta

### 5.2 Correlación ≠ Causalidad

Ejemplo clásico: las ventas de helado y los incidentes de ahogamiento están correlacionados. ¿El helado causa ahogamiento? No. Ambos son causados por el clima cálido (un **factor de confusión**).

En ML: un modelo podría aprender que "ventas de paraguas" predice "lluvia", pero las ventas de paraguas no causan lluvia. Esto importa cuando despliegas el modelo en un nuevo entorno donde la correlación podría romperse.

### 5.3 Paradoja de Simpson

Una tendencia que aparece en varios grupos pero desaparece o se invierte cuando los grupos se combinan.

**Ejemplo**: caso de sesgo de género en UC Berkeley. En 1973, UC Berkeley fue demandada por sesgo de género: en general, los hombres eran admitidos a una tasa más alta que las mujeres. Sin embargo, al examinar departamentos individuales, la mayoría de los departamentos tenían tasas de admisión **iguales o más altas** para las mujeres. La paradoja fue causada porque las mujeres solicitaban ingreso a departamentos más competitivos (con tasas de admisión más bajas en general).

**Conclusión**: siempre verifica las variables de confusión. El rendimiento de tu modelo de ML puede ocultar sesgos que solo aparecen al segmentar los datos.

---

## 6. Pensamiento estadístico en ML

| Concepto de ML | Paralelo estadístico |
|---|---|
| Pérdida de entrenamiento | Error muestral (qué tan bien ajustas los datos) |
| Pérdida de validación | Error fuera de muestra (qué tan bien generalizas) |
| Regularización L2 | Prior Bayesiano (pesos ~ Normal(0, 1/λ)) |
| Regularización L1 | Prior Bayesiano (pesos ~ Laplace(0, 1/λ)) |
| Validación cruzada | Muestreo repetido para estimar el error de generalización |
| Métodos de ensemble | Reducción de varianza promediando múltiples estimadores |
| Descenso por gradiente | Optimización para estimación MLE/MAP |

---

## 7. Errores Comunes

1. **p-hacking**: correr muchas pruebas y reportar solo las significativas. Si pruebas 20 características para significancia con $\alpha=0.05$, una parecerá significativa por casualidad sola.

2. **Ignorar comparaciones múltiples**: si comparas 10 modelos y eliges el mejor, la probabilidad de que el mejor sea una sobreestimación es mucho mayor que 5%. Corregir con Bonferroni o FDR.

3. **Confundir significancia práctica y estadística**: un resultado puede ser estadísticamente significativo (p < 0.05) pero prácticamente irrelevante (mejora de 0.001%).

4. **No verificar supuestos**: los t-tests asumen normalidad. La regresión lineal asume homocedasticidad. Violar supuestos puede invalidar conclusiones.

5. **Sesgo de supervivencia**: analizar solo los casos exitosos. En ML: evaluar modelos solo en datos que llegaron a producción ignora los fracasos, que contienen información valiosa.

---

## 8. Verifica tu Comprensión

1. Test A/B: modelo A tiene 94.2% de precisión, modelo B tiene 94.5% de precisión, p = 0.04. ¿Qué puedes concluir?
2. Tu modelo tiene alta precisión de entrenamiento pero baja precisión de prueba. ¿Es esto sesgo o varianza? ¿Qué haces?
3. ¿Por qué la validación cruzada da una mejor estimación del error de generalización que una sola división entrenamiento/prueba?
4. Encuentras una correlación de -0.9 entre años de experiencia y errores. ¿Puedes concluir que más experiencia causa menos errores?
5. Un intervalo de confianza del 95% para la tasa de clics es [0.032, 0.038]. ¿Esto significa que hay un 95% de probabilidad de que la CTR real esté entre 3.2% y 3.8%?

---

## 9. Resumen

La estadística es el puente entre los datos y las decisiones. La estadística descriptiva resume lo que ves. La estadística inferencial te dice qué puedes concluir más allá de tus datos. Las pruebas de hipótesis te ayudan a separar la señal del ruido. La compensación sesgo-varianza es el concepto unificador que conecta la estadística con ML — explica el sobreajuste, el subajuste, y por qué la complejidad del modelo debe ser cuidadosamente controlada.

---

## 10. Dónde Ir Ahora

- [[Probabilidad]] — La base matemática de la estadística
- [[Evaluación de Modelos]] — Validación cruzada, métricas y cómo medir rendimiento
- [[Pruebas A/B]] — Aplicar pruebas de hipótesis para comparar modelos en producción
- [[Ingeniería de Características]] — Por qué el escalado, la codificación y la selección importan
- [[Inferencia Bayesiana]] — Enfoques bayesianos vs frecuentistas de la estadística
- [[Álgebra Lineal]] — Covarianza, PCA y las matemáticas detrás de los métodos estadísticos
