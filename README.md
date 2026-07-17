# Second Brain — Data Science & Machine Learning

Un vault de **Obsidian** con **180+ notas educativas** que cubren el stack completo de Data Science: desde fundamentos de programación y matemáticas hasta producción de modelos, MLOps, infraestructura y gobernanza. Cada nota es **autocontenida, tutorial-style** (Gen 3): escenario concreto, código con outputs esperados, errores comunes, ejercicios de verificación y enlaces a notas relacionadas.

---

## Contenido (26 secciones, ~35,000 líneas)

### Nivel 1: Fundamentos
| Sección | Notas | Cubre |
|---------|-------|-------|
| **Programming** | 6 | Python, DS&A, OOP, FP, NumPy/Pandas, **Polars** |
| **Mathematics** | 4 | Álgebra Lineal, Cálculo, Probabilidad, Estadística |
| **Databases & SQL** | 5 | Modelo relacional, Joins, Window Functions, Optimización, **NoSQL** |
| **Data Visualization** | 5 | Percepción visual, matplotlib, seaborn, plotly/altair, Storytelling |
| **Tools & Practices** | 5 | Git, Testing, Virtualenv, CLI, Code Quality |

### Nivel 2: Machine Learning
| Sección | Notas | Cubre |
|---------|-------|-------|
| **Machine Learning** | 9 | Supervised, Unsupervised, Eval, Regularization, FE, **Ensemble (XGBoost/LGBM/CAT)**, **Anomaly Detection**, **DimRed (t-SNE/UMAP)**, **Advanced FE** |
| **Deep Learning** | 6 | NN, CNNs, RNNs, Transformers, Training Techniques, Transfer Learning |
| **Optimization** | 4 | Convex, Gradient-Based, Hyperparameter Tuning (**Optuna**), Constraint |
| **Advanced Statistics** | 6 | Experimental Design, Causal Inference, Bootstrap, Bayesian Stats, Non-parametric, **Causal ML (DoWhy)** |
| **Information Theory** | 6 | Information Theory, Graph Theory, Numerical Methods, Adv. Linear Algebra, Optimization Theory, **Graph Neural Networks** |

### Nivel 3: Dominios especializados
| Sección | Notas | Cubre |
|---------|-------|-------|
| **Computer Vision** | 5 | Image Processing, Object Detection, GANs, Vision Transformers, Multimodal |
| **NLP** | 5 | Preprocessing, Word2Vec, Text Classification, NER, Topic Modeling |
| **LLMs** | 6 | Transformers, Prompt Engineering, Fine-tuning, RAG, Eval, Agentic Systems |
| **Time Series** | 4 | Fundamentals, Classical, Deep Learning, Anomaly Detection |
| **Reinforcement Learning** | 5 | RL Fundamentals, Value/Policy-Based, Model-Based, **RLHF** |
| **Bayesian Methods** | 4 | Bayesian Inference, Probabilistic Programming, MCMC, Gaussian Processes |
| **Recommender Systems** | 5 | CF, Content-Based, Hybrid/Session, Production, MAB |

### Nivel 4: Ingeniería y producción
| Sección | Notas | Cubre |
|---------|-------|-------|
| **Data Engineering** | 6 | ETL, Airflow, Kafka, Warehousing, dbt, Data Quality |
| **Infrastructure & DevOps** | 5 | IaC, Terraform (Fundamentos, State, Modules), AWS CodeBuild |
| **Containers & Orchestration** | 6 | Docker, Compose, K8s, K8s+ML, Helm, CI/CD + GitOps |
| **Software Engineering** | 5 | Design Patterns, API Design (FastAPI), System Design, Clean Code, Testing |
| **MLOps & Observability** | 6 | Pipelines, Monitoring, Drift, **MLflow**, A/B Testing, Observability |
| **ML in Production** | 5 | Feature Stores, Model Serving, Deploy Strategies, Compression, Continuous Training |

### Nivel 5: Negocio y gobernanza
| Sección | Notas | Cubre |
|---------|-------|-------|
| **Data Product & Analytics** | 5 | Product Thinking, Funnel/Cohort, Experimentation, Stakeholder Comms, Self-Serve |
| **Responsible AI** | 4 | Bias & Fairness, **SHAP/LIME**, Privacy (DP, Federated), ML Governance |
| **Data Security & Compliance** | 4 | Data Classification, Encryption, GDPR/CCPA/EU AI Act, Secure ML |

---

## Estructura del vault

```
Brain/
├── 1 - Projects/          → Proyectos activos
├── 2 - Areas/              → Responsabilidades continuas
├── 3 - Resources/          → 26 secciones de conocimiento (156 notas de contenido)
├── 4 - Archives/           → Items inactivos
├── Meta/                   → Dashboard y navegación
├── Daily/                  → Notas diarias
├── Templates/              → Plantillas para notas
└── mcp_server/             → Servidor MCP para opencode
```

---

## Características

- **Todas en español** — monolingüe, consistente
- **Formato tutorial Gen 3**: cada nota tiene escenario de aprendizaje, código con outputs esperados, errores comunes, check-your-understanding, y enlaces a notas relacionadas
- **180+ notas interconectadas** con enlaces internos wiki
- **Ruta de lectura recomendada** en Meta/_index.md (5 niveles de dificultad ascendente)
- **Stack completo**: de `print("hello")` a producción de modelos con Kubernetes y CI/CD
- **Listo para Obsidian** — configurado con `.obsidian/` y templates

---

## Cómo usar

1. Clona el repo: `git clone https://github.com/veterano11/secondbrain-DataScientist.git`
2. Abrí la carpeta como vault en Obsidian (File → Open vault → Open folder as vault)
3. Empezá por `Meta/_index.md` para ver la ruta de lectura recomendada
4. Seguí el orden secuencial: Fundamentos → ML → Dominios → Producción → Negocio

### Orden de lectura recomendado

**Nivel 1 — Fundamentos** (primero)
1. Programming
2. Mathematics
3. Databases & SQL
4. Data Visualization
5. Tools & Practices

**Nivel 2 — Machine Learning**
6. Machine Learning
7. Deep Learning
8. Optimization
9. Advanced Statistics
10. Information Theory

**Nivel 3 — Dominios especializados**
11. Computer Vision
12. NLP
13. LLMs
14. Time Series
15. Reinforcement Learning
16. Bayesian Methods
17. Recommender Systems

**Nivel 4 — Ingeniería y producción**
18. Data Engineering
19. Infrastructure & DevOps
20. Containers & Orchestration
21. Software Engineering
22. MLOps & Observability
23. ML in Production

**Nivel 5 — Negocio y gobernanza** (último)
24. Data Product & Analytics
25. Responsible AI
26. Data Security & Compliance

---

## Stack tecnológico cubierto

| Categoría | Herramientas |
|-----------|-------------|
| **Lenguajes** | Python, SQL, Bash |
| **ML/DL** | scikit-learn, XGBoost, LightGBM, CatBoost, PyTorch, TensorFlow |
| **LLMs** | OpenAI API, HuggingFace, LangChain, RAG |
| **MLOps** | MLflow, DVC, Weights & Biases |
| **Visualización** | matplotlib, seaborn, plotly, altair |
| **Infraestructura** | Docker, K8s, Terraform, AWS, Helm, ArgoCD |
| **Data Engineering** | Airflow, Kafka, dbt, Spark, Feast |
| **Optimización** | Optuna, SHAP, LIME |

---

## Estadísticas

- **180 notas totales**
- **26 secciones** de conocimiento
- **~35,000 líneas** de contenido educativo
- **100% tutorial-style** (código + explicación + ejercicios)
- **Todas en español**
- **Versionado en git** desde el día 1

---

*Built with opencode + Obsidian + dedicación.*
