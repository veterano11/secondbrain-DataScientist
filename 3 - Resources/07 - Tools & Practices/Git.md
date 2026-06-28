---
tags: [tools, git, foundational]
status: growing
created: 2026-06-27
---

# Git

## 1. Escenario de aprendizaje

Imagina que trabajas con tu equipo en un análisis de datos colaborativo. Sin control de versiones, compartir el progreso es un caos: aparecen archivos como "datos_limpios_final_v3_ok.ipynb", no puedes revertir cambios cuando algo se rompe y colaborar sin sobrescribir el trabajo de otros es casi imposible.

Con Git, cada cambio queda registrado, cada experimento queda vinculado a un estado específico del código y la colaboración a escala se vuelve posible.

---

## 2. Conceptos Fundamentales

### 2.1 Repositorio

Una carpeta `.git` en tu proyecto que registra todos los cambios.

```bash
git init      # crear un nuevo repo
git clone     # copiar un repo existente
```

### 2.2 Commit

Una foto de todos los archivos rastreados en un momento dado.

```bash
git add file.py        # preparar cambios para el commit
git commit -m "agregar pipeline de preprocesamiento"
```

Cada commit tiene:
- Un **hash** (identificador único: `abc123def456...`)
- Un **autor**, una **fecha** y un **mensaje**
- Un commit **padre** (estado anterior)

### 2.3 Rama (Branch)

Una línea de desarrollo paralela.

```bash
git branch feature-xyz    # crear una rama
git checkout feature-xyz  # cambiarse a ella
git checkout -b nueva-rama  # crear y cambiarse
```

**Rama principal**: normalmente `main` o `master` — la versión estable y publicable.

### 2.4 Remoto

Una copia del repositorio alojada en un servidor (GitHub, GitLab, Bitbucket).

```bash
git push origin main    # subir commits locales
git pull origin main    # descargar cambios remotos
git fetch origin        # descargar sin fusionar
```

---

## 3. Flujo de Trabajo Básico

### 3.1 Flujo Individual

Combínalo con herramientas de [[CLI & Productivity]] para un control de versiones eficiente.

```bash
# Iniciar trabajo
git checkout -b feature/nuevo-pipeline

# Trabajar y hacer commits sobre la marcha
git add src/pipeline.py
git commit -m "agregar paso de validación de datos"
git add src/pipeline.py
git commit -m "agregar paso de ingeniería de características"

# Fusionar a main al terminar
git checkout main
git merge feature/nuevo-pipeline
```

### 3.2 Flujo Colaborativo

```bash
# Obtener lo último
git checkout main
git pull origin main

# Crear rama de funcionalidad
git checkout -b feature/agregar-xgboost

# Trabajar, commitear, subir
git add . && git commit -m "agregar modelo XGBoost"
git push origin feature/agregar-xgboost

# Crear Pull Request en GitHub
# Un compañero revisa, aprueba y fusiona
```

---

## 4. Aspectos Específicos para Ciencia de Datos

### 4.1 Qué NO Subir al Repositorio

- Datos grandes (archivos CSV, Parquet)
- Checkpoints de modelos (`.pt`, `.h5`, `.pkl`)
- Salidas de Jupyter notebooks (`.ipynb` con resultados)
- Entornos virtuales (`.venv/`, `env/`)
- Archivos compilados (`__pycache__/`, `.pyc`)
- Claves de API, contraseñas, secretos

Usa **`.gitignore`** para excluir estos archivos automáticamente.

### 4.2 DVC (Data Version Control)

DVC extiende Git para rastrear datos y modelos:

```bash
dvc init                     # inicializar DVC
dvc add data/train.csv       # rastrear un archivo de datos
git add data/train.csv.dvc   # commitear el archivo puntero (no los datos reales)
git commit -m "agregar datos de entrenamiento"
dvc push                     # subir los datos reales al almacenamiento remoto
```

DVC guarda un pequeño archivo puntero en Git (que registra la versión de los datos) y los datos reales en un almacén remoto (S3, GCS, disco local). Versiona pipelines de datos junto con el código usando [[ML Pipelines]].

### 4.3 Git Amigable con Notebooks

Los notebooks de Jupyter son archivos JSON — hacer diff de ellos es tedioso (las celdas de salida cambian en cada ejecución).

**Soluciones**:
- **nbstripout**: elimina las salidas de los notebooks antes de commitear
- **Jupytext**: empareja `.ipynb` con archivos `.py` para diffs limpios
- Revisa notebooks en HTML con nbconvert, no en JSON crudo
- Usa las convenciones de [[Python for Data Science]] para flujos de trabajo compatibles con notebooks

---

## 5. Common Mistakes

1. **Committing large files**: a 5GB CSV in Git forever. Use DVC or Git LFS.

2. **Meaningless commit messages**: "fix", "update", "changes" — write messages that explain WHY.

3. **Working on main**: always create feature branches. main should always be deployable.

4. **Not pulling before pushing**: creates merge conflicts. Pull (or fetch + rebase) before push.

5. **Merging without review**: direct pushes to main skip code review. Use Pull Requests. Enforce [[Code Quality]] checks in CI before merging.

---

## 6. Check Your Understanding

1. What is the difference between `git pull` and `git fetch`? (Fetch downloads changes but does not merge. Pull = fetch + merge.)

2. A teammate commits "fixed bug". What information is missing? (Which bug? How was it fixed? What was the root cause?)

3. You accidentally committed a 2GB CSV file. What do you do? (Use `git filter-branch` or `BFG Repo Cleaner` to remove it from history.)

4. Why should a Jupyter notebook be stripped of outputs before committing? (Outputs change every run, creating noisy diffs. They also contain large base64 encoded images.)

5. main should always be deployable. What workflow ensures this? (Feature branches → code review → CI passes → merge to main.)

---

## 7. Summary

Git tracks changes, enables collaboration, and links code versions to experiment results. Use branches for features, write descriptive commit messages, exclude large files and outputs with .gitignore, and use DVC for data/model versioning. The golden rule: main should always be deployable.

---

## 8. Where to Go Next

- [[Experiment Tracking]] — Linking git commits to experiment runs
- [[Testing for Data Science]] — CI/CD with git-based workflows
- [[Virtual Environments]] — Reproducible environments
