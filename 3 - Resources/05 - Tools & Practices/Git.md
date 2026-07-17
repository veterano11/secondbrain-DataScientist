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

## 5. Errores Comunes

1. **Hacer commit de archivos grandes**: un CSV de 5GB en Git para siempre. Usar DVC o Git LFS.

2. **Mensajes de commit sin significado**: "fix", "update", "changes" — escribe mensajes que expliquen POR QUÉ.

3. **Trabajar en main**: siempre crear ramas de funcionalidad. main siempre debe ser desplegable.

4. **No hacer pull antes de push**: crea conflictos de fusión. Pull (o fetch + rebase) antes de push.

5. **Fusión sin revisión**: los pushes directos a main saltan la revisión de código. Usar Pull Requests. Aplicar verificaciones de [[Calidad de Código]] en CI antes de fusionar.

---

## 6. Verifica tu Comprensión

1. ¿Cuál es la diferencia entre `git pull` y `git fetch`? (Fetch descarga cambios pero no fusiona. Pull = fetch + merge.)

2. Un compañero hace commit de "fixed bug". ¿Qué información falta? (¿Qué bug? ¿Cómo se arregló? ¿Cuál fue la causa raíz?)

3. Haces accidentalmente commit de un archivo CSV de 2GB. ¿Qué haces? (Usar `git filter-branch` o `BFG Repo Cleaner` para eliminarlo del historial.)

4. ¿Por qué se deben eliminar las salidas de un Jupyter notebook antes de hacer commit? (Las salidas cambian en cada ejecución, creando diffs ruidosos. También contienen imágenes grandes codificadas en base64.)

5. main siempre debe ser desplegable. ¿Qué flujo de trabajo asegura esto? (Ramas de funcionalidad → revisión de código → CI pasa → fusión a main.)

---

## 7. Resumen

Git rastrea cambios, permite la colaboración, y vincula versiones de código con resultados de experimentos. Usa ramas para funcionalidades, escribe mensajes de commit descriptivos, excluye archivos grandes y salidas con .gitignore, y usa DVC para versionado de datos/modelos. La regla de oro: main siempre debe ser desplegable.

---

## 8. Dónde Ir Ahora

- [[Seguimiento de Experimentos]] — Vincular commits de git con ejecuciones de experimentos
- [[Testing para Ciencia de Datos]] — CI/CD con flujos de trabajo basados en git
- [[Entornos Virtuales]] — Entornos reproducibles
