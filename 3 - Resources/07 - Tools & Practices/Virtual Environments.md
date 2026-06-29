---
tags: [tools, python, foundational]
status: growing
created: 2026-06-27
---

# Virtual Environments

## 1. Escenario de aprendizaje

Trabajas en dos proyectos de ciencia de datos. El primero necesita `scikit-learn==1.2` por compatibilidad con un pipeline legacy; el segundo requiere `scikit-learn==1.5` para usar las últimas funcionalidades. Sin entornos aislados, solo puedes tener una versión instalada a nivel de sistema —uno de los dos proyectos se rompe.

Los entornos virtuales resuelven este conflicto creando espacios aislados de Python por proyecto, cada uno con sus propias dependencias sin interferencias.

---

## 2. El Problema

```
Proyecto A: necesita pandas 1.5, numpy 1.24
Proyecto B: necesita pandas 2.0, numpy 1.26

Sin entornos: solo se puede instalar UNA versión
→ El proyecto A o B se rompe
```

**Solución**:
```
Proyecto A: .venv/A/ → pandas 1.5, numpy 1.24
Proyecto B: .venv/B/ → pandas 2.0, numpy 1.26
Cada entorno está aislado.
```

---

## 3. Herramientas

### 3.1 venv (Incorporado)

Combínalo con [[CLI & Productivity]] y sus alias para cambios rápidos de entorno.

```bash
# Crear entorno
python -m venv .venv

# Activar
source .venv/bin/activate    # Linux/Mac
.venv\Scripts\activate       # Windows

# Instalar paquetes
pip install pandas scikit-learn

# Registrar dependencias
pip freeze > requirements.txt

# Reproducir en otra máquina
pip install -r requirements.txt

# Desactivar
deactivate
```

**Cuándo usarlo**: proyectos pequeños, dependencias simples, cuando no quieres herramientas externas.

### 3.2 Conda

```bash
# Crear entorno
conda create -n myenv python=3.11

# Activar
conda activate myenv

# Instalar
conda install pandas scikit-learn

# Exportar
conda env export > environment.yml

# Reproducir
conda env create -f environment.yml
```

**Ventajas**: maneja dependencias no Python (CUDA, librerías C), ideal para stacks de ciencia de datos.

**Desventajas**: más lento, más pesado, índice de paquetes distinto al de PyPI.

### 3.3 Poetry

```bash
# Inicializar
poetry new myproject
poetry init

# Agregar dependencias
poetry add pandas scikit-learn

# Instalar desde lockfile
poetry install

# Compilar y publicar
poetry build
poetry publish
```

**Ventajas**: builds deterministas (lockfile), resolución de dependencias, sistema de empaquetado.

**Desventajas**: curva de aprendizaje, resolución más lenta.

### 3.4 uv

```bash
# Crear entorno
uv venv

# Instalar
uv pip install pandas scikit-learn

# Sincronizar desde requirements
uv pip sync requirements.txt
```

**Ventajas**: 10–100 veces más rápido que pip, moderno, compatible con el formato de requirements de pip.

---

## 4. Entornos Reproducibles

### 4.1 Archivos de Requirements

```
# requirements.in (dependencias directas, sin versiones fijas)
pandas
scikit-learn
torch

# requirements.txt (versiones fijas con hashes, generado)
pandas==2.0.3 --hash=sha256:abc123...
scikit-learn==1.3.0 --hash=sha256:def456...
```

**Generar con**:
```bash
pip-compile requirements.in  # pip-tools
uv pip compile requirements.in -o requirements.txt  # uv
```

### 4.2 Ubicación del Entorno Virtual

Convenciones comunes:
- `.venv/` en la raíz del proyecto (práctica actual recomendada)
- `venv/`
- `env/`

Agrega el entorno virtual a `.gitignore`:
```
# .gitignore
.venv/
venv/
env/
```

---

## 5. Buenas Prácticas

1. **Usa siempre un entorno virtual**: sin excepciones. Los paquetes del sistema son solo para herramientas del sistema.

2. **Incluye requirements.txt o lockfile en el repositorio**: cualquiera que clone el repo puede reproducir el entorno. Usa herramientas de [[Code Quality]] para validar los archivos de dependencias.

3. **Usa Python 3.11+**: las versiones anteriores (3.7, 3.8) llegaron al final de su vida útil y carecen de funcionalidades modernas.

4. **Un entorno por proyecto**: no compartas entornos entre proyectos.

5. **Documenta la configuración**: un README debería explicar cómo crear y activar el entorno.

---

## 6. Common Mistakes

1. **Not activating the environment**: `pip install pandas` installs to the system Python, not your project. Always check `which python`.

2. **Committing the virtual environment**: `.venv/` contains platform-specific binaries. Add it to `.gitignore`.

3. **Not pinning versions**: `pip freeze` generates a list without version constraints. Use `pip-compile` or Poetry for pinned, reproducible builds.

4. **Using `pip` without a virtual environment**: eventually breaks your system Python.

5. **Mixing pip and conda**: can cause hard-to-diagnose conflicts. Use one or the other (pip inside conda is fine; conda inside pip is not).

---

## 7. Check Your Understanding

1. Your project uses `python 3.11` and `scikit-learn 1.3`. Your teammate has `python 3.9` and `scikit-learn 1.0`. How do you ensure the same environment? (Create a `requirements.txt` or `environment.yml` and share it.)

2. Why should you NOT commit `.venv/` to git? (It contains platform-specific binaries, is large, and can be regenerated from requirements.)

3. What is the difference between `requirements.in` and `requirements.txt`? (.in lists direct dependencies; .txt pins exact versions with hashes for reproducibility.)

4. You run `pip install` and it says "permission denied". What is the most likely cause? (You are not in a virtual environment and pip is trying to write to system directories.)

5. Conda handles non-Python dependencies. Give an example where this matters. (CUDA toolkit, libgcc, OpenBLAS — these are C/C++ libraries needed by numpy and torch.)

---

## 8. Resumen

Virtual environments isolate Python dependencies per project. venv is built-in and sufficient for most cases. Conda handles non-Python dependencies. Poetry and uv provide deterministic builds. Always use an environment, commit the dependency file, and add `.venv/` to `.gitignore`. The rule: if you are not using a virtual environment, you are doing it wrong.

---

## 9. Where to Go Next

- [[Git]] — Version control + environments for reproducibility
- [[Python Fundamentals]] — Python basics before setting up environments
- [[Testing for Data Science]] — Testing across environments
- [[ML Pipelines]] — Reproducible pipeline environments
