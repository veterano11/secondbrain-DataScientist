---
tags: [tools, python, foundational]
status: growing
created: 2026-06-27
---

# Virtual Environments

## 1. Why This Matters

Every data science project depends on a specific set of packages at specific versions. One project needs `scikit-learn==1.2`, another needs `scikit-learn==1.5`. Without isolation, these requirements conflict — you can only have one version installed system-wide. See [[Python for Data Science]] for language-level context.

Virtual environments solve this by creating **isolated Python environments** per project. Each environment has its own packages, independent of every other environment and the system Python.

---

## 2. The Problem

```
Project A: needs pandas 1.5, numpy 1.24
Project B: needs pandas 2.0, numpy 1.26

Without environments: can only install ONE version
→ Project A or B breaks
```

**Solution**:
```
Project A: .venv/A/ → pandas 1.5, numpy 1.24
Project B: .venv/B/ → pandas 2.0, numpy 1.26
Each environment is isolated.
```

---

## 3. Tools

### 3.1 venv (Built-in)

Combine with [[CLI & Productivity]] aliases for faster environment switching.

```bash
# Create environment
python -m venv .venv

# Activate
source .venv/bin/activate    # Linux/Mac
.venv\Scripts\activate       # Windows

# Install packages
pip install pandas scikit-learn

# Record dependencies
pip freeze > requirements.txt

# Reproduce on another machine
pip install -r requirements.txt

# Deactivate
deactivate
```

**When to use**: small projects, simple dependencies, when you want zero external tools.

### 3.2 Conda

```bash
# Create environment
conda create -n myenv python=3.11

# Activate
conda activate myenv

# Install
conda install pandas scikit-learn

# Export
conda env export > environment.yml

# Reproduce
conda env create -f environment.yml
```

**Advantages**: handles non-Python dependencies (CUDA, C libraries), good for data science stacks.

**Disadvantages**: slower, larger, different package index than PyPI.

### 3.3 Poetry

```bash
# Initialize
poetry new myproject
poetry init

# Add dependencies
poetry add pandas scikit-learn

# Install from lockfile
poetry install

# Build and publish
poetry build
poetry publish
```

**Advantages**: deterministic builds (lockfile), dependency resolution, build system.

**Disadvantages**: learning curve, slower resolution.

### 3.4 uv

```bash
# Create environment
uv venv

# Install
uv pip install pandas scikit-learn

# Sync from requirements
uv pip sync requirements.txt
```

**Advantages**: 10-100× faster than pip, modern, compatible with pip's requirements format.

---

## 4. Reproducible Environments

### 4.1 Requirements Files

```
# requirements.in (direct dependencies, no versions pinned)
pandas
scikit-learn
torch

# requirements.txt (pinned with hashes, generated)
pandas==2.0.3 --hash=sha256:abc123...
scikit-learn==1.3.0 --hash=sha256:def456...
```

**Generate with**:
```bash
pip-compile requirements.in  # pip-tools
uv pip compile requirements.in -o requirements.txt  # uv
```

### 4.2 Virtual Environment Location

Common conventions:
- `.venv/` in the project root (current best practice)
- `venv/`
- `env/`

Add the virtual environment to `.gitignore`:
```
# .gitignore
.venv/
venv/
env/
```

---

## 5. Best Practices

1. **Always use a virtual environment**: no exceptions. System-wide packages are for system tools only.

2. **Commit requirements.txt or lockfile**: anyone cloning the repo can reproduce the environment. Use [[Code Quality]] tools to validate dependency files.

3. **Use Python 3.11+**: older versions (3.7, 3.8) are end-of-life and missing features.

4. **One environment per project**: do not share environments across projects.

5. **Document the setup**: a README should say how to create and activate the environment.

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

## 8. Summary

Virtual environments isolate Python dependencies per project. venv is built-in and sufficient for most cases. Conda handles non-Python dependencies. Poetry and uv provide deterministic builds. Always use an environment, commit the dependency file, and add `.venv/` to `.gitignore`. The rule: if you are not using a virtual environment, you are doing it wrong.

---

## 9. Where to Go Next

- [[Git]] — Version control + environments for reproducibility
- [[Python Fundamentals]] — Python basics before setting up environments
- [[Testing for Data Science]] — Testing across environments
- [[ML Pipelines]] — Reproducible pipeline environments
