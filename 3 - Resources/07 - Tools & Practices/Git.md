---
tags: [tools, git, foundational]
status: growing
created: 2026-06-27
---

# Git

## 1. Why This Matters

Git is the standard for version control in software engineering and data science. It tracks changes to code (and with DVC, data), enables collaboration, and provides a safety net for experimentation.

Without Git, you have:
- "model_final_v3_final_actually_final.ipynb"
- No way to revert when you break something
- No way to collaborate without overwriting each other's work

With Git, every change is tracked, every experiment is linked to a specific code state, and collaboration scales.

---

## 2. Core Concepts

### 2.1 Repository

A `.git` folder in your project that tracks all changes.

```bash
git init      # create a new repo
git clone     # copy an existing repo
```

### 2.2 Commit

A snapshot of all tracked files at a point in time.

```bash
git add file.py        # stage changes for commit
git commit -m "add preprocessing pipeline"
```

Each commit has:
- A **hash** (unique identifier: `abc123def456...`)
- An **author**, **date**, and **message**
- A **parent** commit (previous state)

### 2.3 Branch

A parallel line of development.

```bash
git branch feature-xyz    # create a branch
git checkout feature-xyz  # switch to it
git checkout -b new-branch  # create and switch
```

**Main branch**: typically `main` or `master` — the stable, deployable version.

### 2.4 Remote

A hosted copy of the repo (GitHub, GitLab, Bitbucket).

```bash
git push origin main    # upload local commits
git pull origin main    # download remote changes
git fetch origin        # download without merging
```

---

## 3. Basic Workflow

### 3.1 Individual Workflow

Combine Git with [[CLI & Productivity]] tools for efficient version control.

```bash
# Start work
git checkout -b feature/new-pipeline

# Work, commit as you go
git add src/pipeline.py
git commit -m "add data validation step"
git add src/pipeline.py
git commit -m "add feature engineering step"

# Merge to main when done
git checkout main
git merge feature/new-pipeline
```

### 3.2 Collaborative Workflow

```bash
# Get latest
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/add-xgboost

# Work, commit, push
git add . && git commit -m "add XGBoost model"
git push origin feature/add-xgboost

# Create Pull Request on GitHub
# Teammate reviews, approves, merges
```

---

## 4. Data Science Specifics

### 4.1 What NOT to Commit

- Large datasets (CSV, Parquet files)
- Model checkpoints (`.pt`, `.h5`, `.pkl`)
- Jupyter notebook outputs (`.ipynb` with outputs)
- Virtual environments (`.venv/`, `env/`)
- Compiled files (`__pycache__/`, `.pyc`)
- API keys, passwords, secrets

Use **`.gitignore`** to exclude these automatically.

### 4.2 DVC (Data Version Control)

DVC extends Git to track data and models:

```bash
dvc init                     # initialize DVC
dvc add data/train.csv       # track a data file
git add data/train.csv.dvc   # commit the pointer file (not the actual data)
git commit -m "add training data"
dvc push                     # upload actual data to remote storage
```

DVC stores a small pointer file in Git (tracking the data version) and the actual data in a remote store (S3, GCS, local drive). Version data pipelines alongside code with [[ML Pipelines]].

### 4.3 Notebook-Friendly Git

Jupyter notebooks are JSON files — diffing them is painful (output cells change every time).

**Solutions**:
- **nbstripout**: strips notebook outputs before committing
- **Jupytext**: pairs `.ipynb` with `.py` files for clean diffs
- Review notebooks in nbconvert HTML, not in raw JSON
- Use [[Python for Data Science]] conventions for notebook-friendly workflows

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
