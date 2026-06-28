---
tags: [tools, productivity, foundational]
status: growing
created: 2026-06-27
---

# CLI & Productivity

## 1. Why This Matters

The command line is the most powerful interface for a data scientist. GUI tools hide complexity; the CLI exposes it. Every time you use a GUI to do something the CLI could do in one command, you are wasting time.

Data science is fundamentally about automating data processing — and the CLI is the ultimate automation interface. A well-crafted one-liner can replace hours of manual work.

---

## 2. Essential CLI Tools

### 2.1 File Operations

| Tool | What It Does | Example |
|---|---|---|
| `ls` | List files | `ls -la` (detailed listing) |
| `find` | Find files by name/type | `find . -name "*.csv"` |
| `fd` | Fast find (alternative) | `fd csv` |
| `grep / rg` | Search file contents | `rg "accuracy" *.py` |
| `head / tail` | View file start/end | `tail -f log.txt` (follow) |

### 2.2 Data Processing

| Tool | What It Does | Example |
|---|---|---|
| `jq` | Query JSON | `cat data.json \| jq '.results[].name'` |
| `csvkit` | CSV processing | `csvstat data.csv` (summary stats) |
| `awk` | Column processing | `awk '{print $1, $3}' file.csv` |
| `sort / uniq` | Sort and deduplicate | `sort file.csv \| uniq -c` |

### 2.3 Monitoring

| Tool | What It Does | Example |
|---|---|---|
| `htop` | Process monitor | See CPU/memory per process |
| `du` | Disk usage | `du -sh *` (dir sizes) |
| `df` | Disk free | `df -h` (free space) |

---

## 3. Productivity Workflow

### 3.1 Shell Aliases

Add these to `~/.zshrc` (or `~/.bashrc`):

```bash
# Git shortcuts
alias gs="git status"
alias gc="git commit -m"
alias gp="git push"
alias gl="git log --oneline --graph"

# Python
alias python="python3"
alias pip="pip3"
alias jl="jupyter lab"

# Data science
alias df="python -c 'import pandas as pd; pd.read_csv(\"$1\").info()'"
```

### 3.2 One-Liner Examples

```bash
# Find largest CSV files
find . -name "*.csv" -exec du -h {} \; | sort -rh | head -10

# Count lines of Python code per directory
find . -name "*.py" | xargs wc -l | sort -rn

# Search for a pattern in notebooks (stripping output)
jq '.cells[] | select(.cell_type == "code") | .source[]' notebook.ipynb | rg "train"

# Check GPU usage (Linux)
watch -n 1 nvidia-smi

# Extract column from CSV and compute statistics
cut -d, -f3 data.csv | sort -n | awk '{sum+=$1; n++} END {print sum/n}'
```

---

## 4. Python CLI Tools

### 4.1 Click / Typer

Build CLI interfaces for your data scripts (see [[Python for Data Science]] for language context):

```python
# cli.py
import typer
app = typer.Typer()

@app.command()
def train(
    data_path: str = typer.Argument(..., help="Path to training data"),
    model: str = typer.Option("rf", help="Model type"),
    lr: float = typer.Option(0.001, help="Learning rate"),
):
    """Train a model on the specified data."""
    print(f"Training {model} on {data_path} with lr={lr}")

if __name__ == "__main__":
    app()
```

```bash
python cli.py train data.csv --model xgboost --lr 0.01
```

### 4.2 Rich

Beautiful terminal output:

```python
from rich.console import Console
from rich.progress import track
from rich.table import Table

console = Console()
console.print("[bold green]Training complete![/]")

# Progress bar
for i in track(range(100), description="Training..."):
    time.sleep(0.01)

# Table
table = Table(title="Model Comparison")
table.add_column("Model", style="cyan")
table.add_column("Accuracy", style="green")
table.add_row("Random Forest", "0.92")
table.add_row("XGBoost", "0.94")
console.print(table)
```

---

## 5. Automation

| Tool | When to Use | Example |
|---|---|---|
| **Make** | Run tasks with dependencies | `make train`, `make test` |
| **Just** | Simpler Make alternative | `just train` |
| **Cron** | Scheduled execution | Run training every night at 2 AM |
| **launchd** | macOS task scheduler | Mac equivalent of cron |

Combine with [[Testing for Data Science]] to run tests via `make test`.

### Makefile Example

```makefile
.PHONY: train test clean

train:
	python src/train.py --data data/processed/ --model models/

test:
	pytest tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
```

---

## 6. Notebooks vs Scripts

| Aspect | Jupyter Notebook | Python Script |
|---|---|---|
| **Exploration** | Excellent | Poor |
| **Reproducibility** | Poor (state, output) | Excellent |
| **Version control** | Poor (JSON, outputs) | Excellent |
| **Automation** | Poor | Excellent |
| **Debugging** | Moderate | Excellent |

**Best practice**: explore in notebooks, productionize in scripts. Use `nbconvert` to extract scripts from notebooks.

---

## 7. Common Mistakes

1. **Not using aliases**: repeating long commands wastes hours per month. Alias everything you type more than once.

2. **Manual file operations**: "I will rename these 50 files manually" — write a one-liner.

3. **Ignoring the terminal**: GUIs are convenient for exploration; the CLI is faster for everything else.

4. **Not knowing `jq`**: working with JSON APIs (common in ML) without jq is painfully slow.

5. **One-off scripts without CLI arguments**: hardcoding file paths in scripts means editing the script every time. Use Click/Typer for arguments.

---

## 8. Check Your Understanding

1. How do you find all CSV files larger than 1GB? (`find . -name "*.csv" -size +1G`)

2. Your training script hardcodes the data path. What is the problem? (You must edit the script to train on different data. Use a command-line argument instead.)

3. You run `python train.py` every night at 2 AM. How do you automate this? (Cron job: `0 2 * * * cd ~/project && python train.py`)

4. What is the advantage of `make` over a shell script? (Make tracks file dependencies — it only re-runs steps if inputs changed.)

5. Why is `jq` useful for data scientists? (Many ML APIs return JSON. jq lets you query, filter, and transform JSON from the command line.)

---

## 9. Summary

The CLI is the most powerful tool in a data scientist's arsenal. Learn `find`, `grep`/`rg`, `jq`, and `awk` for data processing. Use `htop`/`du` for monitoring. Build CLI tools with Click or Typer. Automate with Make and cron. The GUI is for exploration; the CLI is for automation.

---

## 10. Where to Go Next

- [[Git]] — CLI for version control
- [[Virtual Environments]] — CLI for environment management
- [[Code Quality]] — CLI tools for linting and formatting
- [[Experiment Tracking]] — CLI for logging experiment metadata
- [[ML Pipelines]] — Automating pipeline orchestration from the CLI
