---
tags: [tools, productivity, foundational]
status: growing
created: 2026-06-27
---

# CLI & Productivity

## 1. Escenario de aprendizaje

Necesitas procesar 200 archivos CSV, calcular estadísticas por grupo, filtrar filas con ciertas condiciones y generar un reporte final. Hacerlo manualmente en un notebook te tomaría horas abriendo archivo por archivo. Con una línea de comando bien construida, el mismo trabajo se hace en segundos.

La línea de comandos es la interfaz más poderosa para un científico de datos. Las interfaces gráficas ocultan la complejidad; la CLI la expone. Cada vez que usas una GUI para algo que la CLI puede hacer en un solo comando, estás perdiendo tiempo.

---

## 2. Herramientas CLI Esenciales

### 2.1 Operaciones con Archivos

| Herramienta | Qué hace | Ejemplo |
|---|---|---|
| `ls` | Listar archivos | `ls -la` (listado detallado) |
| `find` | Buscar archivos por nombre/tipo | `find . -name "*.csv"` |
| `fd` | Find rápido (alternativa) | `fd csv` |
| `grep / rg` | Buscar contenido en archivos | `rg "accuracy" *.py` |
| `head / tail` | Ver inicio/final de archivo | `tail -f log.txt` (seguir) |

### 2.2 Procesamiento de Datos

| Herramienta | Qué hace | Ejemplo |
|---|---|---|
| `jq` | Consultar JSON | `cat data.json \| jq '.results[].name'` |
| `csvkit` | Procesamiento de CSV | `csvstat data.csv` (estadísticas resumidas) |
| `awk` | Procesamiento de columnas | `awk '{print $1, $3}' file.csv` |
| `sort / uniq` | Ordenar y deduplicar | `sort file.csv \| uniq -c` |

### 2.3 Monitoreo

| Herramienta | Qué hace | Ejemplo |
|---|---|---|
| `htop` | Monitor de procesos | Ver CPU/memoria por proceso |
| `du` | Uso de disco | `du -sh *` (tamaños de directorios) |
| `df` | Espacio libre en disco | `df -h` (espacio disponible) |

---

## 3. Flujo de Trabajo Productivo

### 3.1 Alias del Shell

Agrega estos a `~/.zshrc` (o `~/.bashrc`):

```bash
# Atajos de Git
alias gs="git status"
alias gc="git commit -m"
alias gp="git push"
alias gl="git log --oneline --graph"

# Python
alias python="python3"
alias pip="pip3"
alias jl="jupyter lab"

# Ciencia de datos
alias df="python -c 'import pandas as pd; pd.read_csv(\"$1\").info()'"
```

### 3.2 Ejemplos de One-Liners

```bash
# Encontrar los archivos CSV más grandes
find . -name "*.csv" -exec du -h {} \; | sort -rh | head -10

# Contar líneas de código Python por directorio
find . -name "*.py" | xargs wc -l | sort -rn

# Buscar un patrón en notebooks (eliminando salidas)
jq '.cells[] | select(.cell_type == "code") | .source[]' notebook.ipynb | rg "train"

# Ver uso de GPU (Linux)
watch -n 1 nvidia-smi

# Extraer columna de CSV y calcular estadísticas
cut -d, -f3 data.csv | sort -n | awk '{sum+=$1; n++} END {print sum/n}'
```

---

## 4. Herramientas CLI de Python

### 4.1 Click / Typer

Construye interfaces CLI para tus scripts de datos (consulta [[Python for Data Science]] para contexto del lenguaje):

```python
# cli.py
import typer
app = typer.Typer()

@app.command()
def train(
    data_path: str = typer.Argument(..., help="Ruta a los datos de entrenamiento"),
    model: str = typer.Option("rf", help="Tipo de modelo"),
    lr: float = typer.Option(0.001, help="Tasa de aprendizaje"),
):
    """Entrena un modelo con los datos especificados."""
    print(f"Entrenando {model} con {data_path} y lr={lr}")

if __name__ == "__main__":
    app()
```

```bash
python cli.py train data.csv --model xgboost --lr 0.01
```

### 4.2 Rich

Salida de terminal elegante:

```python
from rich.console import Console
from rich.progress import track
from rich.table import Table

console = Console()
console.print("[bold green]¡Entrenamiento completado![/]")

# Barra de progreso
for i in track(range(100), description="Entrenando..."):
    time.sleep(0.01)

# Tabla
table = Table(title="Comparación de Modelos")
table.add_column("Modelo", style="cyan")
table.add_column("Accuracy", style="green")
table.add_row("Random Forest", "0.92")
table.add_row("XGBoost", "0.94")
console.print(table)
```

---

## 5. Automatización

| Herramienta | Cuándo usarla | Ejemplo |
|---|---|---|
| **Make** | Ejecutar tareas con dependencias | `make train`, `make test` |
| **Just** | Alternativa más simple a Make | `just train` |
| **Cron** | Ejecución programada | Entrenar cada noche a las 2 AM |
| **launchd** | Programador de tareas de macOS | Equivalente a cron en Mac |

Combínalo con [[Testing for Data Science]] para ejecutar tests mediante `make test`.

### Ejemplo de Makefile

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

| Aspecto | Jupyter Notebook | Script de Python |
|---|---|---|
| **Exploración** | Excelente | Malo |
| **Reproducibilidad** | Mala (estado, salidas) | Excelente |
| **Control de versiones** | Malo (JSON, salidas) | Excelente |
| **Automatización** | Mala | Excelente |
| **Depuración** | Moderada | Excelente |

**Mejor práctica**: explora en notebooks, produce en scripts. Usa `nbconvert` para extraer scripts a partir de notebooks.

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

## 9. Resumen

The CLI is the most powerful tool in a data scientist's arsenal. Learn `find`, `grep`/`rg`, `jq`, and `awk` for data processing. Use `htop`/`du` for monitoring. Build CLI tools with Click or Typer. Automate with Make and cron. The GUI is for exploration; the CLI is for automation.

---

## 10. Where to Go Next

- [[Git]] — CLI for version control
- [[Virtual Environments]] — CLI for environment management
- [[Code Quality]] — CLI tools for linting and formatting
- [[Experiment Tracking]] — CLI for logging experiment metadata
- [[ML Pipelines]] — Automating pipeline orchestration from the CLI
