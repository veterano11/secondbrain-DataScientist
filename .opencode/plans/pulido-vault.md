# Plan de pulido — Second Brain

## Objetivo
Pulir el vault entero para formato profesional: sin emojis decorativos, secciones consistentes, idioma unificado (espanol), y estructura estandarizada.

## Tareas

### 1. README.md — remover emojis decorativos
- 7 emojis en titulos: 🧠 📚 🏗️ 🎯 🚀 🔗 📈
- Reemplazar por texto plano profesional

### 2. Notas con emojis decorativos (8 archivos)
Reemplazar emojis por texto equivalente:

| Archivo | Linea | Emoji | Reemplazo |
|---------|-------|-------|-----------|
| `14 - Databases & SQL/Advanced Querying.md` | 104 | 💡 en blockquote | `Nota:` |
| `14 - Databases & SQL/Relational Model & SQL Fundamentals.md` | 40 | ▶ en diagrama | `>` |
| `14 - Databases & SQL/Relational Model & SQL Fundamentals.md` | 199 | 💡 en blockquote | `Nota:` |
| `15 - Data Visualization/Visualization Fundamentals.md` | 192 | ❌/✅ en tabla | `No / Si` |
| `02 - Mathematics/Statistics.md` | 102-104 | ❌/✅ como bullet | Texto plano |
| `03 - Machine Learning/Ensemble Methods.md` | 199 | ⚠️ admonicion | `Advertencia:` |
| `06 - MLOps & Observability/ML Pipelines.md` | 45-48 | ✓ en lista | Texto plano |
| `14 - Databases & SQL/Query Optimization & Indexing.md` | 29 | ▶ en diagrama | `>` |

### 3. Agregar Resumen a 7 notas que faltan
Notas sin seccion Summary/Resumen:
- `03 - Resources/10 - Reinforcement Learning/Model-Based RL.md` — ademas traducir Common Pitfalls, Check Your Understanding, Where to Go Next de ingles a espanol
- `03 - Resources/12 - Optimization/Constraint & Discrete Optimization.md` — idem
- `03 - Resources/13 - Infrastructure & DevOps/` — las 5 notas (IaC Fundamentals, Terraform Foundations, State & Backends, Modules & Project Structure, AWS CodeBuild)

### 4. Estandarizar numeracion en seccion 13
Las 5 notas de Infrastructure & DevOps usan `## Check Your Understanding` y `## Where to Go Next` sin numerar. Agregar numero secuencial (ej: `## 9. Check Your Understanding`, `## 10. Where to Go Next`).

### 5. Unificar idioma: Summary → Resumen
~58 notas de las secciones 01-20 usan `## N. Summary` (ingles). Cambiar a `## N. Resumen` (espanol).

### 6. Commit y push
Un solo commit con todos los cambios.
