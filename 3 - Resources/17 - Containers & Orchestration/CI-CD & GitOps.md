---
tags:
  - ci-cd
  - gitops
  - devops
  - github-actions
  - argocd
status: seedling
created: 2026-06-28
---

# CI-CD & GitOps

## Escenario de aprendizaje

Hacés push a main. Automáticamente querés que: los tests pasen, la imagen Docker se construya y suba a ECR, se deploye a staging, corran tests de integración, y si todo sale bien, se deploye a producción. Y si algo falla en prod, querés volver atrás automáticamente en segundos. Esto es CI/CD. GitOps es la evolución donde Git es la fuente de verdad del estado del cluster: no aplicás cambios manualmente con `kubectl`, sino que un operador (ArgoCD) sincroniza Git con el cluster.

## 1. CI vs CD

| Etapa | Qué hace | Herramientas |
|-------|----------|--------------|
| **Continuous Integration (CI)** | Build, lint, test, security scan, push de imagen | GitHub Actions, GitLab CI, Jenkins, CircleCI |
| **Continuous Delivery (CD)** | Deploy a staging, tests de integración, aprobación manual a prod | ArgoCD, Flux, Spinnaker |
| **Continuous Deployment** | Deploy automático a prod sin aprobación humana | ArgoCD + health checks |

CI se asegura de que el código es válido y está empaquetado. CD se asegura de que llegue a los ambientes correctos. GitOps es la implementación moderna de CD usando Git como fuente de verdad, similar a cómo [[Infrastructure as Code Fundamentals]] usa Git para infraestructura.

## 2. GitHub Actions — Conceptos

```yaml
name: CI Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v4
      with:
        python-version: "3.11"
    - run: pip install -r requirements-dev.txt
    - run: pytest --cov --cov-report=xml
    - uses: codecov/codecov-action@v3
```

- **Workflow**: archivo YAML en `.github/workflows/`
- **Jobs**: unidades que corren en paralelo o secuenciales (`needs:`)
- **Steps**: comandos individuales dentro de un job
- **Actions**: bloques reutilizables (checkout, setup, login to cloud)
- **Triggers**: `push`, `pull_request`, `schedule` (cron), `workflow_dispatch` (manual)

Las Actions se componen como pipes de [[CLI & Productivity]], y los secretos se guardan en GitHub Secrets (Settings → Secrets and variables).

## 3. Pipeline de CI

Un pipeline CI típico para una app containerizada:

```yaml
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - run: pip install ruff && ruff check .

  test:
    needs: lint
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - run: pip install -r requirements-dev.txt && pytest

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - run: docker build -t mi-app:${{ github.sha }} .
    - run: |
        aws ecr get-login-password | docker login --password-stdin $ECR_REGISTRY
        docker push $ECR_REGISTRY/mi-app:${{ github.sha }}
    - run: trivy image $ECR_REGISTRY/mi-app:${{ github.sha }}
```

```bash
$ gh workflow run "CI Pipeline"  # trigger manual
```

**Salida esperada:**
```
lint: ruff check passed (0 errors)
test: pytest passed (45 tests, 0 failures)
build: docker build + push to ECR + trivy scan (0 critical)
```

El tag de la imagen es el commit SHA (`${{ github.sha }}`), lo que garantiza trazabilidad — cada imagen está linkeada a un commit. Esto se conecta con [[Docker Fundamentals]] en el versionado de imágenes.

## 4. Pipeline de CD

```yaml
deploy-staging:
  needs: build
  runs-on: ubuntu-latest
  environment: staging
  steps:
  - uses: actions/checkout@v4
  - run: |
      helm upgrade mi-app ./charts/mi-app \
        --values values-staging.yaml \
        --set image.tag=${{ github.sha }} \
        --install --wait

e2e-tests:
  needs: deploy-staging
  runs-on: ubuntu-latest
  steps:
  - run: pytest tests/e2e/ --base-url https://staging.midominio.com

deploy-prod:
  needs: e2e-tests
  runs-on: ubuntu-latest
  environment: production
  steps:
  - uses: actions/checkout@v4
  - run: |
      helm upgrade mi-app ./charts/mi-app \
        --values values-prod.yaml \
        --set image.tag=${{ github.sha }} \
        --install --wait --timeout 5m

smoke-tests:
  needs: deploy-prod
  runs-on: ubuntu-latest
  steps:
  - run: pytest tests/smoke/ --base-url https://midominio.com
```

- **Environments**: staging y prod tienen approval gates distintos
- **Helm upgrade + --wait**: espera a que los Pods estén listos
- **Smoke tests**: verificaciones rápidas post-deploy (health endpoint, DB connection)

El CD usa [[Helm & Package Management]] para deployar con valores específicos de cada ambiente.

## 5. GitOps con ArgoCD

GitOps cambia el paradigma: en vez de CI/CD push, ArgoCD **tira** (pull) el estado desde Git:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: mi-app
spec:
  destination:
    namespace: default
    server: https://kubernetes.default.svc
  project: default
  source:
    path: charts/mi-app
    repoURL: https://github.com/mi-org/mi-app
    targetRevision: main
    helm:
      valueFiles:
      - values-prod.yaml
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
    - CreateNamespace=true
```

**Ventajas de GitOps:**
- **Source of truth**: el repo de Git es el único lugar donde se define el estado deseado
- **Auto-healing**: si alguien hace `kubectl delete deployment`, ArgoCD lo restaura
- **Prune**: si eliminás un archivo del repo, ArgoCD elimina el recurso del cluster
- **Audit trail**: cada cambio en el cluster está en el historial de Git

```bash
$ argocd app sync mi-app
$ argocd app get mi-app
$ argocd app rollback mi-app --prune
```

ArgoCD se complementa con [[Git]] para revisar cambios vía PRs antes de sincronizar.

## 6. Estrategias de deploy

| Estrategia | Downtime | Riesgo | Velocidad |
|------------|----------|--------|-----------|
| **Rolling update** (K8s default) | 0 | Bajo | Lenta |
| **Blue-green** | 0 | Medio | Instantánea |
| **Canary** | 0 | Bajo | Gradual |
| **Recreate** | Sí | Alto | Rápida |

**Canary deployment con ArgoCD Rollouts:**

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
spec:
  replicas: 10
  strategy:
    canary:
      steps:
      - setWeight: 10
      - pause: {duration: 5m}
      - setWeight: 50
      - pause: {duration: 5m}
      - setWeight: 100
```

El tráfico se redirige gradualmente: 10%, luego 50%, luego 100%. Si hay errores en el camino, se aborta automáticamente. Esto protege la experiencia de usuario y es parte de [[Model Monitoring]] cuando deployás modelos ML.

## 7. Common Mistakes

- **Secrets en los pipelines**: jamás hardcodees credenciales; usá GitHub Secrets / AWS Secrets Manager
- **No testear rollback**: probá `argocd app rollback` y `helm rollback` en staging
- **Deployments sin health checks**: si no hay liveness/readiness probes, el deploy no sabe si la app arrancó bien
- **CI lento**: optimizá caching de dependencias (pip cache, Docker layer caching) para que CI no tome 20 minutos
- **No versionar imágenes**: usá el commit SHA como tag; `latest` no es trazable
- **Permisos excesivos**: los tokens de CI/CD deben tener el mínimo permiso necesario (principle of least privilege)

## Resumen

1. CI buildea, testea y empaqueta; CD deploya con estrategias controladas (rolling, blue-green, canary)
2. GitHub Actions usa workflows, jobs, steps y triggers (push, PR, schedule)
3. CI pipeline típico: lint → test → build → push → security scan
4. CD pipeline: deploy staging → e2e tests → deploy prod → smoke tests
5. GitOps con ArgoCD usa Git como fuente de verdad: sync, auto-healing, prune
6. Las estrategias de deploy (rolling, blue-green, canary) ofrecen trade-offs entre velocidad y riesgo

## Check Your Understanding

1. ¿Cuál es la diferencia entre Continuous Delivery y Continuous Deployment? <!-- Delivery requiere aprobación manual para pasar a prod; Deployment es automático. -->
2. ¿Qué ventaja tiene GitOps sobre el CD tradicional? <!-- Git es la fuente de verdad; ArgoCD sincroniza automáticamente, hay auto-healing y audit trail completo. -->
3. ¿Para qué sirve `--wait` en `helm upgrade`? <!-- Para que el comando espere hasta que los Pods estén listos antes de continuar. -->
4. ¿En qué se diferencia un canary deployment de un blue-green? <!-- Canary redirige tráfico gradualmente (10% → 50% → 100%); blue-green cambia todo el tráfico de una vez entre dos entornos completos. -->
5. ¿Por qué es importante usar el commit SHA como tag de imagen en CI? <!-- Para tener trazabilidad entre el código y la imagen desplegada; cada imagen se corresponde unívocamente a un commit. -->

## Where to Go Next

- [[Docker Fundamentals]]
- [[Kubernetes Fundamentals]]
- [[Git]]
- [[Helm & Package Management]]
- [[CLI & Productivity]]
- [[AWS CodeBuild with Terraform]]
- [[Infrastructure as Code Fundamentals]]
- [[Testing for Data Science]]
