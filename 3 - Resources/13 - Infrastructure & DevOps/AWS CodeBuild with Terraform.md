---
tags: [terraform, aws, codebuild, cicd, devops]
status: growing
created: 2026-06-28
---

# AWS CodeBuild con Terraform

## El objetivo

Al final de esta nota, tendrás un pipeline en AWS CodeBuild que:

1. Al abrir un Pull Request → ejecuta `terraform plan` y publica el resultado como comentario
2. Al hacer merge a main → ejecuta `terraform apply` automáticamente

Tu infraestructura se despliega desde código, con revisión, sin que nadie ejecute comandos manualmente.

## Requisitos previos

Antes de empezar, necesitas tener claro:

- [[Terraform Foundations]] — HCL básico, init, plan, apply
- [[Terraform State & Backends]] — state remoto en S3 con DynamoDB locking
- Un repo en GitHub (o CodeCommit) con tu código Terraform
- Cuenta AWS con permisos para CodeBuild, S3, CloudWatch Logs, IAM

## Arquitectura del pipeline

```
[GitHub PR] → CodeBuild (plan) → publica resultado
      ↓
[GitHub merge a main] → CodeBuild (apply) → despliega
      ↓
  [S3 Backend] ← state remoto
  [DynamoDB]   ← state lock
```

Cada ejecución de CodeBuild corre Terraform en un entorno efímero (contenedor). Por eso el backend remoto es obligatorio — el state no puede vivir en el build.

## 1. Política IAM para CodeBuild

CodeBuild necesita permisos para:

- Ejecutar Terraform (crear/modificar/destruir recursos)
- Leer y escribir el state en S3
- Adquirir/release lock en DynamoDB

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject"
      ],
      "Resource": "arn:aws:s3:::mi-terraform-state-2026/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:GetItem",
        "dynamodb:PutItem",
        "dynamodb:DeleteItem"
      ],
      "Resource": "arn:aws:dynamodb:us-east-1:123456789012:table/terraform-state-lock"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ec2:DescribeInstances",
        "ec2:RunInstances",
        "ec2:TerminateInstances",
        "ec2:CreateSecurityGroup",
        "ec2:AuthorizeSecurityGroupIngress",
        "ec2:DescribeSecurityGroups",
        "ec2:DeleteSecurityGroup"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:CreateBucket",
        "s3:ListBucket",
        "s3:PutBucketPolicy",
        "s3:DeleteBucket",
        "s3:PutBucketVersioning"
      ],
      "Resource": "*"
    }
  ]
}
```

> **Principio de mínimo privilegio**: los permisos de `ec2:*` y `s3:*` deberían limitarse a los recursos específicos que tu Terraform gestiona. Pero al empezar, `Resource: "*"` es aceptable. A medida que tu infraestructura crece, ajusta las políticas.

Crea un rol IAM para CodeBuild con esta política. Lo usarás en el proyecto de CodeBuild.

## 2. Estructura del repositorio

```
infra-repo/
├── environments/
│   └── dev/
│       ├── main.tf
│       ├── variables.tf
│       ├── outputs.tf
│       └── backend.hcl
├── modules/
│   └── ec2-instance/
├── buildspec/
│   ├── plan.yml          ← buildspec para PR (solo plan)
│   └── apply.yml         ← buildspec para merge (plan + apply)
└── .tfversion            ← contiene "1.9.0"
```

## 3. Buildspec — el corazón del pipeline

CodeBuild usa un archivo `buildspec.yml` que define los pasos de construcción.

### buildspec/plan.yml — para Pull Requests

```yaml
version: 0.2

phases:
  install:
    runtime-versions:
      python: 3.12
    commands:
      # Descargar e instalar Terraform
      - TF_VERSION=$(cat .tfversion)
      - wget -q https://releases.hashicorp.com/terraform/${TF_VERSION}/terraform_${TF_VERSION}_linux_amd64.zip
      - unzip -q terraform_${TF_VERSION}_linux_amd64.zip -d /usr/local/bin
      - terraform --version

  pre_build:
    commands:
      - cd environments/dev
      - terraform init -backend-config=backend.hcl
      - echo "Inicialización completada"

  build:
    commands:
      - terraform plan -no-color -out=tfplan 2>&1 | tee plan_output.txt

  post_build:
    commands:
      # Mostrar resultado del plan en los logs
      - echo "=== PLAN OUTPUT ==="
      - cat plan_output.txt
      # Guardar el plan como artefacto para depuración
      - cp plan_output.txt /tmp/plan_output.txt
      - cp tfplan /tmp/tfplan

artifacts:
  files:
    - plan_output.txt
    - tfplan
  base-directory: /tmp
  discard-paths: yes
```

### buildspec/apply.yml — para merge a main

```yaml
version: 0.2

phases:
  install:
    runtime-versions:
      python: 3.12
    commands:
      - TF_VERSION=$(cat .tfversion)
      - wget -q https://releases.hashicorp.com/terraform/${TF_VERSION}/terraform_${TF_VERSION}_linux_amd64.zip
      - unzip -q terraform_${TF_VERSION}_linux_amd64.zip -d /usr/local/bin
      - terraform --version

  pre_build:
    commands:
      - cd environments/dev
      - terraform init -backend-config=backend.hcl

  build:
    commands:
      - terraform plan -no-color -out=tfplan
      - terraform apply -auto-approve tfplan
```

Diferencias clave entre plan.yml y apply.yml:

| Aspecto | plan.yml | apply.yml |
|---------|----------|-----------|
| Cuándo se ejecuta | PR (cada push) | Merge a main |
| Acción | `terraform plan` | `terraform apply` |
| Auto-approve | No aplica | `-auto-approve` |
| Artefacto | Guarda plan como evidencia | No necesita |

## 4. Crear el proyecto CodeBuild

### Opción A: Desde la consola AWS

1. Ve a CodeBuild → Create build project
2. **Project name**: `terraform-dev-plan`
3. **Source**: GitHub (o CodeCommit) — conecta tu repo
4. **Source version**: `main` (para plan) (o usa webhook para PR)
5. **Environment**:
   - Managed image: `aws/codebuild/standard:7.0`
   - Runtime: Standard
   - Image version: 7.0
   - Privileged: true (necesario si usas Docker dentro de Terraform)
   - Service role: el rol IAM que creaste antes
6. **Buildspec**: `buildspec/plan.yml`
7. **Logs**: CloudWatch (para ver los logs de plan)

### Opción B: Con Terraform (apropiado — IaC para el pipeline mismo)

```hcl
resource "aws_codebuild_project" "terraform_plan" {
  name          = "terraform-dev-plan"
  description   = "Terraform plan for dev environment"
  build_timeout = 30

  service_role = aws_iam_role.codebuild.arn

  artifacts {
    type = "NO_ARTIFACTS"
  }

  environment {
    compute_type    = "BUILD_GENERAL1_SMALL"
    image           = "aws/codebuild/standard:7.0"
    type            = "LINUX_CONTAINER"
    privileged_mode = true
  }

  source {
    type      = "GITHUB"
    location  = "https://github.com/tu-org/infra-repo.git"
    buildspec = "buildspec/plan.yml"

    git_submodules_config {
      fetch_submodules = true
    }
  }

  logs_config {
    cloudwatch_logs {
      group_name  = "/aws/codebuild/terraform"
      stream_name = "dev-plan"
    }
  }
}
```

## 5. Configurar webhooks (PR triggers)

Para que CodeBuild se ejecute automáticamente en cada PR, necesitas un webhook.

En el proyecto de CodeBuild, ve a **Build triggers** y:

```
1. Rebuild every time a code change is pushed
2. Event type: PULL_REQUEST_CREATED, PULL_REQUEST_UPDATED
3. Filter: HEAD_REF = refs/heads/main  (o la rama que uses como base)
```

Si usas Terraform para definir el proyecto, agrega un webhook filter:

```hcl
resource "aws_codebuild_webhook" "plan" {
  project_name = aws_codebuild_project.terraform_plan.name

  filter_group {
    filter {
      type    = "EVENT"
      pattern = "PULL_REQUEST_CREATED,PULL_REQUEST_UPDATED,PULL_REQUEST_REOPENED"
    }
    filter {
      type    = "HEAD_REF"
      pattern = "refs/heads/main"
    }
  }
}
```

## 6. Segundo proyecto: apply en merge

Crea un proyecto similar pero con:

- **Name**: `terraform-dev-apply`
- **Buildspec**: `buildspec/apply.yml`
- **Trigger**: `PUSH` a la rama `main`

```hcl
resource "aws_codebuild_project" "terraform_apply" {
  name          = "terraform-dev-apply"
  description   = "Terraform apply for dev environment"
  build_timeout = 60

  service_role = aws_iam_role.codebuild.arn

  environment {
    compute_type    = "BUILD_GENERAL1_SMALL"
    image           = "aws/codebuild/standard:7.0"
    type            = "LINUX_CONTAINER"
    privileged_mode = true
  }

  source {
    type      = "GITHUB"
    location  = "https://github.com/tu-org/infra-repo.git"
    buildspec = "buildspec/apply.yml"
  }

  logs_config {
    cloudwatch_logs {
      group_name  = "/aws/codebuild/terraform"
      stream_name = "dev-apply"
    }
  }
}

resource "aws_codebuild_webhook" "apply" {
  project_name = aws_codebuild_project.terraform_apply.name

  filter_group {
    filter {
      type    = "EVENT"
      pattern = "PUSH"
    }
    filter {
      type    = "HEAD_REF"
      pattern = "refs/heads/main"
    }
  }
}
```

## 7. Plan de aprobación (opcional pero recomendado)

Para producción, probablemente quieras que `apply` no sea automático. Una forma común:

```
PR mergeado a main → CodeBuild plan (automático)
                          ↓
                [Se genera un artefacto con el plan]
                          ↓
            Aprobación manual (consola AWS o SNS)
                          ↓
                CodeBuild apply (manual trigger)
```

O usa **CodePipeline** entre plan y apply con un paso de aprobación manual:

```
CodePipeline:
  Source → CodeBuild (plan) → Manual Approval → CodeBuild (apply)
```

Pero si estás empezando, el flujo automático plan → apply es suficiente para un entorno dev.

## 8. Variables de entorno en CodeBuild

Nunca quemes secretos (AWS access keys) en el buildspec. En su lugar:

```yaml
environment_variables:
  - name: AWS_DEFAULT_REGION
    value: us-east-1
  - name: TF_VAR_environment
    value: dev
  - name: TF_VAR_bucket_name
    value: "mi-logs-dev"
```

Para secretos (API keys, tokens), usa `type: PARAMETER_STORE` o `type: SECRETS_MANAGER`:

```yaml
environment_variables:
  - name: GITHUB_TOKEN
    value: /codebuild/github-token
    type: PARAMETER_STORE
```

## 9. Plan de respuesta a fallos

Posibles fallos y cómo debuguearlos:

| Error | Causa probable | Solución |
|-------|---------------|----------|
| `Error acquiring state lock` | Otro build ejecutándose | Espera. Verifica en DynamoDB console |
| `BucketAlreadyExists` | Nombre de bucket no único | Cambia el nombre en tfvars |
| `InvalidTemplateException` | Error de sintaxis en HCL | `terraform validate` localmente |
| `AccessDenied` → S3 | Rol IAM sin permisos | Revisa la política IAM del rol |
| Terraform no instalado | `install` phase falló | Verifica que `wget` funciona en el contenedor |

## 10. Ejemplo completo del flujo

```
1. Developer: git checkout -b feature/agregar-sg
2. Developer: modifica main.tf (agrega security group)
3. Developer: git push origin feature/agregar-sg
4. Developer: abre PR contra main
5. CodeBuild: detecta PR → ejecuta buildspec/plan.yml
6. CodeBuild: muestra terraform plan en logs
7. Developer: revisa el plan, ve que crea un security group
8. Team: aprueba PR en GitHub
9. Developer: mergea PR a main
10. CodeBuild: detecta push a main → ejecuta buildspec/apply.yml
11. CodeBuild: terraform apply -auto-approve
12. Infraestructura actualizada. Todo en código, todo revisado.
```

## Common Pitfalls

- **State remoto no configurado**: CodeBuild ejecuta desde un contenedor efímero. Sin backend remoto, el state se pierde al terminar el build. La automatización del CLI es un pilar de [[CLI & Productivity]].
- **Lock conflicts**: si haces push frecuente, dos builds pueden intentar apply simultáneamente. Configura la cola de CodeBuild para ejecutar un build a la vez.
- **IAM permissions insuficientes**: el error típico es `AccessDenied` al leer el state o crear recursos. Dale permisos de escritura al rol de CodeBuild.
- **`.tfversion` olvidado**: si el archivo no existe, el build falla al hacer `cat .tfversion`. Verifica que existe en el repo.
- **Buildspec en la raíz**: puedes poner `buildspec.yml` en la raíz del repo. Pero es mejor usar subdirectorios para plan/apply separados.
- **Terraform versión incompatible**: CodeBuild usa Amazon Linux. Asegúrate de que la versión de Terraform que descargas sea compatible con linux_amd64.

## Check Your Understanding

1. ¿Por qué no puedes usar backend local cuando ejecutas Terraform desde CodeBuild?
2. En el buildspec de plan.yml, ¿por qué usamos `-no-color` y `tee`?
3. Si quieres que apply no sea automático sino que requiera aprobación, ¿qué cambios harías al pipeline?
4. Un build de plan falla con "Error acquiring state lock: ConditionalCheckFailedException". ¿Qué está pasando y qué haces?

## Where to Go Next

- [[Terraform State & Backends]] — la base para que CodeBuild funcione
- [[Terraform Modules & Project Structure]] — organiza el repo para múltiples entornos
- [[Terraform Foundations]] — domina el CLI antes de automatizarlo
- [[ML Pipelines]] — infraestructura similar aplicada a ML
- [[Git]] — el flujo de PRs y merges es la base de GitOps
