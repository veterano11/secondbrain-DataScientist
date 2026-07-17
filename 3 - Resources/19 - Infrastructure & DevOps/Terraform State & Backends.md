---
tags: [terraform, state, backend, aws, s3, dynamodb]
status: growing
created: 2026-06-28
---

# Terraform State & Backends

## 1. Escenario de aprendizaje

Cuando ejecutaste `terraform apply`, Terraform creó un archivo `terraform.tfstate`. Ese archivo contiene un mapeo exacto entre tu código y los recursos reales en AWS:

```json
{
  "resources": [
    {
      "module": "root",
      "mode": "managed",
      "type": "aws_s3_bucket",
      "name": "logs",
      "provider": "provider[\"registry.terraform.io/hashicorp/aws\"]",
      "instances": [
        {
          "attributes": {
            "id": "mi-logs-20260628",
            "arn": "arn:aws:s3:::mi-logs-20260628",
            "bucket": "mi-logs-20260628",
            "tags": { "Name": "LogsBucket" }
          }
        }
      ]
    }
  ]
}
```

Sin este archivo, Terraform no sabe que `mi-logs-20260628` fue creado por él. Si pierdes el state:

- `terraform plan` muestra que va a **crear todo de nuevo** (porque no sabe que ya existe)
- Hacer `apply` falla porque el bucket ya existe
- O peor: Terraform crea recursos duplicados con diferentes nombres

> **Escenario real**: ejecutas Terraform desde tu laptop, creas 20 recursos. Tu laptop muere. El state estaba en el disco local. Ahora tienes 20 recursos en AWS que Terraform no puede gestionar. No sabes cuáles creó él y cuáles existían antes. La única solución es importar cada recurso manualmente.

## 2. Backend remoto: la solución

Como viste en [[Terraform Foundations]], el state debe vivir fuera de tu máquina. En lugar de guardarlo en tu disco, lo guardas en **S3**. Así:

- Es accesible desde cualquier máquina (tu laptop, CodeBuild, un colega)
- Versionado: S3 guarda versiones anteriores del state
- Seguro: cifrado en reposo con KMS

Configuración en `main.tf`:

```hcl
terraform {
  backend "s3" {
    bucket         = "mi-terraform-state-2026"
    key            = "infra/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-state-lock"
  }
}
```

> **Orden crítico**: primero debes crear el bucket S3 y la tabla DynamoDB. Puedes hacerlo con Terraform usando backend local, y luego migrar. O crearlos a mano una sola vez. No puedes usar S3 como backend si el bucket aún no existe.

## 3. Paso a paso: migrar de local a remoto

### 1. Crear bucket y tabla DynamoDB

Crea un proyecto aparte (o usa un backend local temporal):

```hcl
# infra-bootstrap/main.tf
resource "aws_s3_bucket" "state" {
  bucket = "mi-terraform-state-2026"
}

resource "aws_s3_bucket_versioning" "state" {
  bucket = aws_s3_bucket.state.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "state" {
  bucket = aws_s3_bucket.state.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_dynamodb_table" "lock" {
  name         = "terraform-state-lock"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "LockID"
  attribute {
    name = "LockID"
    type = "S"
  }
}
```

Ejecuta con backend local. Este es el **único recurso que crearás fuera del backend remoto**.

> **Alternativa más simple**: crea el bucket y la tabla desde la consola AWS o AWS CLI. Son dos comandos y no cambian frecuentemente:
> ```bash
> aws s3 mb s3://mi-terraform-state-2026 --region us-east-1
> aws s3api put-bucket-versioning --bucket mi-terraform-state-2026 --versioning-configuration Status=Enabled
> aws dynamodb create-table --table-name terraform-state-lock --attribute-definitions AttributeName=LockID,AttributeType=S --key-schema AttributeName=LockID,KeyType=HASH --billing-mode PAY_PER_REQUEST
> ```

### 2. Agregar backend a tu proyecto

```hcl
terraform {
  backend "s3" {
    bucket         = "mi-terraform-state-2026"
    key            = "infra/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-state-lock"
  }
}
```

### 3. Inicializar con migración

```bash
$ terraform init -migrate-state

Initializing the backend...

Do you want to copy existing state to the new backend?
  Pre-existing state was found while migrating the previous "local" backend.
  No changes were detected on the backend. Would you like to copy the state
  from your local backend to the S3 backend?

  Enter a value: yes

Successfully configured the backend "s3"!
Terraform will automatically use this backend for all operations.
```

A partir de ahora, cada `terraform plan` y `apply` lee y escribe el state desde S3.

## 4. DynamoDB Locking: evitar corrupción

Cuando trabajas en equipo con [[Git]] (o desde CI/CD), dos procesos pueden ejecutar `terraform apply` al mismo tiempo. El que escribe primero actualiza el state. El segundo escribe sobre un state desactualizado → corrupción o recursos duplicados.

DynamoDB actúa como un **lock distribuido**:

```
Proceso A (CodeBuild): 
  1. Adquiere lock en DynamoDB
  2. Lee state de S3
  3. Ejecuta apply
  4. Escribe state a S3
  5. Libera lock

Proceso B (tu laptop): intenta apply
  1. Intenta adquirir lock → BLOQUEADO
  2. Espera o falla con: "Acquiring state lock. LockInfo: ..."
```

Si ves este error:

```
Error: Error acquiring the state lock

Error message: 2 conflicts
Lock Info:
  ID:        abc123...
  Path:      infra/terraform.tfstate
  Operation: OperationTypeApply
  Who:       CodeBuild@build-123
  Version:   1.9.0
  Created:   2026-06-28 10:30:00
```

Tienes tres opciones:

1. **Esperar**: el lock expira si el proceso se cuelga (timeout configurable)
2. **Investigar**: ¿quién ejecutó apply? Si fue otro build en CodeBuild, espera
3. **Forzar** (solo si estás seguro de que nadie está ejecutando): `terraform force-unlock <LOCK_ID>`

**NUNCA fuerces unlock si hay un apply en progreso. Corromperás el state.**

## 5. State para múltiples entornos

Con backend remoto, cada entorno tiene su propio state:

```hcl
# Entornos separados por key diferente

# dev
backend "s3" {
  key = "dev/terraform.tfstate"
}

# staging
backend "s3" {
  key = "staging/terraform.tfstate"
}

# prod
backend "s3" {
  key = "prod/terraform.tfstate"
}
```

Puedes usar partial configuration para no hardcodear el key:

```hcl
# backend.hcl
bucket  = "mi-terraform-state-2026"
region  = "us-east-1"
encrypt = true
dynamodb_table = "terraform-state-lock"
```

```hcl
# main.tf — sin backend block completo
terraform {
  backend "s3" {}  # config se pasa aparte
}
```

```bash
terraform init -backend-config=backend.hcl -backend-config="key=dev/terraform.tfstate"
terraform init -backend-config=backend.hcl -backend-config="key=prod/terraform.tfstate"
```

## 6. Terraform Workspaces (alternativa más simple)

Workspaces permite usar el mismo código con states separados sin cambiar archivos:

```bash
$ terraform workspace new dev
Created and switched to workspace "dev"!

$ terraform workspace new prod
Created and switched to workspace "prod"!

$ terraform workspace list
  default
  dev
* prod

$ terraform workspace select dev
Switched to workspace "dev".
```

Con backend S3, cada workspace tiene su propio state: `env:/dev/infra/terraform.tfstate`.

Usa `terraform.workspace` en tu código:

```hcl
resource "aws_s3_bucket" "logs" {
  bucket = "logs-${terraform.workspace}-2026"
  tags = {
    Environment = terraform.workspace
  }
}
```

**Workspaces son útiles para proyectos pequeños. Para equipos grandes, prefiere directorios separados o repos separados por entorno.**

## 7. Common Mistakes

- **Bucket S3 borrado accidentalmente**: si borras el bucket que contiene el state, pierdes todo. Activa versioning y MFA delete. Haz backups periódicos.
- **Lock ídem**: si forzas unlock mientras un build de CodeBuild está ejecutando apply, el state se corrompe. Siempre verifica primero.
- **State compartido sin lock**: dos personas ejecutan `terraform apply` sin DynamoDB. El segundo override escribe sobre el state desactualizado y Terraform pierde recursos sincronizados.
- **Path del state incorrecto**: si cambias el `key` del backend, Terraform crea un state nuevo y pierde el rastro de los recursos existentes. `terraform init -migrate-state` para mover.
- **Hardcodear backend en CI/CD**: usa partial configuration o variables de entorno, no quemes el bucket name en el código.

Seguir estas buenas prácticas de [[Code Quality]] previene la mayoría de los desastres con state.

## 8. Resumen

1. El state file es el activo más valioso de Terraform: mapea recursos declarados a recursos reales.
2. El backend local es funcional para experimentar pero peligroso para equipos.
3. S3 + DynamoDB es el backend remoto estándar: almacenamiento compartido + locking.
4. `terraform init -migrate-state` mueve el state de local a remoto sin perder recursos.
5. Workspaces permiten múltiples entornos con un solo backend, pero keys separadas son más explícitas.

## 9. Check Your Understanding

1. Dos developers ejecutan `terraform apply` al mismo tiempo. Uno tiene DynamoDB locking configurado, el otro no. ¿Qué diferencia hay en el comportamiento?
2. Perdiste el state local pero los recursos en AWS existen. ¿Cómo recuperas el control sin destruir nada?
3. ¿En qué caso preferirías workspaces sobre keys de backend separados para entornos?

## 10. Where to Go Next

- [[Terraform Modules & Project Structure]] — organizar el código limpio para varios entornos
- [[AWS CodeBuild with Terraform]] — CI/CD con state remoto
- [[Infrastructure as Code Fundamentals]] — el concepto de state en el contexto general de IaC
