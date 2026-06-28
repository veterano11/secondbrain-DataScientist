---
tags: [terraform, hcl, fundamentals, aws]
status: growing
created: 2026-06-28
---

# Terraform Foundations

## 1. Escenario de aprendizaje

Vas a desplegar un bucket S3 y una instancia EC2 con Nginx, todo desde tu máquina local. Al terminar esta nota, tendrás claros los conceptos fundamentales y podrás leer y modificar cualquier proyecto Terraform básico.

## 2. Requisitos

```
Terraform ≥ 1.6    → https://developer.hashicorp.com/terraform/install
AWS CLI configurado → aws configure (access key + secret)
Cuenta AWS con permisos para EC2, S3, Security Groups
```

Verifica:

```bash
$ terraform --version
Terraform v1.9.0

$ aws sts get-caller-identity
{
    "Account": "123456789012",
    "UserId": "AIDA...",
    "Arn": "arn:aws:iam::123456789012:user/tu-usuario"
}
```

Si `aws sts get-caller-identity` falla, no puedes seguir. Detente y configura AWS CLI primero.

## 3. HCL — HashiCorp Configuration Language

Terraform usa HCL. No es JSON ni YAML, aunque se parece más a YAML. Bloques con llaves y argumentos con `=`:

```hcl
resource "aws_s3_bucket" "mi_bucket" {
  bucket = "nombre-unico-global-2026"
  tags = {
    Name        = "Mi bucket"
    Environment = "dev"
  }
}
```

Estructura general:

```
<tipo> "<etiqueta_tipo>" "<nombre_local>" {
  <argumento> = <valor>
}
```

El **nombre local** (`mi_bucket`) es interno de Terraform. Otros recursos se refieren a él como `aws_s3_bucket.mi_bucket`. El nombre real en AWS lo defines con el argumento `bucket`.

## 4. Tu primer main.tf

Crea un directorio vacío y dentro pon esto:

```hcl
# main.tf
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-east-1"
}

resource "aws_s3_bucket" "logs" {
  bucket = "mi-logs-20260628"  # usa un nombre único global
  tags = {
    Name = "LogsBucket"
  }
}
```

> **Atención**: el nombre del bucket S3 debe ser único en toda AWS (no solo en tu cuenta). Incluye tu nombre o un sufijo único.

## 5. terraform init — descargar providers

```bash
$ terraform init

Initializing the backend...
Initializing provider plugins...
- Finding hashicorp/aws versions matching "~> 5.0"...
- Installing hashicorp/aws v5.70.0...
- Installed hashicorp/aws v5.70.0 (signed by HashiCorp)

Terraform has been successfully initialized!
```

`terraform init` descarga los providers (en este caso AWS) en el directorio `.terraform/`. Es el **primer comando que siempre ejecutas** en un proyecto Terraform, y cada vez que agregas un nuevo provider.

¿Qué pasó en el sistema de archivos?

```
mi-proyecto/
├── main.tf
├── .terraform/            ← providers descargados
│   └── providers/
│       └── registry.terraform.io/
│           └── hashicorp/
│               └── aws/
└── .terraform.lock.hcl    ← lock file (versiona esto en git)
```

## 6. terraform plan — qué va a pasar

```bash
$ terraform plan

Terraform used the selected providers to generate the following execution plan.
Resource actions are indicated with the following symbols:
  + create

Terraform will perform the following actions:

  # aws_s3_bucket.logs will be created
  + resource "aws_s3_bucket" "logs" {
      + bucket                      = "mi-logs-20260628"
      + id                          = (known after apply)
      + tags                        = {
          + "Name" = "LogsBucket"
        }
      + tags_all                    = {
          + "Name" = "LogsBucket"
        }
      # (9 unchanged attributes hidden)
    }

Plan: 1 to add, 0 to change, 0 to destroy.
```

`+` significa "crear". `-` sería destruir. `~` sería modificar.

El plan **no ejecuta nada**. Solo te muestra lo que haría. Siempre revisa el plan antes de apply.

> **Pregunta**: ¿por qué `id` dice "(known after apply)"? Porque Terraform no sabe el ID del bucket hasta que AWS lo crea. Algunos valores solo se conocen en tiempo de ejecución.

## 7. terraform apply — ejecutar

```bash
$ terraform apply

Terraform will perform the following actions:
  # aws_s3_bucket.logs will be created
  + resource "aws_s3_bucket" "logs" { ... }

Plan: 1 to add, 0 to change, 0 to destroy.

Do you want to perform these actions?
  Terraform will perform the actions described above.
  Only 'yes' will be accepted to approve.

  Enter a value: yes

aws_s3_bucket.logs: Creating...
aws_s3_bucket.logs: Creation complete after 1s [id=mi-logs-20260628]

Apply complete! Resources: 1 added, 0 changed, 0 destroyed.
```

Terraform te pide confirmación. En CI/CD usas `terraform apply -auto-approve`, pero en local es buena práctica revisar.

Después de apply, aparece un archivo nuevo:

```
mi-proyecto/
├── terraform.tfstate       ← el state file
├── terraform.tfstate.backup ← backup del state anterior
```

**Nunca edites terraform.tfstate a mano. Nunca.**

## 8. terraform destroy — limpiar

```bash
$ terraform destroy
  # aws_s3_bucket.logs will be DESTROYED
Do you want to perform these actions? yes
aws_s3_bucket.logs: Destroying...
aws_s3_bucket.logs: Destruction complete after 0s

Destroy complete! Resources: 1 destroyed.
```

**Cuidado**: `terraform destroy` borra TODO lo que está en el state. En producción nunca ejecutes esto sin revisar el plan.

## 9. Variables — no quemes valores

Hasta ahora los valores están hardcodeados. Las variables hacen el código reutilizable:

```hcl
# variables.tf
variable "bucket_name" {
  description = "Nombre único del bucket S3"
  type        = string
}

variable "environment" {
  description = "Entorno (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "instance_type" {
  description = "Tipo de instancia EC2"
  type        = string
  default     = "t2.micro"
}
```

```hcl
# terraform.tfvars  — archivo con valores concretos (NO se versiona en git)
bucket_name = "mi-logs-dev-20260628"
environment = "dev"
```

> **Importante**: `terraform.tfvars` se carga automáticamente. Para archivos con otro nombre, usas `-var-file=prod.tfvars`.

Ahora usa las variables en main.tf:

```hcl
resource "aws_s3_bucket" "logs" {
  bucket = var.bucket_name
  tags = {
    Name        = "LogsBucket"
    Environment = var.environment
  }
}
```

Al ejecutar:

```bash
$ terraform plan
# usa terraform.tfvars automáticamente

$ terraform plan -var="bucket_name=otro-nombre"
# override inline

$ terraform plan -var-file=prod.tfvars
# usa otro archivo de variables
```

## 10. Outputs — qué información sacar

Los outputs exprimen información útil después del apply:

```hcl
# outputs.tf
output "bucket_arn" {
  description = "ARN del bucket S3"
  value       = aws_s3_bucket.logs.arn
}

output "bucket_id" {
  value = aws_s3_bucket.logs.id
}
```

Después de `terraform apply`:

```
Apply complete! Resources: 1 added.

Outputs:

bucket_arn = "arn:aws:s3:::mi-logs-20260628"
bucket_id = "mi-logs-20260628"
```

También puedes consultar outputs en cualquier momento:

```bash
$ terraform output bucket_arn
"arn:aws:s3:::mi-logs-20260628"

$ terraform output --json
{"bucket_arn": {"value": "arn:aws:s3:::mi-logs-20260628"}}
```

Esto es útil para scripts que necesitan valores de infraestructura.

## 11. Data Sources — leer infraestructura existente

No todo lo creas con Terraform. A veces necesitas referenciar algo que ya existe (una VPC, un AMI):

```hcl
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]  # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-22.04-amd64-server-*"]
  }
}

resource "aws_instance" "web" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = "t2.micro"
}
```

Los data sources se leen en cada `plan` y `apply`, pero no crean ni modifican nada.

## 12. Dependencias implícitas y explícitas

Terraform descubre automáticamente el orden de creación. Si declaras:

```hcl
resource "aws_s3_bucket" "logs" { ... }
resource "aws_s3_bucket_policy" "logs_policy" {
  bucket = aws_s3_bucket.logs.id
  ...
}
```

Terraform sabe que `logs_policy` depende de `logs` y crea el bucket primero. Eso es **dependencia implícita**.

A veces necesitas forzar una dependencia con `depends_on`:

```hcl
resource "aws_instance" "web" {
  depends_on = [aws_s3_bucket.logs]
  # ...
}
```

Usa `depends_on` cuando Terraform no puede inferir la dependencia (por ejemplo, si pasas el nombre del bucket como string plano en vez de usar `aws_s3_bucket.logs.id`).

## 13. Flujo de trabajo completo (resumen ejecutable)

```bash
# 1. Crear proyecto
mkdir terraform-demo && cd terraform-demo
# 2. Escribir main.tf, variables.tf, outputs.tf (copiar de arriba)
# 3. Inicializar
terraform init
# 4. Validar sintaxis
terraform validate
# 5. Ver plan
terraform plan
# 6. Aplicar
terraform apply
# 7. Destruir cuando termines
terraform destroy
```

Estos comandos forman el núcleo de tu [[CLI & Productivity]] con Terraform. Cada vez que modificas archivos `.tf`, repites `plan → apply`. No necesitas volver a correr `init` a menos que agregues providers.

## 14. Common Mistakes

- **Olvidar `terraform init`**: el error clásico. Clonas un repo con [[Git|Terraform]] y ejecutas plan sin init. Terraform se queja de que no encuentra los providers.
- **Bucket name no único**: S3 exige nombres globalmente únicos. Si ves `BucketAlreadyExists`, cambia el nombre.
- **Ejecutar apply sin revisar plan**: Terraform puede destruir recursos si cambiaste algo. Siempre lee el plan.
- **Mezclar regiones**: un resource en us-east-1 y otro en sa-east-1 funcionan, pero la latencia y los costos de transferencia te van a sorprender.
- **State file en local**: si pierdes el archivo, Terraform no sabe qué creó. Un backup remoto es [[Code Quality|obligatorio]] para cualquier proyecto real.
- **Tocar resources fuera de Terraform**: modificas el bucket desde la consola AWS → el próximo `apply` lo revierte al estado declarado.

## Check Your Understanding

1. Escribiste `terraform plan` y ves `-` (destroy) al lado de un resource que no tocaste. ¿Qué puede estar pasando?
2. Si quieres desplegar el mismo código en dev, staging y prod cambiando solo el nombre del bucket y el instance_type, ¿cómo lo harías?
3. `terraform init` falla con "Could not retrieve provider hashicorp/aws". ¿Qué verificas primero?

## Where to Go Next

- [[Terraform State & Backends]] — cómo hacer que el state sea compartible y seguro
- [[Terraform Modules & Project Structure]] — organizar proyectos que crecen
- [[AWS CodeBuild with Terraform]] — integrar Terraform en CI/CD
- [[Infrastructure as Code Fundamentals]] — por qué Terraform hace esto así
