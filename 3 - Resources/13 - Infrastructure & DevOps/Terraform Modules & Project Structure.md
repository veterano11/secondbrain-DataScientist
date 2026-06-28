---
tags: [terraform, modules, project-structure, best-practices]
status: growing
created: 2026-06-28
---

# Terraform Modules & Project Structure

## 1. Escenario de aprendizaje

Cuando empiezas, todo tu código Terraform cabe en un solo `main.tf`:

```
main.tf                  ← 300 líneas, todo junto
variables.tf
outputs.tf
terraform.tfstate
```

Esto funciona para un proyecto pequeño con 5 recursos. Pero cuando tienes VPC, subnets, EC2, RDS, load balancers, SQS, IAM roles...

```
main.tf                  ← 2000 líneas
variables.tf
```

Ahora `terraform plan` tarda 30 segundos aunque solo hayas cambiado una tag. Cualquier persona que abra `main.tf` tiene que entender todo el sistema para hacer un cambio pequeño. La [[CLI & Productivity]] se resiente con monolitos.

Los módulos resuelven esto.

## 2. ¿Qué es un módulo?

Un módulo es cualquier directorio con archivos `.tf`. Incluso tu `main.tf` raíz es un módulo (el "root module").

Un módulo reutilizable tiene:

```
modules/
└── ec2-instance/
    ├── main.tf         ← recursos
    ├── variables.tf    ← inputs
    └── outputs.tf      ← outputs
```

Lo llamas desde tu root module:

```hcl
module "web_server" {
  source   = "./modules/ec2-instance"
  ami_id   = "ami-0c55b159cbfafe1f0"
  name     = "web-server"
  subnet_id = "subnet-12345"
}
```

## 3. Paso a paso: crear tu primer módulo

### 1. Estructura de directorios

```
infra/
├── main.tf              ← root module (orquestador)
├── variables.tf
├── outputs.tf
├── modules/
│   └── ec2-instance/
│       ├── main.tf
│       ├── variables.tf
│       └── outputs.tf
└── envs/
    ├── dev.tfvars
    └── prod.tfvars
```

### 2. modules/ec2-instance/main.tf

```hcl
resource "aws_instance" "this" {
  ami                    = var.ami_id
  instance_type          = var.instance_type
  subnet_id              = var.subnet_id
  vpc_security_group_ids = var.security_group_ids

  tags = {
    Name = var.name
  }
}

resource "aws_eip" "this" {
  instance = aws_instance.this.id
  count    = var.assign_public_ip ? 1 : 0
}
```

### 3. modules/ec2-instance/variables.tf

```hcl
variable "ami_id" {
  description = "AMI ID para la instancia"
  type        = string
}

variable "instance_type" {
  description = "Tipo de instancia"
  type        = string
  default     = "t2.micro"
}

variable "name" {
  description = "Nombre del servidor"
  type        = string
}

variable "subnet_id" {
  description = "Subnet donde desplegar"
  type        = string
}

variable "security_group_ids" {
  description = "Lista de security groups"
  type        = list(string)
  default     = []
}

variable "assign_public_ip" {
  description = "Asignar IP pública"
  type        = bool
  default     = false
}
```

### 4. modules/ec2-instance/outputs.tf

```hcl
output "instance_id" {
  description = "ID de la instancia EC2"
  value       = aws_instance.this.id
}

output "public_ip" {
  description = "IP pública (si asignada)"
  value       = try(aws_eip.this[0].public_ip, null)
}

output "private_ip" {
  value = aws_instance.this.private_ip
}
```

### 5. main.tf raíz (usa el módulo)

```hcl
module "web" {
  source = "./modules/ec2-instance"

  ami_id      = "ami-0c55b159cbfafe1f0"
  name        = "web-server"
  subnet_id   = aws_subnet.public.id
  instance_type = var.instance_type
}

# También puedes usar variables para pasar al módulo
module "worker" {
  source = "./modules/ec2-instance"

  ami_id          = "ami-0c55b159cbfafe1f0"
  name            = "worker-${var.environment}"
  subnet_id       = aws_subnet.private.id
  security_group_ids = [aws_security_group.worker.id]
}
```

Ventajas:
- `web` y `worker` comparten la misma lógica (son el mismo módulo)
- Cambias algo en el módulo → ambos se actualizan
- Cada uno tiene sus propias variables

## 4. Módulos del Registry (reutilizar lo que ya existe)

No todo módulo tienes que escribirlo tú. El [Terraform Registry](https://registry.terraform.io/) tiene módulos oficiales y de la comunidad:

```hcl
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "5.0.0"

  name = "my-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["us-east-1a", "us-east-1b"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]

  enable_nat_gateway = true
  enable_vpn_gateway = false
}
```

Así creas una VPC completa con subnets, route tables, NAT gateway, y más, en ~20 líneas en vez de 200.

**Regla práctica**: si estás escribiendo un módulo de VPC, S3, RDS o EKS desde cero — primero revisa el Registry. El 90% de las veces ya existe.

## 5. Estructura de proyecto recomendada

A medida que el proyecto crece, usa esta estructura (adaptada de la referencia de HashiCorp):

```
infrastructure/
├── environments/
│   ├── dev/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── outputs.tf
│   │   ├── terraform.tfvars
│   │   └── backend.hcl
│   ├── staging/
│   │   └── ...
│   └── prod/
│       └── ...
├── modules/
│   ├── ec2-instance/
│   ├── vpc/
│   ├── rds/
│   └── iam/
├── global/               ← recursos compartidos (IAM, Route53)
│   └── main.tf
└── build/
    └── buildspec.yml     ← para CodeBuild
```

Cada entorno es independiente:

```bash
cd environments/dev
terraform init -backend-config=backend.hcl
terraform plan -var-file=terraform.tfvars
terraform apply -var-file=terraform.tfvars
```

## 6. Módulos remotos (con source desde git o HTTP)

Además de rutas locales, puedes cargar módulos desde:

```hcl
# Desde git
module "vpc" {
  source = "git::https://github.com/org/terraform-modules.git//vpc?ref=v1.2.0"
}

# Desde un bucket S3
module "lambda" {
  source = "s3::https://s3.amazonaws.com/mi-bucket/modules/lambda.zip"
}

# Desde el Registry (con o sin namespace)
module "bucket" {
  source  = "hashicorp/consul/aws"
  version = "0.11.0"
}
```

**Ventaja de módulos remotos**: versionados, compartidos entre equipos, y no duplicas código en cada repo.

## 7. Cómo pensar al diseñar módulos

| Buena señal de módulo | Mala señal |
|----------------------|------------|
| Tiene una responsabilidad clara (una VPC, un EC2, un RDS) | Hace "de todo" (crea VPC + instala software + configura DNS) |
| Expone variables útiles con defaults sensatos | No expone variables, o expone 50 con nombres genéricos |
| Documenta cada variable y output | Sin descripciones |

Estos criterios son principios de [[Code Quality]] aplicados a infraestructura.

**No modularices demasiado pronto**. Empieza con un root module sencillo. Cuando veas que repites el mismo bloque de recursos 3+ veces, extráelo a un módulo.

## 8. Common Mistakes

- **Módulo sin outputs**: usas un módulo y luego no puedes acceder a los valores de los recursos que creó. Siempre expone outputs útiles.
- **Source local sin path relativo**: `source = "modules/ec2"` es relativo al sistema de archivos del invocador. Usa `"./modules/ec2"` para ser explícito.
- **Versionar módulos locales**: si todos apuntan a `./modules/ec2`, un cambio en el módulo afecta a todos los entornos a la vez. Los entornos deberían apuntar a tags o commits específicos.
- **No bloquear versiones de módulos del Registry**: `source = "terraform-aws-modules/vpc/aws"` sin `version` → en el próximo init obtienes la versión más reciente, que podría romper tu código. Siempre fija la versión.

## Check Your Understanding

1. Si dos entornos (dev, prod) apuntan al mismo módulo local, y modificas el módulo, ¿qué entorno se actualiza primero?
2. ¿Por qué los módulos del Registry especifican `source` con un namespace (`terraform-aws-modules/vpc/aws`) en vez de una URL de git?
3. En la estructura de proyecto recomendada, ¿por qué cada entorno tiene su propio directorio en vez de usar workspaces?

## Where to Go Next

- [[AWS CodeBuild with Terraform]] — pipeline CI/CD que usa esta estructura de proyecto
- [[Terraform State & Backends]] — cada entorno necesita su propio backend
- [[Infrastructure as Code Fundamentals]] — principios de diseño que aplican a módulos
- [[Git]] — cómo versionar módulos con tags
