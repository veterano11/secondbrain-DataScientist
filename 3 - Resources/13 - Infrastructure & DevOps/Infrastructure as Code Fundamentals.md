---
tags: [iac, fundamentals, devops, terraform]
status: growing
created: 2026-06-28
---

# Infrastructure as Code Fundamentals

## Por dónde empezamos: el problema

Imagina que tienes que levantar un servidor web en AWS. Hoy haces clic en la consola: lanzas una EC2, configuras el security group, asignas una IP elástica, creas un bucket S3 para logs. Todo funciona.

Tres meses después, tu equipo necesita replicar exactamente la misma infraestructura para staging. ¿Te acuerdas de cada clic que hiciste? ¿O tienes que entrar a la consola, inspeccionar cada recurso, e imitarlo a mano?

Ese es el problema que resuelve Infrastructure as Code (IaC).

## ¿Qué es Infrastructure as Code?

Es la práctica de describir tu infraestructura — servidores, redes, bases de datos, permisos — en archivos de configuración legibles por humanos y por máquinas. Esos archivos son tu "código fuente" de infraestructura: los versionas en git, los revisas en PRs, los ejecutas de forma automatizada.

> **Analogía**: Así como Docker describe un contenedor en un Dockerfile, Terraform describe tu infraestructura cloud en archivos `.tf`.

## ¿Por qué importa? Escenario concreto

Sin IaC (click-ops):

```
dev     → alguien creó una t2.micro hace meses, nadie sabe cómo
staging → parecido pero no igual, la BD es más grande
prod    → "es como staging pero nadie se atreve a tocarlo"
```

Con IaC:

```
main.tf → describe exactamente la infraestructura
dev     → terraform apply con variables de dev
staging → terraform apply con variables de staging
prod    → terraform apply con variables de prod
```

**Los tres entornos son idénticos en estructura**. Lo único que cambia son los parámetros (tamaño de instancia, número de réplicas, etc.).

## Declarativo vs Imperativo

Aquí está la decisión de diseño más importante en IaC.

### Imperativo (Ansible, CloudFormation — en parte)

Dices **cómo** llegar al estado deseado, paso a paso:

```
1. Crear security group
2. Lanzar instancia EC2
3. Asignar IP elástica
```

Si alguien borra el security group fuera del script, el script falla al llegar al paso 1 porque ya no existe. O peor: intenta crearlo de nuevo y choca.

### Declarativo (Terraform, Pulumi)

Dices **qué** estado quieres, no cómo llegar:

```
Quiero: una EC2, un security group, una IP elástica, conectados así.
```

Terraform compara tu declaración con lo que existe realmente y calcula los pasos necesarios. Si alguien borró el security group, Terraform lo detecta (el estado real diverge del deseado) y lo recrea automáticamente.

```hcl
# Terraform: declaras el resultado, no los pasos
resource "aws_instance" "web" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t2.micro"
}

resource "aws_security_group" "web_sg" {
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
```

Pruébalo: no ves `create_security_group()` ni `run_instances()`. Solo ves "quiero que exista una instancia con estas propiedades, y un security group con estas reglas".

## Idempotencia: la propiedad mágica

Una operación es idempotente si ejecutarla una vez o varias produce el mismo resultado.

- `terraform apply` es idempotente: la primera vez crea los recursos, la segunda vez no hace nada (ya están en el estado deseado).
- Un script shell que hace `mkdir /data` no es idempotente: la segunda vez falla porque el directorio ya existe.

Terraform CLI es una herramienta central de [[CLI & Productivity]]: con un solo comando declarás, aplicás y verificás infraestructura.

> **Pregunta de verificación**: antes de seguir, piensa por qué la idempotencia es crucial para CI/CD. Si tu pipeline ejecuta terraform apply en cada deploy, ¿qué pasa si alguien ya lo ejecutó?

Respuesta: no pasa nada. Terraform detecta que el estado real coincide con el deseado y no hace cambios. Eso es idempotencia.

## State: el cerebro de Terraform

Terraform guarda un archivo de **estado** (terraform.tfstate) que mapea los recursos declarados en tu código contra los recursos reales en AWS/GCP/Azure.

```
Código (main.tf): "quiero una instancia EC2"
        ↓
State (tfstate): "la instancia EC2 que creé tiene id i-12345, su IP es x.x.x.x"
        ↓
Realidad (AWS): la instancia i-12345 existe, su IP es x.x.x.x
```

Cada vez que ejecutas `terraform plan`, Terraform:
1. Lee tu código → estado deseado
2. Lee el state → recursos que ya existen
3. Consulta AWS → estado real
4. Calcula la diferencia → plan de acción

Si el state se pierde, Terraform no sabe qué recursos creó. **El state es tu activo más valioso**.

## GitOps: infraestructura como código + git

El flujo de trabajo moderno combina IaC con GitOps:

```
1. Haces cambios en main.tf
2. Abres un Pull Request
3. CI ejecuta terraform plan → revisas los cambios
4. Al hacer merge, CI ejecuta terraform apply
```

Ventajas:
- **Historial**: cada cambio a infraestructura queda registrado en git
- **Revisión**: nadie hace `terraform apply` sin [[Code Quality|revisión de código]] por otro miembro del equipo
- **Rollback**: `git revert` + `terraform apply` vuelve al estado anterior
- **Auditoría**: sabes quién cambió qué y cuándo

## Common Pitfalls

- **No versionar el state**: guardar terraform.tfstate solo en local. Si pierdes el archivo, pierdes el control de tu infraestructura. Remedio: backend remoto (S3 + DynamoDB).
- **Editar recursos a mano desde la consola AWS**: haces un cambio fuera de Terraform → el código ya no refleja la realidad. Terraform lo revertirá en el próximo apply.
- **Mezclar declarativo e imperativo**: tener algunos recursos en Terraform y otros creados a mano. No sabes qué existe ni por qué. Todo debe estar en código.
- **No probar en un [[Virtual Environments|entorno aislado]]**: hacer terraform apply directo a producción sin pasar por dev/staging. Así es como se borran bases de datos.

## Check Your Understanding

1. Un colega modificó el security group desde la consola AWS. Luego tú ejecutas `terraform apply`. ¿Qué esperas que pase?
2. ¿Por qué un script de bash con `aws ec2 run-instances` **no** es IaC, aunque automatice la creación?
3. Si tu state file se corrompe y no tienes backup, ¿qué opciones tienes para recuperar el control de la infraestructura?

## Where to Go Next

- [[Terraform Foundations]] — escribir tu primer main.tf desde cero
- [[Terraform State & Backends]] — cómo proteger y compartir el state
- [[AWS CodeBuild with Terraform]] — automatizar todo en CI/CD
- [[Git]] — GitOps requiere dominar git para PRs y rollbacks
