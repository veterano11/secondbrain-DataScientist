---
tags:
  - encryption
  - access-control
  - iam
  - kms
  - secrets-management
  - hipaa
  - rbac
  - abac
status: seedling
created: 2026-06-28
---

## 1. Escenario de aprendizaje

Tu pipeline de datos procesa información médica (HIPAA). El equipo guarda archivos Parquet en S3 con datos de pacientes. Alguien descubre que el bucket tiene acceso público. Necesitas implementar encriptación en reposo y tránsito, IAM policies estrictas, y un sistema de secrets management para que esto no ocurra.

## 2. Requisitos previos

- AWS básico: S3, IAM, KMS
- CLI: bash, aws-cli, jq
- Conceptos generales de seguridad: cifrado simétrico/asimétrico, TLS
- Familiaridad con [[Infrastructure as Code Fundamentals]]

## 3. Encryption at rest

Cifrar datos almacenados asegura que, incluso si alguien accede al disco o al bucket, no pueda leerlos sin la clave.

### 3.1 AES-256 y KMS

```bash
# Crear una clave KMS simétrica
aws kms create-key --description "Data pipeline encryption key" \
  --key-usage ENCRYPT_DECRYPT --key-spec SYMMETRIC_DEFAULT

# Crear un alias para referenciarla fácilmente
aws kms create-alias \
  --alias-name alias/data-pipeline-key \
  --target-key-id $(aws kms list-keys --query "Keys[0].KeyId" --output text)
```

**Salida esperada:**
```
{
    "KeyMetadata": {
        "KeyId": "arn:aws:kms:us-east-1:123456789012:key/abc12345-...",
        "KeySpec": "SYMMETRIC_DEFAULT"
    }
}
```

### 3.2 Server-side encryption en S3

```bash
# Habilitar SSE-KMS por defecto en un bucket
aws s3api put-bucket-encryption \
  --bucket datos-medicos-prod \
  --server-side-encryption-configuration '{
    "Rules": [
      {
        "ApplyServerSideEncryptionByDefault": {
          "SSEAlgorithm": "aws:kms",
          "KMSMasterKeyID": "alias/data-pipeline-key"
        }
      }
    ]
  }'

# Subir archivo con encriptación explícita
aws s3 cp pacientes.parquet s3://datos-medicos-prod/ \
  --sse aws:kms --sse-kms-key-id alias/data-pipeline-key
```

**Salida esperada:**
```
upload: ./pacientes.parquet to s3://datos-medicos-prod/pacientes.parquet
```

### 3.3 Encriptación en RDS y EBS

```sql
-- RDS: habilitar encryption al crear la instancia (no se puede después)
CREATE DB INSTANCE pacientes-db
  STORAGE ENCRYPTED  -- habilita AES-256 automático
  KMS KEY ID alias/data-pipeline-key;
```

## 4. Encryption in transit

Los datos en movimiento (red) deben ir cifrados con TLS. Esto aplica a conexiones entre servicios y a la comunicación cliente-servidor [[Terraform Foundations]].

```bash
# Verificar TLS de un endpoint
openssl s_client -connect datos-api.empresa.cl:443 -tls1_2

# Forzar TLS 1.2+ en política de bucket S3
aws s3api put-bucket-policy --bucket datos-medicos-prod --policy '{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Deny",
    "Principal": "*",
    "Action": "s3:*",
    "Resource": "arn:aws:s3:::datos-medicos-prod/*",
    "Condition": {
      "Bool": {"aws:SecureTransport": "false"}
    }
  }]
}'
```

**Salida esperada:**
```
depth=0 CN = datos-api.empresa.cl
verify return:1
---
Certificate chain
 0 s:CN = datos-api.empresa.cl
   i:C = US, O = Amazon, OU = RDS
```

### 4.1 mTLS entre servicios

```yaml
# docker-compose con mTLS
services:
  spark-service:
    image: spark:3.5
    environment:
      SPARK_SSL_ENABLED: "true"
      SPARK_SSL_KEYSTORE: /certs/server.keystore.jks
      SPARK_SSL_KEYSTORE_PASSWORD: ${KEYSTORE_PASS}
    volumes:
      - ./certs:/certs:ro
```

## 5. IAM policies

Las políticas IAM definen quién puede hacer qué sobre qué recursos. El principio de [[Data Classification & Governance]] se traduce en políticas.

### 5.1 Resource-based vs Identity-based

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "LeastPrivilegeAccess",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject"
      ],
      "Resource": "arn:aws:s3:::datos-medicos-prod/etl/*",
      "Condition": {
        "StringEquals": {
          "s3:x-amz-server-side-encryption": "aws:kms"
        },
        "IpAddress": {
          "aws:SourceIp": "10.0.0.0/16"
        }
      }
    }
  ]
}
```

```bash
# Adjuntar política a un role
aws iam put-role-policy \
  --role-name data-engineer-role \
  --policy-name s3-medical-access \
  --policy-document file://policy.json
```

### 5.2 Policy conditions

```json
{
  "Condition": {
    "StringEquals": {
      "aws:RequestTag/classification": "restricted"
    },
    "Bool": {
      "aws:MultiFactorAuthPresent": "true"
    },
    "DateGreaterThan": {
      "aws:CurrentTime": "2026-01-01T00:00:00Z"
    }
  }
}
```

**Salida esperada (fallo si no cumple condiciones):**
```
An error occurred (AccessDenied) when calling the GetObject operation: 
User is not authorized to access this resource
```

## 6. RBAC vs ABAC

RBAC asigna permisos por rol. ABAC usa atributos (tags, tiempo, ubicación).

| Aspecto | RBAC | ABAC |
|---------|------|------|
| Base | Rol del usuario | Atributos del usuario/recurso/entorno |
| Escala | N rojos × M permisos | Políticas genéricas reutilizables |
| Ejemplo | data_scientist puede leer confidencial | Si usuario.tag.departamento = recurso.tag.departamento |
| Ventaja | Simple de entender | Escala mejor en equipos grandes |
| Desventaja | Explosión de roles | Curva de aprendizaje más alta |

```bash
# ABAC con tags en AWS
aws sts assume-role \
  --role-arn "arn:aws:iam::123456789012:role/data-access-${classification}" \
  --role-session-name "data-pipeline" \
  --tags Key=classification,Value=restricted
```

## 7. Secrets management

Nunca, bajo ninguna circunstancia, pongas credenciales en código. [[AWS CodeBuild with Terraform]] puede integrar Secrets Manager para pasar credenciales de forma segura.

### 7.1 AWS Secrets Manager

```bash
# Almacenar secreto
aws secretsmanager create-secret \
  --name prod/db/password \
  --secret-string '{"username":"admin","password":"M1Cr0s3rv1c3!!"}'

# Rotación automática con Lambda
aws secretsmanager rotate-secret \
  --secret-id prod/db/password \
  --rotation-lambda-arn arn:aws:lambda:us-east-1:123456789012:function:rotate-db-creds \
  --rotation-rules AutomaticallyAfterDays=30
```

### 7.2 HashiCorp Vault

```bash
# Escribir y leer secreto
vault kv put secret/data-pipeline/db \
  username=admin \
  password="$(openssl rand -base64 32)"

vault kv get secret/data-pipeline/db
```

**Salida esperada:**
```
====== Data ======
Key         Value
---         -----
username    admin
password    a9f2b8c7d6e5f4a3b2c1d0e9f8a7b6c5d4e3f2a1b0c9d8e7f6a5b4c3d2e1f
```

### 7.3 Uso seguro desde aplicación

```python
import boto3
from botocore.config import Config

def get_db_credentials():
    """
    Obtiene credenciales desde Secrets Manager.
    Nunca imprime ni loggea el secreto.
    """
    session = boto3.session.Session()
    client = session.client(
        service_name="secretsmanager",
        config=Config(retries={"max_attempts": 3}),
    )
    response = client.get_secret_value(SecretId="prod/db/password")
    return json.loads(response["SecretString"])

creds = get_db_credentials()
engine = create_engine(
    f"postgresql://{creds['username']}:{creds['password']}@prod-db:5432/pacientes"
)
```

## 8. Common Mistakes

1. **IAM policies con `Action: "*"`**: Es el equivalente a dar las llaves de la casa. Siempre limita las acciones a las mínimas necesarias.
2. **Secrets en código fuente**: Aparecen en logs, repositorios, Slack. Usa secrets manager siempre. Si ves una password en un `git commit`, es una emergencia.
3. **No rotar keys**: Usar la misma clave KMS por años. La rotación automática (anual o mensual) limita el daño si una clave se compromete.
4. **Bucket público con datos sensibles**: S3 Block Public Access debe estar habilitado a nivel de organización. Haz `aws s3control put-public-access-block` en todas las cuentas.
5. **Encriptación solo en reposo, no en tránsito**: Si interceptan la red, los datos viajan en texto claro. TLS debe ser obligatorio con `aws:SecureTransport` condition.

## Resumen

La seguridad de datos requiere tres capas: encriptación en reposo (AES-256 con KMS), encriptación en tránsito (TLS 1.2+, mTLS), y control de acceso (IAM con least privilege). Las políticas pueden ser resource-based o identity-based, con condiciones como MFA, IP, y tags. RBAC funciona para equipos pequeños, ABAC escala mejor. Los secrets (passwords, tokens) nunca van en código — usa Secrets Manager o Vault con rotación automática. La combinación de estas prácticas es obligatoria para cumplir HIPAA, GDPR y estándares de [[Privacy Regulations]].

## Comprueba tu Conocimiento

1. ¿Qué diferencia hay entre SSE-S3, SSE-KMS y SSE-C?
2. ¿Qué condición IAM usarías para forzar TLS en S3?
3. ¿Cuándo conviene ABAC en vez de RBAC?
4. ¿Por qué es peligroso tener `Action: "*"` en una IAM policy?
5. ¿Qué significa "rotación automática" de secrets y por qué es importante?

<!--
1. SSE-S3: AWS maneja las claves (AES-256). SSE-KMS: tú controlas las claves via KMS con auditoría. SSE-C: tú provees tu propia clave.
2. `"Bool": {"aws:SecureTransport": "false"}` con Effect: Deny para rechazar conexiones no TLS.
3. ABAC conviene cuando hay muchos roles y recursos con atributos dinámicos (departamento, proyecto, entorno). Escala mejor que RBAC cuando tienes cientos de combinaciones.
4. `Action: "*"` permite cualquier operación. Si un atacante obtiene esas credenciales, puede hacer todo: leer, escribir, eliminar, modificar políticas. Rompe el principio de least privilege.
5. Rotar automáticamente significa reemplazar el secreto periódicamente sin intervención humana. Limita la ventana de exposición si un secreto se filtra y reduce el riesgo de credenciales estáticas.
-->

## ¿Dónde ir Siguente?

- [[Data Classification & Governance]]
- [[Privacy Regulations]]
- [[Infrastructure as Code Fundamentals]]
- [[Terraform Foundations]]
- [[Secure ML & Incident Response]]
- [[AWS CodeBuild with Terraform]]
- [[CLI & Productivity]]
