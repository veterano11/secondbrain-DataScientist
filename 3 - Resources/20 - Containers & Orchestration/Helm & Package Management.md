---
tags:
  - kubernetes
  - helm
  - devops
  - package-management
status: seedling
created: 2026-06-28
---

# Helm & Package Management

## Escenario de aprendizaje

Tenés 10 microservicios, cada uno con 5 manifiestos YAML de K8s (Deployment, Service, ConfigMap, HPA, Ingress). Son 50 archivos. Cuando cambia una variable de entorno (ej: `DB_HOST`), tenés que editar 50 archivos. Y cuando pasás de dev a prod, los valores son distintos pero la estructura es la misma. Helm empaqueta y parametriza la configuración de K8s para que un solo chart genere los 50 YAMLs con los valores correctos para cada ambiente.

## 1. ¿Qué es Helm?

Helm es el gestor de paquetes de Kubernetes. Sus conceptos clave:

| Concepto | Descripción |
|----------|-------------|
| **Chart** | Paquete de archivos YAML templateados que describen una aplicación |
| **Release** | Una instancia de un chart corriendo en un cluster |
| **Repository** | Lugar donde se almacenan y comparten charts |
| **Templating** | Motor Go templates que genera YAML dinámicamente |

```bash
$ helm repo add bitnami https://charts.bitnami.com/bitnami
$ helm install mi-app bitnami/nginx
$ helm list
$ helm uninstall mi-app
```

Sin Helm cada microservicio requiere 5+ archivos YAML repetitivos. Con Helm, es un solo chart con valores parametrizados, similar a como [[Terraform Modules & Project Structure]] modulariza infraestructura.

## 2. Estructura de un chart

```
mi-chart/
├── Chart.yaml          # Metadata: name, version, description, dependencies
├── values.yaml         # Valores por defecto
├── charts/             # Dependencias (sub-charts)
├── templates/          # Manifiestos Go templates
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── _helpers.tpl    # Named templates reutilizables
│   └── hpa.yaml
└── .helmignore         # Archivos a excluir
```

**Chart.yaml:**

```yaml
apiVersion: v2
name: mi-api
version: 0.1.0
description: API REST templateada
appVersion: "1.0"
```

**_helpers.tpl:**

```yaml
{{- define "mi-api.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 }}
{{- end }}

{{- define "mi-api.labels" -}}
app.kubernetes.io/name: {{ include "mi-api.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
```

Las `named templates` evitan repetir lógica en cada archivo, manteniendo el chart DRY.

## 3. Templates

Helm usa Go templates con funciones adicionales de Sprig:

```yaml
# templates/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ include "mi-api.name" . }}-config
  labels: {{- include "mi-api.labels" . | nindent 4 }}
data:
  LOG_LEVEL: {{ .Values.logLevel | default "info" | quote }}
  DB_HOST: {{ .Values.database.host }}
  DB_PORT: {{ .Values.database.port | toString }}
  FEATURES: {{ .Values.features | join "," }}
```

**Control flow:**

```yaml
{{- if .Values.metrics.enabled }}
apiVersion: v1
kind: Service
metadata:
  annotations:
    prometheus.io/scrape: "true"
{{- end }}
```

**Range (iterar listas):**

```yaml
env:
{{- range $key, $val := .Values.envVars }}
- name: {{ $key }}
  value: {{ $val | quote }}
{{- end }}
```

El debugging local es clave:

```bash
$ helm template ./mi-chart --values values-dev.yaml
$ helm lint ./mi-chart
$ helm diff upgrade mi-release ./mi-chart  # plugin helm-diff
```

Combiná templates con [[CLI & Productivity]] para alias y scripts que aceleren el iteration loop.

## 4. Values por ambiente

```yaml
# values.yaml (defaults)
replicaCount: 1
image:
  repository: mi-app
  tag: latest
resources:
  requests:
    cpu: 100m
    memory: 128Mi
```

```yaml
# values-prod.yaml
replicaCount: 5
image:
  tag: 0.1.0
resources:
  requests:
    cpu: 500m
    memory: 512Mi
  limits:
    cpu: 1
    memory: 1Gi
ingress:
  enabled: true
  host: api.midominio.com
```

```bash
$ helm install mi-release ./mi-chart --values values-prod.yaml
$ helm upgrade mi-release ./mi-chart --values values-prod.yaml --set image.tag=0.2.0
```

El flag `--set` es ideal para CI/CD donde el tag de la imagen viene del pipeline. En [[CI-CD & GitOps]] es común pasar el commit SHA como `--set image.tag=$COMMIT_SHA`.

## 5. Releases

El ciclo de vida de una release:

```bash
$ helm install mi-app ./mi-chart            # crear release
$ helm upgrade mi-app ./mi-chart -f prod.yaml  # actualizar
$ helm rollback mi-app 1                    # volver a revisión 1
$ helm history mi-app                       # ver revisiones
$ helm list                                 # todas las releases
$ helm uninstall mi-app                     # eliminar release
```

Helm guarda el historial de revisiones en Secrets dentro del namespace:

```bash
$ kubectl get secret -n default | grep sh.helm
sh.helm.release.v1.mi-app.v1
sh.helm.release.v1.mi-app.v2
```

**Rollback strategy:** siempre verificá que el `helm rollback` funciona antes de necesitarlo. Documentalo en el runbook, similar a los rollbacks en [[Kubernetes Fundamentals]].

## 6. Dependencias

Un chart puede depender de otros charts (ej: tu app depende de PostgreSQL y Redis):

```yaml
# Chart.yaml
dependencies:
- name: postgresql
  version: "12.x"
  repository: https://charts.bitnami.com/bitnami
  condition: postgresql.enabled
- name: redis
  version: "17.x"
  repository: https://charts.bitnami.com/bitnami
  condition: redis.enabled
```

```bash
$ helm dependency update ./mi-chart
$ helm dependency build ./mi-chart
```

Las dependencias se descargan a la carpeta `charts/`. Usá `condition` para habilitar/deshabilitar dependencias según el ambiente (PostgreSQL en dev, RDS en prod). Es el equivalente a los módulos de [[Terraform Modules & Project Structure]] pero para K8s.

## 7. Common Mistakes

- **No versionar charts**: cada cambio debe incrementar `version` en Chart.yaml; `helm list` muestra la versión desplegada
- **Values hardcodeadas**: si ves `value: "produccion"` en un template, algo está mal — todo debe ser parametrizable
- **No usar helm-diff**: hacé `helm diff upgrade` antes de aplicar cambios para revisar qué va a cambiar
- **Ignorar el orden de upgrade**: `helm upgrade` aplica todos los manifiestos en orden; los errors silenciosos pueden dejar el release roto
- **Falta de rollback testing**: probá `helm rollback` en staging antes de producción
- **Sobreescribir valores globales**: usá `--set` con cuidado; los overrides se acumulan y pueden ser difíciles de rastrear

## Resumen

1. Helm es el gestor de paquetes para K8s: charts, releases, repositories y Go templates
2. Un chart tiene `Chart.yaml`, `values.yaml`, `templates/` y `_helpers.tpl` para lógica reutilizable
3. Go templates con Sprig generan YAML: `{{ .Values.* }}`, `if/else`, `range`, `nindent`
4. Los valores por ambiente se manejan con distintos `values-*.yaml` y `--set`
5. `helm install`, `upgrade`, `rollback`, `list`, `uninstall` manejan el ciclo de vida de releases
6. Las dependencias permiten empaquetar charts anidados (PostgreSQL, Redis) con condiciones por ambiente

## Check Your Understanding

1. ¿Qué diferencia hay entre un chart y una release? <!-- Un chart es la plantilla; una release es una instancia del chart desplegada en el cluster. -->
2. ¿Para qué sirve el archivo `_helpers.tpl`? <!-- Para definir named templates reutilizables que evitan repetir lógica en varios manifiestos. -->
3. ¿Cómo harías para deployar la misma app con configuraciones distintas en dev y prod? <!-- Dos archivos values-dev.yaml y values-prod.yaml, pasados con --values. -->
4. ¿Qué comando usarías para ver qué cambios va a aplicar `helm upgrade` sin realmente aplicarlos? <!-- helm diff upgrade (requiere plugin helm-diff) o helm template para ver el YAML generado. -->
5. ¿Por qué es importante versionar los charts? <!-- Para tener trazabilidad de qué versión está desplegada y poder hacer rollback a una versión específica. -->

## Where to Go Next

- [[Kubernetes Fundamentals]]
- [[K8s for ML Workloads]]
- [[CI-CD & GitOps]]
- [[CLI & Productivity]]
- [[Git]]
- [[Terraform Modules & Project Structure]]
- [[Docker Fundamentals]]
