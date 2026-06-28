---
tags:
  - kubernetes
  - k8s
  - container
  - orchestration
  - devops
status: seedling
created: 2026-06-28
---

# Kubernetes Fundamentals

## Escenario de aprendizaje

Tenés 5 microservicios en Docker. Los corrés con `docker compose` en desarrollo. Pero en producción necesitás escalar cada servicio según demanda, actualizar versiones sin downtime, reiniciar containers cuando fallan, y distribuir la carga entre varias máquinas. Docker Compose no escala a ese nivel. Necesitás Kubernetes: un orquestador que administra containers en un cluster de nodos.

## 1. Arquitectura

Kubernetes tiene dos planos:

**Control Plane** (el cerebro del cluster):
- **API Server**: punto de entrada; procesa requests REST (vía `kubectl` o SDK)
- **Scheduler**: asigna Pods a nodos según recursos disponibles
- **etcd**: base de datos clave-valor que guarda el estado del cluster (el "source of truth")
- **Controller Manager**: loops que mantienen el estado deseado (si un Pod muere, crea otro)

**Worker Nodes** (las máquinas que corren apps):
- **kubelet**: agente que recibe órdenes del API Server y maneja los Pods
- **kube-proxy**: red y balanceo de carga entre Pods
- **Container Runtime**: Docker, containerd, CRI-O — corre los containers

Esta separación es clave cuando comparás con [[Docker Fundamentals]] donde solo hay un container runtime local.

## 2. Pods

Un **Pod** es la unidad mínima. Puede tener uno o varios containers que comparten red y almacenamiento:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: ml-api
spec:
  containers:
  - name: api
    image: ml-api:0.1.0
    ports:
    - containerPort: 8000
```

**Sidecar containers**: containers auxiliares dentro del mismo Pod (ej: log collector, proxy, health checker). Son parte del patrón estándar para [[K8s for ML Workloads]].

**Init containers**: corren y terminan antes que el container principal (ej: migraciones de DB, setup de archivos).

## 3. Deployments

Un **Deployment** maneja Pods replicados con actualizaciones controladas:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api
  template:
    metadata:
      labels:
        app: api
    spec:
      containers:
      - name: api
        image: api:0.1.0
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 256Mi
```

- **ReplicaSet**: creado automáticamente por el Deployment; mantiene N réplicas de un Pod template
- **Rolling update**: actualiza Pods de a uno; `kubectl rollout status deployment/api`
- **Rollback**: `kubectl rollout undo deployment/api` si la nueva versión falla
- **Estrategias**: `RollingUpdate` (default) o `Recreate` (mata todo y crea nuevo — downtime)

```bash
$ kubectl apply -f deployment.yaml
$ kubectl rollout status deployment/api
$ kubectl rollout undo deployment/api --to-revision=2
```

**Salida esperada:**
```
deployment.apps/api created
Waiting for deployment "api" rollout to finish: 1 of 3 updated replicas...
deployment "api" successfully rolled out
```

## 4. Services

Los Pods tienen IPs efímeras. Un **Service** provee una IP/ DNS estable y balanceo de carga:

| Tipo | Descripción |
|------|-------------|
| **ClusterIP** | IP interna accesible solo dentro del cluster |
| **NodePort** | Abre un puerto en cada nodo (30000-32767) |
| **LoadBalancer** | Crea un balanceador externo (cloud provider) |

```yaml
apiVersion: v1
kind: Service
metadata:
  name: api-service
spec:
  selector:
    app: api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

Los `selector` labels conectan el Service con los Pods del Deployment. Es el mecanismo de service discovery que reemplaza la red automática de [[Docker Compose & Multi-Service]].

## 5. ConfigMap y Secrets

Separar configuración del código:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: api-config
data:
  LOG_LEVEL: info
  DB_HOST: postgres-clusterip
---
apiVersion: v1
kind: Secret
metadata:
  name: db-secret
type: Opaque
stringData:
  DB_PASSWORD: supersecreto
```

```yaml
# en el Deployment
containers:
- name: api
  envFrom:
  - configMapRef:
      name: api-config
  - secretRef:
      name: db-secret
```

Los Secrets se almacenan en etcd (opcionalmente encriptados). Nunca los versiones en git — usá [[Helm & Package Management]] con valores externos o un vault externo como [[Infrastructure as Code Fundamentals]] sugiere.

## 6. kubectl — comandos diarios

```bash
$ kubectl get pods -n default -w
$ kubectl describe pod ml-api-7d4f8b9c6-xk9w2
$ kubectl logs deployment/api -f --tail=100
$ kubectl exec -it deploy/api -- bash
$ kubectl apply -f deployment.yaml
$ kubectl delete pod ml-api-7d4f8b9c6-xk9w2
$ kubectl get svc,deploy,pod
$ kubectl top pods  # métricas de CPU/memoria
```

`describe` es tu mejor amigo para debugging — muestra eventos, condiciones, y por qué un Pod no arranca. La combinación con [[CLI & Productivity]] (alias, plugins como `kubectx`) acelera el día a día.

## 7. Common Mistakes

- **No setear resource requests/limits**: un Pod puede consumir todo el nodo y matar a los demás
- **Usar `:latest` tag**: K8s no hace `docker pull` si ya tiene la imagen local; siempre usá tags específicos (SHA)
- **Sin liveness/readiness probes**: K8s no sabe si tu app está viva o lista para recibir tráfico
- **Exponer Secrets en logs/envars**: los Secrets aparecen en `describe` si no se filtran
- **Olvidar namespaces**: mezclar servicios de distintos equipos en `default` es caos
- **No configurar PVC (Persistent Volume Claims)**: los datos efímeros se pierden al reiniciar un Pod

```yaml
# Ejemplo de probes
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 10
readinessProbe:
  httpGet:
    path: /ready
    port: 8000
```

Las probes son esenciales para [[Model Monitoring]] en producción: detectan apps colgadas y las reinician automáticamente.

## Resumen

1. K8s orquesta containers en un cluster con control plane (API, scheduler, etcd) y worker nodes
2. Pods son la unidad mínima; pueden tener sidecars e init containers
3. Deployments manejan réplicas, rolling updates y rollbacks
4. Services proveen IPs estables y balanceo; hay ClusterIP, NodePort y LoadBalancer
5. ConfigMap y Secrets separan configuración del código de la imagen
6. `kubectl` es la CLI principal: apply, get, describe, logs, exec
7. Siempre setear resource limits, probes y tags específicos — nunca `latest`

## Check Your Understanding

1. ¿Cuál es la diferencia entre un Pod y un Deployment? <!-- Un Pod es una unidad atómica; un Deployment maneja réplicas, actualizaciones y rollbacks del Pod template. -->
2. ¿Para qué sirve un Service de tipo LoadBalancer? <!-- Expone los Pods externamente creando un balanceador de carga del cloud provider. -->
3. ¿Qué pasa si un Deployment tiene `replicas: 3` y un Pod se cae? <!-- El ReplicaSet (manejado por el Deployment) crea un nuevo Pod para mantener 3 réplicas. -->
4. ¿Por qué es peligroso usar `image: api:latest` en un Deployment? <!-- K8s no actualiza la imagen si ya está presente localmente; no hay trazabilidad. -->
5. ¿Qué mide un liveness probe vs un readiness probe? <!-- Liveness: ¿el container está vivo? Readiness: ¿está listo para recibir tráfico? -->

## Where to Go Next

- [[Docker Fundamentals]]
- [[Docker Compose & Multi-Service]]
- [[K8s for ML Workloads]]
- [[Helm & Package Management]]
- [[CI-CD & GitOps]]
- [[Model Monitoring]]
