---
tags:
  - kubernetes
  - k8s
  - machine-learning
  - mlops
  - gpu
status: seedling
created: 2026-06-28
---

# K8s for ML Workloads

## Escenario de aprendizaje

Tu equipo de Data Science tiene 3 GPUs en el cluster. Juan entrena un modelo y ocupa las 3 GPUs por 4 horas. María quiere ejecutar un experimento rápido que usa 1 GPU, pero todas están ocupadas. Además, el entrenamiento de Juan falla a las 3 horas porque un nodo se reinicia — y no hay checkpointing. Necesitás Kubernetes para ML: Jobs que corren hasta completarse, scheduling inteligente de GPUs, y herramientas como Kubeflow para orquestar pipelines de ML completos.

## 1. K8s Jobs y CronJobs

Los Deployments asumen que el container corre "para siempre". Los **Jobs** son para tareas run-to-completion:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: entrenamiento
spec:
  completions: 3
  parallelism: 2
  backoffLimit: 4
  ttlSecondsAfterFinished: 3600
  template:
    spec:
      containers:
      - name: trainer
        image: ml-trainer:0.2.0
        command: ["python", "train.py"]
      restartPolicy: Never
```

- **completions**: cuántos Pods exitosos necesita el Job
- **parallelism**: cuántos Pods correr en paralelo
- **backoffLimit**: reintentos antes de marcar el Job como Failed
- **ttlSecondsAfterFinished**: elimina automáticamente Pods viejos (evita [[Model Monitoring]] saturación)

**CronJobs** agregan un schedule tipo cron para ejecuciones periódicas (reentrenamiento nocturno, batch scoring):

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: reentrenamiento-diario
spec:
  schedule: "0 2 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: trainer
            image: ml-trainer:0.2.0
          restartPolicy: Never
```

Los Jobs son la base de [[ML Pipelines]]: cada paso del pipeline (feature engineering, entrenamiento, evaluación) es un Job que consume y produce artefactos.

## 2. GPU Scheduling

Las GPUs son recursos especiales. K8s las maneja con `nvidia.com/gpu` en resource limits:

```yaml
resources:
  requests:
    cpu: 4
    memory: 16Gi
    nvidia.com/gpu: 1
  limits:
    cpu: 8
    memory: 32Gi
    nvidia.com/gpu: 1
```

**Reglas críticas:**
- `requests` debe ser igual a `limits` para GPUs (no hay overcommit)
- No podés compartir una GPU entre containers (a menos que uses MIG o Time-Slicing)
- Los GPU nodes deben tener el driver NVIDIA y `nvidia-container-toolkit`

Aislá los GPU nodes con taints y tolerations para que solo Pods que necesitan GPU caigan ahí:

```bash
$ kubectl taint nodes gpu-node-1 nvidia.com/gpu=present:NoSchedule
```

```yaml
tolerations:
- key: nvidia.com/gpu
  operator: Exists
  effect: NoSchedule
nodeSelector:
  nvidia.com/gpu: present
```

Sin taints, un Pod de NGINX puede caer en un nodo con GPU y desperdiciar el recurso. Esto se complementa con [[Kubernetes Fundamentals]] para entender cómo el Scheduler asigna Pods a nodos.

## 3. Volúmenes: Efímeros vs Persistentes

Los datasets y checkpoints no pueden vivir en containers efímeros:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: data-pvc
spec:
  accessModes:
  - ReadWriteMany
  resources:
    requests:
      storage: 500Gi
  storageClassName: ssd
```

```yaml
volumes:
- name: data
  persistentVolumeClaim:
    claimName: data-pvc
containers:
- name: trainer
  volumeMounts:
  - mountPath: /data
    name: data
```

- **ReadWriteOnce**: un solo nodo lee/escribe (default)
- **ReadWriteMany**: múltiples nodos leen/escriben (ideal para datasets compartidos, [[Experiment Tracking]])
- **StorageClass**: `ssd` para GPUs, `hdd` para archivos fríos

Los **emptyDir** son útiles para datos temporales inter-paso en [[Hyperparameter Tuning]], pero no sobreviven a fallos del nodo.

## 4. Kubeflow

Kubeflow es la plataforma de ML sobre K8s. Componentes clave:

**Pipelines:** definís DAGs de componentes como si fueran Jobs encadenados:

```yaml
# Pseudocódigo de un pipeline Kubeflow
@dsl.pipeline(name="Entrenamiento")
def pipeline(dataset: str, epochs: int):
    preprocess_op = preprocess(dataset)
    train_op = train(preprocess_op.output, epochs)
    evaluate_op = evaluate(train_op.output)
```

**Notebooks:** servidores Jupyter efímeros con GPUs asignadas. Cada data scientist tiene su notebook server con sus propios recursos.

**Katib:** hyperparameter tuning automatizado:

```bash
$ kubectl apply -f katib-experiment.yaml
$ kubectl get experiments -n kubeflow
```

Katib prueba combinaciones de hiperparámetros (grid search, Bayesian optimization, NAS) lanzando Jobs paralelos. Se integra con [[Experiment Tracking]] para registrar resultados.

## 5. Seldon Core / KServe

El serving de modelos tiene sus propios requirements: autoscaling a 0, canary deployments, explicabilidad:

```yaml
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: modelo-prod
spec:
  predictors:
  - name: default
    replicas: 2
    componentSpecs:
    - spec:
        containers:
        - name: classifier
          image: modelo:v1
          resources:
            requests:
              nvidia.com/gpu: 0  # CPU serving
    graph:
      children: []
      endpoint:
        type: REST
      name: classifier
      type: MODEL
```

KServe (antes KFServing) agrega serverless inference, autoscaling con Knative, y canary routing. El tráfico se divide entre versiones v1 (90%) y v2 (10%) para validar el nuevo modelo.

Ambos se integran con [[Model Monitoring]] para detectar data drift y concept drift en producción.

## 6. Common Mistakes

- **CPU requests = limits con GPUs no es suficiente**: también necesitás `nvidia.com/gpu` bien configurado
- **No poner taints en GPU nodes**: un Pod cualquiera puede ocupar el nodo y bloquear trabajos ML
- **Pedir GPUs sin metrics**: si no monitoreás utilización de GPU (con `dcgm-exporter`), no sabés si se están aprovechando
- **Jobs sin backoffLimit**: un Job falla infinitamente si hay un bug; siempre poné un límite
- **Dangling PVCs**: al eliminar un Job, el PVC queda ocupando storage; usá `ttlSecondsAfterFinished` y políticas de cleanup
- **Docker images enormes**: imágenes ML con CUDA, PyTorch, etc. pesan >5 GB; optimizá con multi-stage y capas bases slim

## Resumen

1. K8s Jobs manejan run-to-completion con parallelism, backoff y TTL; CronJobs para tareas periódicas
2. GPU scheduling requiere `nvidia.com/gpu`, taints/tolerations, y nodeSelector para aislar nodos GPU
3. Volúmenes persistentes (PVC, StorageClass) guardan datasets y checkpoints; ReadWriteMany para datos compartidos
4. Kubeflow aporta pipelines, notebooks y Katib para hyperparameter tuning sobre K8s
5. Seldon Core y KServe sirven modelos con autoscaling, canary releases y serverless
6. Siempre monitoreá GPUs, poné taints en nodos GPU, y evitá dangling PVCs

## Check Your Understanding

1. ¿Cuál es la diferencia entre un Deployment y un Job en K8s? <!-- Un Deployment corre containers "para siempre"; un Job ejecuta una tarea run-to-completion. -->
2. ¿Por qué necesitás taints en los nodos con GPU? <!-- Para evitar que Pods que no necesitan GPU ocupen esos nodos y desperdicien el recurso. -->
3. ¿Qué ventaja tiene ReadWriteMany sobre ReadWriteOnce para datasets de ML? <!-- Permite que múltiples Pods en distintos nodos lean el mismo dataset simultáneamente. -->
4. ¿Qué hace Katib dentro de Kubeflow? <!-- Hyperparameter tuning automatizado: prueba combinaciones de hiperparámetros lanzando Jobs paralelos. -->
5. ¿Qué problema resuelve KServe con canary deployments? <!-- Validar un nuevo modelo en producción redirigiendo un % pequeño del tráfico antes del rollout completo. -->

## Where to Go Next

- [[Kubernetes Fundamentals]]
- [[Docker Fundamentals]]
- [[Helm & Package Management]]
- [[Training Techniques]]
- [[Model Monitoring]]
- [[Experiment Tracking]]
- [[ML Pipelines]]
- [[Hyperparameter Tuning]]
