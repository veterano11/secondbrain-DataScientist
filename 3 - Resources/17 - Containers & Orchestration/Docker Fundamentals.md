---
tags:
  - container
  - docker
  - devops
status: seedling
created: 2026-06-28
---

# Docker Fundamentals

## Escenario de aprendizaje

Tu app de Python funciona perfecto en tu laptop. La subes a staging y todo explota — librerías con versiones distintas, falta una dependencia del sistema, rutas absolutas que no existen. "En mi máquina funciona" resume el problema. Docker empaqueta tu app con todo lo que necesita (sistema de archivos, librerías, variables de entorno) para que corra idéntico en cualquier lado: tu laptop, staging, producción, hasta la máquina de un compañero.

## 1. Docker vs Máquina Virtual

| Aspecto | Docker | VM |
|---------|--------|----|
| Kernel | Comparte el kernel del host | Cada VM tiene su propio kernel |
| Inicio | Segundos | Minutos |
| Tamaño | MB | GB |
| Aislamiento | A nivel de proceso | Hipervisor, completo |

Docker **NO** virtualiza hardware. Usa *cgroups* y *namespaces* del kernel Linux para aislar procesos. Por eso un container de [[CLI & Productivity]] corre casi tan rápido como un proceso nativo, mientras una VM necesita emular hardware completo.

## 2. Dockerfile

El `Dockerfile` es la receta de tu imagen. Cada instrucción crea una *layer*:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "main.py"]
```

- **FROM**: imagen base (elegí `slim` para reducir tamaño)
- **WORKDIR**: directorio de trabajo dentro del container
- **COPY**: copia archivos del host al container
- **RUN**: ejecuta comandos durante el build (instalar dependencias)
- **CMD**: comando por defecto al iniciar el container

Alternativamente `ENTRYPOINT` define el ejecutable y `CMD` sus argumentos por defecto. La combinación es útil para imágenes tipo "CLI tool" similares a [[Virtual Environments]].

## 3. Construir y etiquetar imágenes

```bash
$ docker build -t mi-app:0.1.0 .
$ docker tag mi-app:0.1.0 usuario/mi-app:latest
$ docker push usuario/mi-app:0.1.0
```

**Salida esperada:**
```
 => [1/4] FROM python:3.11-slim
 => [2/4] WORKDIR /app
 => [3/4] COPY requirements.txt .
 => [4/4] RUN pip install -r requirements.txt
 => exporting to image
 => => naming to docker.io/usuario/mi-app:0.1.0
```

**Tagging strategies:**
- `latest`: útil para dev, peligroso en producción (¿qué versión es?)
- `semántico` (`0.1.0`, `1.2.3`): trazable, reproducible
- `commit SHA`: ideal para [[CI-CD & GitOps]]

## 4. Correr containers

```bash
$ docker run -d --name mi-api -p 8000:8000 -v $(pwd):/app -e DB_URL=postgres://... mi-app
$ docker ps
$ docker logs -f mi-api
$ docker exec -it mi-api bash
$ docker run --rm alpine echo "hola mundo"
```

**Explicación de flags:**
- `-p 8000:8000`: mapea puerto host : puerto container
- `-v $(pwd):/app`: bind mount del directorio actual
- `-e VAR=val`: variable de entorno
- `--rm`: elimina container al detenerse (ideal para pruebas)
- `-d`: detached (background)

Usá `docker logs` para debugging en caliente y `docker exec` para inspeccionar un container corriendo, muy parecido a hacer SSH en [[Kubernetes Fundamentals]].

## 5. Layers y caching

Docker cachea cada layer del Dockerfile. Si una capa no cambió, reusa la del cache. **El orden importa:**

```dockerfile
# ❌ Lento: copia todo el código ANTES de instalar dependencias
COPY . .
RUN pip install -r requirements.txt

# ✅ Óptimo: copia requirements PRIMERO (cambia poco), instala, luego copia el código
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
```

**Multi-stage build:** usá una imagen con herramientas de build y otra más chica para producción:

```dockerfile
FROM python:3.11 AS builder
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.11-slim
COPY --from=builder /root/.local /root/.local
COPY . .
CMD ["python", "main.py"]
```

## 6. Registries

```bash
$ docker login
$ docker pull python:3.11-slim
$ docker push usuario/mi-app:0.1.0
```

Además de [[Docker Hub]], existen registries privados como **Amazon ECR**, **Google Container Registry (GCR)**, **GitHub Container Registry** y **Azure Container Registry**. En producción, los [[CI-CD & GitOps]] pipelines pushean imágenes con el SHA del commit como tag.

## 7. Common Mistakes

- **Olvidar `.dockerignore`**: incluye `node_modules`, `__pycache__`, `.env` en la imagen
- **Imágenes enormes**: usá imágenes slim/alpine, multi-stage builds
- **Hardcodear secrets**: jamás pongas contraseñas en el Dockerfile; usá variables de entorno o [[Infrastructure as Code Fundamentals]] con secretos externos
- **Correr como root**: usá `USER appuser` en el Dockerfile
- **Tags mutables**: `latest` cambia; siempre usá tags específicos en producción

## Resumen

1. Docker empaqueta app + dependencias en una imagen portable, eliminando el problema de "en mi máquina funciona"
2. Containers comparten el kernel del host, son más livianos y rápidos que VMs
3. El Dockerfile define la imagen usando capas; el orden de las instrucciones afecta el caching
4. `docker build`, `tag`, `push` manejan el ciclo de vida de imágenes; `docker run`, `ps`, `logs`, `exec` manejan containers
5. Multi-stage builds reducen drásticamente el tamaño de la imagen final
6. Los registries (Docker Hub, ECR, GCR) almacenan y distribuyen imágenes

## Check Your Understanding

1. ¿Por qué un container Docker arranca en segundos mientras una VM tarda minutos? <!-- Porque Docker comparte el kernel del host, no necesita bootear un OS completo. -->
2. ¿Qué diferencia hay entre `CMD` y `ENTRYPOINT` en un Dockerfile? <!-- ENTRYPOINT define el ejecutable, CMD sus argumentos por defecto. Juntos permiten imágenes que actúan como comandos. -->
3. ¿Por qué conviene copiar `requirements.txt` antes que el resto del código? <!-- Para aprovechar el caching: las dependencias cambian poco, el código cambia todo el tiempo. -->
4. ¿Qué problema tiene usar `:latest` en producción? <!-- No es reproducible — no sabés qué versión estás corriendo realmente. -->
5. ¿Para qué sirve `.dockerignore`? <!-- Para excluir archivos innecesarios (carpetas de dependencias, secretos, builds locales) de la imagen. -->

## Where to Go Next

- [[Docker Compose & Multi-Service]]
- [[Kubernetes Fundamentals]]
- [[CI-CD & GitOps]]
- [[Infrastructure as Code Fundamentals]]
- [[Virtual Environments]]
