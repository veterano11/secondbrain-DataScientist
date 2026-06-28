---
tags:
  - docker
  - compose
  - container
  - devops
status: seedling
created: 2026-06-28
---

# Docker Compose & Multi-Service

## Escenario de aprendizaje

Tu stack tiene una API en FastAPI, Postgres como base de datos, Redis para caché y un worker de Celery procesando tareas asincrónicas. Con `docker run` necesitarías cuatro comandos, una red custom, volúmenes — y olvidate de mantener sincronizados los starts. Docker Compose define toda la arquitectura en un YAML y levanta todo con `docker compose up`.

## 1. docker-compose.yml

```yaml
services:
  api:
    build: ./api
    ports:
      - "8000:8000"
    environment:
      - DB_URL=postgresql://user:pass@db:5432/mydb
    depends_on:
      db:
        condition: service_healthy

  db:
    image: postgres:16-alpine
    volumes:
      - pgdata:/var/lib/postgresql/data
    environment:
      POSTGRES_PASSWORD: pass
    healthcheck:
      test: ["CMD-SHELL", "pg_isready"]
      interval: 5s

  redis:
    image: redis:7-alpine

  worker:
    build: ./worker
    environment:
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis

volumes:
  pgdata:
```

Cada `service` es un container. Compose crea automáticamente una red donde los servicios se descubren por su nombre: el `api` se conecta a `db` simplemente usando el hostname `db`, sin IPs hardcodeadas. Esto se complementa con [[Docker Fundamentals]] para construir las imágenes individuales.

## 2. Services

- **build**: ruta al Dockerfile (o `context` + `dockerfile` para personalizar)
- **ports**: mapeo `"host:container"`, igual que `docker run -p`
- **depends_on**: orden de inicio — **no** espera a que el servicio esté listo
- **healthcheck**: comando que prueba si el servicio responde; necesario para `condition: service_healthy`

```yaml
depends_on:
  db:
    condition: service_healthy
```

Sin el `condition`, Compose arranca `db` y pasa inmediatamente a `api`, que muere porque Postgres todavía no acepta conexiones. Es el error clásico que mencionamos en [[CLI & Productivity]].

## 3. Networks

Compose crea una red por defecto. Podés definir redes adicionales para aislar servicios:

```yaml
services:
  api:
    networks:
      - frontend
      - backend
  db:
    networks:
      - backend

networks:
  frontend:
  backend:
```

Cada servicio entra en las redes que necesite. El service discovery por nombre de servicio (resuelto por DNS interno) es transparente, igual que en [[Kubernetes Fundamentals]].

## 4. Volumes

| Tipo | Sintaxis | Uso |
|------|----------|-----|
| Named volume | `pgdata:/data` | Persistencia manejada por Docker |
| Bind mount | `./src:/app` | Desarrollo: refleja cambios locales |

```yaml
services:
  api:
    volumes:
      - ./api:/app          # bind mount: código editable
      - /app/__pycache__    # anonymous volume: evitar cache

volumes:
  pgdata:  # named volume declarado arriba
```

Para desarrollo, los bind mounts permiten editar código sin reconstruir la imagen. Es la misma filosofía de entornos reproducibles que encontramos en [[Python for Data Science]].

## 5. Profiles y overlays

**Profiles** activan servicios condicionalmente:

```yaml
services:
  admin-ui:
    image: admin-ui
    profiles: ["dev", "staging"]

  worker:
    image: worker
    # sin profile: siempre se levanta
```

```bash
$ docker compose --profile dev up
```

**Overlay files**: `docker-compose.override.yml` se aplica automáticamente sobre el base. Usalo para diferencias entre entornos (distintas variables, puertos, etc.). También podés usar `-f` para múltiples archivos:

```bash
$ docker compose -f compose.yml -f compose.prod.yml up
```

Esta técnica de composición se alinea con [[Infrastructure as Code Fundamentals]] donde los entornos se parametrizan.

## 6. Comandos

```bash
$ docker compose up -d               # levantar en background
$ docker compose up --build          # reconstruir imágenes antes de levantar
$ docker compose logs -f api         # seguir logs de un servicio
$ docker compose exec api bash       # entrar a un container
$ docker compose down                # detener y remover todo
$ docker compose down -v             # lo mismo + eliminar volúmenes
$ docker compose ps                  # ver estado de servicios
```

El flag `-v` en `down` borra los volúmenes named. ¡Cuidado! Perdés datos de la base de datos. Usalo solo si querés resetear todo, ideal en [[CI-CD & GitOps]] pipelines.

## 7. Common Mistakes

- **Sin healthchecks**: `depends_on` solo espera que el container arranque, no que el servicio esté listo
- **depends_on sin condition**: la app falla al conectar porque la DB todavía no acepta conexiones
- **Volúmenes sin nombre**: los anonymous volumes se pierden al hacer `down`
- **Hardcodear secrets en YAML**: usá un `.env` file o variables de entorno externas
- **Exponer puertos en producción**: servicios internos (DB, Redis) no necesitan `ports`
- **Olvidar `--build`**: Compose usa imágenes cacheadas aunque hayas cambiado el Dockerfile

## Resumen

1. Docker Compose define y corre aplicaciones multi-container con un archivo YAML
2. Los servicios se descubren por nombre de servicio gracias a la red interna de Docker
3. `depends_on` sin healthchecks da falsa sensación de seguridad: el servicio puede no estar listo
4. Bind mounts habilitan desarrollo en caliente; named volumes persisten datos entre reinicios
5. Profiles y override files permiten variar la configuración por entorno
6. `docker compose up -d` levanta todo; `down -v` destruye todo (incluyendo datos)

## Check Your Understanding

1. ¿Cómo descubre un servicio la dirección IP de otro servicio en Compose? <!-- Por DNS interno: el nombre del servicio resuelve a la IP del container. -->
2. ¿Qué diferencia hay entre `depends_on` simple y `depends_on` con healthcheck? <!-- depends_on simple solo espera que el container arranque; con healthcheck espera a que el servicio responda correctamente. -->
3. ¿Cuándo usarías un bind mount en vez de un named volume? <!-- En desarrollo, para que los cambios locales se reflejen sin reconstruir la imagen. -->
4. ¿Para qué sirven los profiles en Compose? <!-- Para activar/desactivar servicios según el entorno (dev, staging) sin tener múltiples archivos. -->
5. ¿Qué pasa si ejecutás `docker compose down -v`? <!-- Detiene los servicios, remueve los containers y elimina los volúmenes named — incluidos los datos de la DB. -->

## Where to Go Next

- [[Docker Fundamentals]]
- [[Kubernetes Fundamentals]]
- [[CLI & Productivity]]
- [[CI-CD & GitOps]]
- [[Python for Data Science]]
