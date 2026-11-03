# Kubernetes

`deploy/helm/ragkit` deploys a ragkit image with:

- the **API** as a Deployment with a Service, HorizontalPodAutoscaler,
  PodDisruptionBudget, liveness, readiness and startup probes, and an optional
  Ingress;
- **workers** as a Deployment scaled by [KEDA](https://keda.sh) on the number of
  queued and running jobs, down to zero;
- one **scheduler** for `sync_every`;
- non-root pods with a read-only root filesystem and all capabilities dropped.

!!! note
    The chart passes `helm lint` and strict schema validation for Kubernetes 1.31
    (including KEDA's resources). Test it on a staging cluster before production.

## Prerequisites

- An image built from `deploy/Dockerfile` and pushed to your registry.
- Postgres with the `vector` extension available (a managed service is fine).
- [KEDA](https://keda.sh/docs/latest/deploy/) installed, unless you set
  `worker.keda.enabled=false`.

## Secrets

Create a Secret with everything the project reads from the environment:

```bash
kubectl create secret generic ragkit-secrets \
  --from-literal=DATABASE_URL='postgresql://ragkit:...@db.internal:5432/ragkit' \
  --from-literal=KEDA_POSTGRES_CONNECTION='postgresql://ragkit:...@db.internal:5432/ragkit?sslmode=require' \
  --from-literal=GOOGLE_API_KEY='...' \
  --from-literal=RAGKIT_KEY_APP='...' \
  --from-literal=RAGKIT_KEY_OPS='...'
```

`KEDA_POSTGRES_CONNECTION` is the connection KEDA uses to count jobs. It is
separate from `DATABASE_URL` so you can add the `sslmode` your database needs and,
ideally, use a read-only user.

## Install

```bash
helm install app deploy/helm/ragkit \
  --set image.repository=ghcr.io/acme/ragkit-app \
  --set image.tag=1.4.0
```

Common values:

| Value | Default | Meaning |
|---|---|---|
| `existingSecret` | `ragkit-secrets` | Secret loaded into every container |
| `env` | `[]` | Extra environment variables, e.g. the trace exporter |
| `projectConfig` | `""` | A `ragkit.yaml` that replaces the one in the image |
| `api.replicas` / `api.autoscaling.*` | 2 / 2 to 10 at 70% CPU | API scaling |
| `api.workers` | 1 | Server processes per pod |
| `worker.concurrency` | 2 | Jobs per worker pod |
| `worker.keda.minReplicaCount` / `maxReplicaCount` | 0 / 10 | Worker scaling range |
| `worker.keda.queuedJobsPerReplica` | 2 | Jobs per worker before scaling out |
| `worker.terminationGracePeriodSeconds` | 330 | Time for running jobs to finish on shutdown |
| `scheduler.enabled` | true | Run the scheduler |
| `ingress.enabled` | false | Create an Ingress |

See `values.yaml` for every value.

## Streaming through an ingress

Server-sent events need proxy buffering off. With ingress-nginx:

```yaml
ingress:
  enabled: true
  className: nginx
  annotations:
    nginx.ingress.kubernetes.io/proxy-buffering: "off"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
  hosts:
    - host: ragkit.example.com
      paths: [{path: /, pathType: Prefix}]
```

## How worker scaling works

KEDA runs this query every `pollingInterval` seconds:

```sql
SELECT COUNT(*) FROM ragkit_jobs WHERE status IN ('queued', 'running')
```

and keeps about one worker pod per `queuedJobsPerReplica` jobs. Running jobs are
counted so KEDA does not scale a worker away in the middle of a job. With
`minReplicaCount: 0`, no worker pods run while the queue is empty; the scheduler
runs separately so scheduled syncs still start.

## Upgrades

Rolling updates keep the API available (`maxUnavailable: 0`). Workers receive
SIGTERM, stop claiming jobs and finish running ones within the grace period; a job
cut off anyway is retried by another worker when its lease expires.
