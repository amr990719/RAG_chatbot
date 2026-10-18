# Deploying the FinSolve RAG Chatbot to Azure Container Apps

A command-by-command runbook. Commands are written for **PowerShell** (your shell).
Nothing here needs Docker installed locally — the image is built in the cloud by Azure
Container Registry.

## What gets deployed
One self-contained container image (built by `Dockerfile`) that includes the code, all Python
deps, the embedding-model weights, and a pre-built `chroma_db/` (ingestion runs during the image
build). At runtime the only external input is `GROQ_API_KEY`, injected as an Azure secret. Azure
Container Apps gives it a free HTTPS URL and calls out to Groq Cloud for generation.

---

## 0. Prerequisites (one time)

1. An Azure subscription (a free/pay-as-you-go account works).
2. **Azure CLI** installed: https://learn.microsoft.com/cli/azure/install-azure-cli-windows
   Verify: `az version`
3. Sign in and register the required resource providers:

```powershell
az login
az extension add --name containerapp --upgrade
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights
az provider register --namespace Microsoft.ContainerRegistry
```

---

## 1. Set variables

ACR names are **globally unique** and alphanumeric only, so we append a random suffix.

```powershell
$RG       = "finsolve-rg"
$LOCATION = "eastus"
$ACR      = "finsolveacr" + (Get-Random -Maximum 99999)
$ENVNAME  = "finsolve-env"
$APP      = "finsolve-chatbot"
$IMAGE    = "finsolve-chatbot:v1"

# Your Groq key (paste it, or read it from the local .env):
$GROQ_KEY = (Get-Content .env | Where-Object { $_ -match '^GROQ_API_KEY=' }) -replace '^GROQ_API_KEY=', ''
```

---

## 2. Create a resource group

A resource group is just a container that holds all the resources for this project (so you can
delete everything at once later).

```powershell
az group create --name $RG --location $LOCATION
```

---

## 3. Create the container registry and build the image in the cloud

`az acr build` uploads the current folder (respecting `.dockerignore`), runs the `Dockerfile` on an
Azure build agent, and stores the resulting image in your registry. This is where `ingest.py` runs
and the vector store gets baked in. Expect this to take a few minutes (torch + model download).

```powershell
az acr create --resource-group $RG --name $ACR --sku Basic --admin-enabled true
az acr build --registry $ACR --image $IMAGE .
```

> If the build fails while compiling a dependency, it's almost always a missing C toolchain in the
> slim base image. Fix: add `RUN apt-get update && apt-get install -y build-essential` near the top
> of the `Dockerfile`, then re-run `az acr build`. (Usually not needed — most deps ship wheels.)

---

## 4. Create the Container Apps environment

The "environment" is the shared boundary (networking + Log Analytics) that one or more container
apps run inside. Created once.

```powershell
az containerapp env create --name $ENVNAME --resource-group $RG --location $LOCATION
```

---

## 5. Create the container app (with the secret + public ingress)

This is the main step. Notes on the flags:
- `--target-port 8000` matches the port uvicorn listens on in the `Dockerfile`.
- `--ingress external` gives it a public HTTPS URL.
- `--secrets` + `--env-vars ...=secretref:...` injects `GROQ_API_KEY` from a secret (never baked
  into the image, never in plain env output).
- `--min-replicas 1 --max-replicas 1` pins it to a **single instance**. This is deliberate: sessions
  are stored in an in-memory dict (see `app.py`), so multiple replicas would each have their own
  session table and logins would break. (See "Hardening" below to lift this.)
- `--cpu 2.0 --memory 4.0Gi` gives torch + the model comfortable headroom.

```powershell
az containerapp create `
  --name $APP `
  --resource-group $RG `
  --environment $ENVNAME `
  --image "$ACR.azurecr.io/$IMAGE" `
  --registry-server "$ACR.azurecr.io" `
  --target-port 8000 `
  --ingress external `
  --min-replicas 1 --max-replicas 1 `
  --cpu 2.0 --memory 4.0Gi `
  --secrets "groq-key=$GROQ_KEY" `
  --env-vars "GROQ_API_KEY=secretref:groq-key"
```

---

## 6. Get the public URL

```powershell
az containerapp show --name $APP --resource-group $RG --query properties.configuration.ingress.fqdn -o tsv
```

Open `https://<that-fqdn>` in a browser — the login screen should appear. Give the first request a
few seconds (the model loads into memory on the first call).

---

## 7. Watch logs / troubleshoot

```powershell
az containerapp logs show --name $APP --resource-group $RG --follow
```

Common issues:
- **503 "GROQ_API_KEY is not set"** → the secret/env wiring in step 5 didn't take; re-check the
  `--secrets` / `--env-vars` values.
- **Out of memory / restarts** → bump `--memory` (must be 2× cpu, e.g. `4.0Gi` with `2.0` cpu).

---

## 8. Updating after code or data changes

Rebuild a new image tag and point the app at it:

```powershell
$IMAGE = "finsolve-chatbot:v2"
az acr build --registry $ACR --image $IMAGE .
az containerapp update --name $APP --resource-group $RG --image "$ACR.azurecr.io/$IMAGE"
```

(Because the vector store is baked into the image, updating `data/` means rebuilding the image —
that's the trade-off of baking it in. See "Hardening" for externalizing it.)

---

## 9. Tear down (stop all costs)

```powershell
az group delete --name $RG --yes --no-wait
```

Deletes the app, environment, and registry in one go.

---

## Cost note
With `--min-replicas 1` the app is always on, so it isn't free, but it's small (a single 2 vCPU /
4 GiB container on the Container Apps consumption plan + a Basic ACR). Setting `--min-replicas 0`
enables scale-to-zero (near-zero idle cost) but adds cold-start latency and, with the current
in-memory sessions, drops everyone's login whenever it scales down — so it's paired with the
hardening step below.

---

## Hardening / productionization (beyond this learning build)
1. **Sessions** — replace the in-memory `_sessions` dict in `app.py` with signed cookies / JWT or
   Azure Cache for Redis. Then you can safely raise `--max-replicas` and enable scale-to-zero.
2. **Real authentication** — swap `DEMO_USERS` for Microsoft Entra ID (SSO). Map Entra **group
   claims → roles**, feeding the same `ROLE_DEPARTMENTS` matrix in `rbac.py`. This is the piece that
   turns the demo login into real RBAC.
3. **Secrets** — reference Azure Key Vault instead of an inline secret, and rotate the Groq key
   (especially since it was pasted in chat during development).
4. **ACR auth** — use a managed identity for image pulls instead of `--admin-enabled true`.
5. **Vector store** — for frequently-changing data, move `chroma_db/` out of the image onto an
   Azure Files volume mount (or a managed vector DB) so you can re-ingest without rebuilding.
6. **Observability** — the environment already ships logs to Log Analytics; add Application Insights
   for request tracing if needed.
7. **Custom domain + TLS** — Container Apps supports custom domains with managed certificates.
