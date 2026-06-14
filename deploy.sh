#!/usr/bin/env bash
set -euo pipefail

# ── VerifiQuant Daily Cloud Run Deploy ──
# Usage:
#   export GCP_PROJECT_ID=your-project-id
#   ./deploy.sh
#
# Optional:
#   GCP_REGION=asia-east1 SERVICE_NAME=verifiquant ./deploy.sh
#   ./deploy.sh --source   # use Cloud Run source deploy instead of explicit Docker build

PROJECT_ID="${GCP_PROJECT_ID:-}"
REGION="${GCP_REGION:-asia-east1}"
SERVICE_NAME="${SERVICE_NAME:-verifiquant}"

MEMORY="${CLOUD_RUN_MEMORY:-2Gi}"
CPU="${CLOUD_RUN_CPU:-2}"
TIMEOUT="${CLOUD_RUN_TIMEOUT:-120}"
MAX_INSTANCES="${CLOUD_RUN_MAX_INSTANCES:-3}"
MIN_INSTANCES="${CLOUD_RUN_MIN_INSTANCES:-0}"

DEMO_MODE="${VERIFIQUANT_DEMO_MODE:-true}"
RATE_LIMIT="${VERIFIQUANT_RATE_LIMIT:-20}"

USE_SOURCE_DEPLOY=false

if [[ "${1:-}" == "--source" ]]; then
  USE_SOURCE_DEPLOY=true
fi

if [[ -z "$PROJECT_ID" ]]; then
  echo "Error: GCP_PROJECT_ID is not set."
  echo "Run:"
  echo "  export GCP_PROJECT_ID=your-project-id"
  exit 1
fi

echo "==> Project: ${PROJECT_ID}"
echo "==> Region: ${REGION}"
echo "==> Service: ${SERVICE_NAME}"

gcloud config set project "$PROJECT_ID" >/dev/null

if [[ "$USE_SOURCE_DEPLOY" == true ]]; then
  echo "==> Deploying from source..."
  gcloud run deploy "$SERVICE_NAME" \
    --source . \
    --region "$REGION" \
    --platform managed \
    --allow-unauthenticated \
    --memory "$MEMORY" \
    --cpu "$CPU" \
    --timeout "$TIMEOUT" \
    --max-instances "$MAX_INSTANCES" \
    --min-instances "$MIN_INSTANCES" \
    --set-env-vars "VERIFIQUANT_DEMO_MODE=${DEMO_MODE},VERIFIQUANT_RATE_LIMIT=${RATE_LIMIT}" \
    --update-secrets "GEMINI_API_KEY=GEMINI_API_KEY:latest"
else
  IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:$(date +%Y%m%d-%H%M%S)"

  echo "==> Building image: ${IMAGE_NAME}"
  gcloud builds submit \
    --tag "$IMAGE_NAME" \
    --timeout=1200

  echo "==> Deploying image to Cloud Run..."
  gcloud run deploy "$SERVICE_NAME" \
    --image "$IMAGE_NAME" \
    --region "$REGION" \
    --platform managed \
    --allow-unauthenticated \
    --memory "$MEMORY" \
    --cpu "$CPU" \
    --timeout "$TIMEOUT" \
    --max-instances "$MAX_INSTANCES" \
    --min-instances "$MIN_INSTANCES" \
    --set-env-vars "VERIFIQUANT_DEMO_MODE=${DEMO_MODE},VERIFIQUANT_RATE_LIMIT=${RATE_LIMIT}" \
    --update-secrets "GEMINI_API_KEY=GEMINI_API_KEY:latest"
fi

echo ""
echo "==> Deployment finished."

URL="$(gcloud run services describe "$SERVICE_NAME" \
  --region "$REGION" \
  --format='value(status.url)')"

echo "Service URL: ${URL}"