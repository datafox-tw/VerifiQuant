#!/usr/bin/env bash
# Deploy the harness demo to Cloud Run. Reuses the GEMINI_API_KEY secret of the project.
#   ./harness_demo/deploy.sh            (project from GCP_PROJECT_ID or the gcloud default)
set -euo pipefail
cd "$(dirname "$0")"
PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${GCP_REGION:-asia-east1}"
SERVICE_NAME="${SERVICE_NAME:-harness-demo}"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:$(date +%Y%m%d-%H%M%S)"

gcloud builds submit --project "$PROJECT_ID" --tag "$IMAGE" --timeout=1200 .
gcloud run deploy "$SERVICE_NAME" --project "$PROJECT_ID" --image "$IMAGE" --region "$REGION" \
  --platform managed --allow-unauthenticated \
  --memory 1Gi --cpu 1 --timeout 300 --max-instances 1 --min-instances 0 --concurrency 20 \
  --set-env-vars "HARNESS_LIVE_PER_IP_HOUR=${HARNESS_LIVE_PER_IP_HOUR:-20},HARNESS_LIVE_PER_DAY=${HARNESS_LIVE_PER_DAY:-200}" \
  --update-secrets "GEMINI_API_KEY=GEMINI_API_KEY:latest"
gcloud run services describe "$SERVICE_NAME" --project "$PROJECT_ID" --region "$REGION" --format='value(status.url)'
