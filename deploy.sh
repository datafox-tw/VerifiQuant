#!/usr/bin/env bash
set -euo pipefail

# ── VerifiQuant Cloud Run Deploy ──
# Prerequisites:
#   1. gcloud CLI installed & authenticated: gcloud auth login
#   2. A GCP project with billing enabled
#   3. GEMINI_API_KEY ready

PROJECT_ID="${GCP_PROJECT_ID:-}"
REGION="${GCP_REGION:-asia-east1}"
SERVICE_NAME="verifiquant"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

if [ -z "$PROJECT_ID" ]; then
  echo "Error: Set GCP_PROJECT_ID environment variable first."
  echo "  export GCP_PROJECT_ID=your-project-id"
  exit 1
fi

echo "==> Setting project to ${PROJECT_ID}"
gcloud config set project "$PROJECT_ID"

echo "==> Enabling required APIs..."
gcloud services enable \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  containerregistry.googleapis.com \
  artifactregistry.googleapis.com

echo "==> Building container image..."
gcloud builds submit --tag "$IMAGE_NAME" --timeout=1200

echo "==> Deploying to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
  --image "$IMAGE_NAME" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 120 \
  --max-instances 3 \
  --min-instances 0 \
  --set-env-vars "VERIFIQUANT_DEMO_MODE=true,VERIFIQUANT_RATE_LIMIT=20,PORT=8080" \
  --update-secrets "GEMINI_API_KEY=GEMINI_API_KEY:latest"

echo ""
echo "==> Deployed! Getting service URL..."
URL=$(gcloud run services describe "$SERVICE_NAME" --region "$REGION" --format='value(status.url)')
echo "Service URL: $URL"
echo ""
echo "==> Next steps:"
echo "  1. Create the secret (if not done):"
echo "     echo -n 'your-api-key' | gcloud secrets create GEMINI_API_KEY --data-file=-"
echo "  2. Map custom domain:"
echo "     gcloud run domain-mappings create --service=$SERVICE_NAME --domain=verifiquant.com --region=$REGION"
echo "  3. Set DNS records as shown by the command above (CNAME or A records)"
