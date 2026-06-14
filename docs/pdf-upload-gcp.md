# PDF Uploads on Cloud Run

This demo supports PDF context uploads by extracting selectable text and appending it
to the existing VerifiQuant context field. The original PDF is only persisted when
`VERIFIQUANT_GCS_BUCKET` is set.

## Runtime Defaults

| Environment variable | Default | Purpose |
|---|---:|---|
| `VERIFIQUANT_PDF_UPLOADS_ENABLED` | `true` | Enable `/api/pdf/extract` |
| `VERIFIQUANT_PDF_RATE_LIMIT` | `10` | PDF uploads per IP per hour |
| `VERIFIQUANT_MAX_PDF_BYTES` | `10485760` | 10 MiB upload cap |
| `VERIFIQUANT_MAX_PDF_PAGES` | `20` | Maximum pages extracted |
| `VERIFIQUANT_MAX_PDF_CHARS` | `24000` | Maximum extracted characters sent to context |
| `VERIFIQUANT_GCS_BUCKET` | unset | Optional bucket for short-lived original PDFs |

The Cloud Run container filesystem is not durable and writes use instance memory, so
do not rely on local files for retention. For temporary retention, use Cloud Storage
with a bucket lifecycle rule.

## Recommended GCP Setup

Create a private bucket:

```bash
export GCP_PROJECT_ID=your-project-id
export GCP_REGION=asia-east1
export PDF_BUCKET="${GCP_PROJECT_ID}-verifiquant-pdf-demo"

gcloud storage buckets create "gs://${PDF_BUCKET}" \
  --project "${GCP_PROJECT_ID}" \
  --location "${GCP_REGION}" \
  --uniform-bucket-level-access
```

Apply a lifecycle rule that deletes uploaded PDFs after one day:

```bash
gcloud storage buckets update "gs://${PDF_BUCKET}" \
  --lifecycle-file docs/pdf-upload-lifecycle.json
```

Grant the Cloud Run service identity permission to create objects in that bucket:

```bash
export SERVICE_ACCOUNT="verifiquant-run@${GCP_PROJECT_ID}.iam.gserviceaccount.com"

gcloud storage buckets add-iam-policy-binding "gs://${PDF_BUCKET}" \
  --member "serviceAccount:${SERVICE_ACCOUNT}" \
  --role "roles/storage.objectCreator"
```

Deploy with the bucket enabled:

```bash
export VERIFIQUANT_GCS_BUCKET="${PDF_BUCKET}"
export VERIFIQUANT_MAX_PDF_BYTES=10485760
export VERIFIQUANT_MAX_PDF_PAGES=20
export VERIFIQUANT_MAX_PDF_CHARS=24000
./deploy.sh
```

If the service must later read or explicitly delete uploaded PDFs itself, replace
`roles/storage.objectCreator` with a narrower custom role or `roles/storage.objectUser`
on only this bucket. For the current lifecycle-cleanup design, object creation is
enough for the running service; lifecycle setup is done by the deployer/admin.
