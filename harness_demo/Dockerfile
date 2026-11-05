# Cloud Run image for the harness demo. Build context is this folder only.
FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir flask google-genai gunicorn

COPY . harness_demo/

ENV PORT=8080
ENV PYTHONUNBUFFERED=1
EXPOSE 8080

# one worker so the in-memory live-mode quota and caches are shared by every request
CMD exec gunicorn "harness_demo.app:app" --bind "0.0.0.0:${PORT}" --workers 1 --threads 8 --timeout 300
