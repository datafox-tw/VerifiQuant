FROM python:3.11-slim AS base

WORKDIR /app

# Web image uses requirements-web.txt (no torch/transformers/etc.), so no
# build toolchain is needed — all wheels are pure-Python or prebuilt.
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt gunicorn

COPY verifiquant/ verifiquant/
COPY templates/ templates/
COPY static/ static/
COPY app.py .

ENV PORT=8080
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["gunicorn", "app:create_app()", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "4", "--timeout", "120"]
