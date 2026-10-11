FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ANONYMIZED_TELEMETRY=False

WORKDIR /app

# Dependencies first: this layer is rebuilt only when requirements.txt changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY alembic.ini ./
COPY alembic ./alembic
COPY app ./app

# Non-root runtime user; /data is where the volumes (uploads, Chroma) are mounted.
RUN useradd --system --uid 10001 --home-dir /app docuask \
    && mkdir -p /data/uploads /data/chroma \
    && chown -R docuask /data
USER docuask

ENV UPLOAD_DIR=/data/uploads \
    CHROMA_DIR=/data/chroma

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# Migrations run before the server starts, so `docker compose up` always serves the current schema.
# `exec` makes uvicorn PID 1's replacement so it receives SIGTERM on `docker stop`.
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]
