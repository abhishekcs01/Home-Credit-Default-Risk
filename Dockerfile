FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME_CREDIT_API_WORKERS=4 \
    HOME_CREDIT_INFERENCE_FOLD_WORKERS=auto \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Multi-worker API (each process loads the bundle). Tune HOME_CREDIT_API_WORKERS for RAM vs throughput.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${HOME_CREDIT_API_PORT:-8000} --workers ${HOME_CREDIT_API_WORKERS:-4}"]
