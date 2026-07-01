FROM python:3.11-slim

LABEL project="Abhishek-ml project" \
      maintainer="Abhishek"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# LightGBM (and CatBoost) link against GNU OpenMP at runtime; slim images omit it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "lightgbm>=4.5.0"

COPY . .
EXPOSE 8000
CMD ["python", "scripts/run_api.py"]
