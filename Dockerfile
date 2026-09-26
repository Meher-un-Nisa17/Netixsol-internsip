FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/uzma

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates passwd curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system uzma \
    && useradd --system --create-home --gid uzma --home-dir /home/uzma uzma

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Chroma's default embedding function downloads this 79 MB model on first use.
# Bake it into the image with a resumable transfer so the first customer request
# does not block while downloading, and so a slow connection can retry safely.
RUN set -eu; \
    model_cache_dir="/home/uzma/.cache/chroma/onnx_models/all-MiniLM-L6-v2"; \
    mkdir -p "$model_cache_dir"; \
    curl --fail --location --show-error --retry 20 --retry-all-errors --retry-delay 5 \
      --connect-timeout 30 --speed-limit 1024 --speed-time 120 --continue-at - \
      "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz" \
      --output "$model_cache_dir/onnx.tar.gz"; \
    echo "913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3  $model_cache_dir/onnx.tar.gz" | sha256sum --check --status; \
    tar -xzf "$model_cache_dir/onnx.tar.gz" -C "$model_cache_dir"; \
    chown -R uzma:uzma /home/uzma/.cache

COPY --chown=uzma:uzma . .
RUN mkdir -p /app/state/home /app/data/audio_output && chown -R uzma:uzma /app/state /app/data/audio_output

USER uzma
EXPOSE 8000 8501

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
