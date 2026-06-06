ARG L4T_TAG=r36.4.0

FROM nvcr.io/nvidia/l4t-jetpack:${L4T_TAG}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    OCR_PROFILE=fast \
    OCR_ENGINE=paddle_static \
    OCR_USE_DOCUMENT_STRUCTURE=1 \
    OCR_TEXT_RECOGNITION_BATCH_SIZE=4 \
    OCR_DATA_DIR=/app/data

ARG DEPS_CACHE_BUSTER=0

RUN apt-get update \
    && echo "Dependency rebuild marker: ${DEPS_CACHE_BUSTER}" >/dev/null \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        libjpeg-turbo8 \
        libopenblas0-pthread \
        libsm6 \
        libxext6 \
        libxrender1 \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY wheels/paddlepaddle_gpu-*.whl /tmp/paddle-wheel/
COPY requirements/ocr.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install /tmp/paddle-wheel/*.whl \
    && python3 -m pip install -r /app/requirements.txt

COPY src /app/src
COPY README.md /app/README.md

RUN mkdir -p /app/data/uploads /app/data/results

EXPOSE 8000

CMD ["python3", "-m", "ocr_service.main"]
