ARG L4T_TAG=r36.4.0

FROM nvcr.io/nvidia/l4t-jetpack:${L4T_TAG}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    OCR_PROFILE=fast \
    OCR_ENGINE=paddle_static \
    OCR_DOC_ORI_MODEL_DIR=/app/models/PP-LCNet_x1_0_doc_ori_infer \
    OCR_DOC_UNWARP_MODEL_DIR=/app/models/UVDoc_infer \
    OCR_TEXTLINE_ORI_MODEL_DIR=/app/models/PP-LCNet_x0_25_textline_ori_infer \
    OCR_DET_MODEL_DIR=/app/models/PP-OCRv5_mobile_det \
    OCR_REC_MODEL_DIR=/app/models/PP-OCRv5_mobile_rec \
    OCR_TEXT_RECOGNITION_BATCH_SIZE=4 \
    OCR_DATA_DIR=/app/data

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY wheels/paddlepaddle_gpu-*.whl /tmp/paddle-wheel/
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install /tmp/paddle-wheel/*.whl \
    && python3 -m pip install -r /app/requirements.txt

COPY src /app/src
COPY README.md /app/README.md

RUN mkdir -p /app/data/uploads /app/data/results /app/models

EXPOSE 8000

CMD ["python3", "-m", "ocr_service.main"]
