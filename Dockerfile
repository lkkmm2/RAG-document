FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/data/models/hf \
    SENTENCE_TRANSFORMERS_HOME=/app/data/models/st

WORKDIR /app

# libgomp1 是 torch / onnxruntime 的运行时依赖；build-essential 给个别包编译用
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
# CPU 版 torch：不加 --extra-index-url 会拉到 2GB+ 的 CUDA 轮子
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu .

COPY configs ./configs
COPY apps ./apps

RUN useradd -m -u 1000 docqa && mkdir -p /app/data && chown -R docqa:docqa /app
USER docqa

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/v1/health || exit 1

CMD ["uvicorn", "docqa.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
