FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    BAHITH_PUBLIC_DEMO=1 \
    BAHITH_PRELOAD_MODEL=1 \
    PORT=7860 \
    HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface

RUN useradd --create-home --uid 1000 user
WORKDIR /app
COPY requirements.txt /app/requirements.txt
COPY LICENSE /app/LICENSE
# Install CPU PyTorch explicitly; a demo container does not need CUDA packages.
RUN python -m pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip check

COPY --chown=user:user app.py search.py documents.py public_demo.py corpus.json /app/
COPY --chown=user:user templates/ /app/templates/
COPY --chown=user:user static/ /app/static/
COPY --chown=user:user deploy/start.sh /app/deploy/start.sh
RUN chmod 755 /app/deploy/start.sh
USER user
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.getenv('PORT','7860')+'/ready', timeout=3)"
CMD ["/app/deploy/start.sh"]
