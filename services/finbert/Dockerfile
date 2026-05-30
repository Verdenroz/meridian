# FinBERT sentiment sidecar (CPU). Build:
#   docker build -t meridian-finbert services/finbert
# Run (container port 80 → host 8088, matching SENTIMENT_ENDPOINT default):
#   docker run -d --name meridian-finbert -p 8088:80 \
#       -v "$PWD/data/finbert:/data" meridian-finbert
FROM python:3.11-slim

WORKDIR /app

# CPU-only torch from the dedicated wheel index, then the serving stack.
RUN pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir \
        "transformers==4.44.2" \
        "fastapi==0.115.0" \
        "uvicorn==0.30.6"

COPY app.py .

# Cache HF weights on the mounted volume so restarts don't re-download.
ENV HF_HOME=/data
EXPOSE 80

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
