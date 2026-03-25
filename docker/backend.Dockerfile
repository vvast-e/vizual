FROM python:3.11-slim

# OpenCV runtime deps + git for pip deps from git
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch
COPY backend/requirements.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.0.1+cpu torchvision==0.15.2+cpu -r requirements.txt

COPY backend/ .

EXPOSE 8000
EXPOSE 8001

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 & uvicorn model_service:app --host 0.0.0.0 --port 8001 & wait"]
