FROM python:3.11-slim

# OpenCV runtime deps + git for pip deps from git
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy backend files first for dependency caching
COPY backend/requirements.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.0.1+cpu torchvision==0.15.2+cpu -r requirements.txt

# Copy full backend
COPY backend/ .

EXPOSE 8000
EXPOSE 8001

# Run both services: main.py on 8000, model_service.py on 8001
CMD sh -c "echo 'Starting main:app on :8000' && uvicorn main:app --host 0.0.0.0 --port 8000 & echo 'Starting model_service:app on :8001' && uvicorn model_service:app --host 0.0.0.0 --port 8001 & wait"