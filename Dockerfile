FROM python:3.11-slim-bullseye

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY constraints.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir \
        -r requirements.txt \
        -c constraints.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu

COPY main.py .
COPY scripts/ ./scripts/
COPY static/ ./static/
COPY templates/ ./templates/

# ROBOFLOW_API_KEY will be injected by Fly.io as a secret (see fly.toml for setup)
# It will be available at runtime as an environment variable

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
