FROM python:3.12-slim

WORKDIR /app

# Runtime libraries for OpenCV / matplotlib
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-docker.txt

COPY . .

ENV USE_CUDA=false
ENV PYTHONUNBUFFERED=1
ENV ANNOTATIONS_DATA_DIR=/app/data/annotations

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
