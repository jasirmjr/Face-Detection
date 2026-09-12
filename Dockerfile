FROM python:3.11-slim

# Install system libraries needed by OpenCV and InsightFace
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download buffalo_s model during build so container boots in <1s without download memory spike
RUN python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_s', allowed_modules=['detection', 'recognition'])"

# Copy application source code
COPY . .

# Expose ports (10000 for Render, 7860 for Hugging Face Spaces, 8000 for local/Docker)
EXPOSE 10000 7860 8000

# Start server with dynamic cloud PORT support (defaults to 10000 for Render)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]

