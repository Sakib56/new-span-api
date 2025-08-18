# Use Python slim image
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Install system dependencies for OpenCV & rembg
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run the FastAPI app
CMD ["main.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
