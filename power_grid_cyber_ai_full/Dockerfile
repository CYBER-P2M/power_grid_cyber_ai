# Dockerfile for Multimodal Cyber-Physical Power Grid Security
# 
# Build:
#   docker build -t power-grid-cyber-ai .
#
# Run:
#   docker run -p 8501:8501 -p 8502:8502 power-grid-cyber-ai

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional vision dependencies
RUN pip install --no-cache-dir \
    ultralytics \
    opencv-python-headless \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Expose ports
# 8501: Main dashboard
# 8502: Multimodal dashboard (if running separately)
EXPOSE 8501 8502

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8501')" || exit 1

# Default command: run main dashboard
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
