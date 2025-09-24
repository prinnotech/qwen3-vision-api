# Use Python 3.11 slim image
FROM python:3.11-slim

# Install system dependencies for video/image processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libopencv-dev \
    python3-opencv \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the Qwen-VL model during build (saves startup time)  
RUN python -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('Qwen/Qwen-VL-Chat', trust_remote_code=True); AutoTokenizer.from_pretrained('Qwen/Qwen-VL-Chat', trust_remote_code=True)"

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]