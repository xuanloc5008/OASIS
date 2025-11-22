# Use official Python image
FROM python:3.10-slim

# Install system dependencies required for OpenCV
# FIXED: Replaced 'libgl1-mesa-glx' with 'libgl1' for newer Debian versions
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    fastapi \
    uvicorn \
    python-multipart \
    numpy \
    opencv-python-headless \
    pillow

# Copy necessary files
COPY app.py .
COPY best.pth .
COPY Models/ ./Models/

# Expose the port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]