# Dockerfile for ISLES Segmentation System
FROM python:3.9-slim

# Install system dependencies for medical imaging (OpenCV, libGL, etc.)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY src/ src/
COPY templates/ templates/
COPY static/ static/
COPY app.py .

# Copy model files from parent directory
# NOTE: In Docker context, models should ideally be copied within the build context.
# Assume models/ is mounted or copied here.
COPY models/ models/

# Expose port
EXPOSE 5000

# Start the application with Gunicorn
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "--timeout", "300", "app:app"]