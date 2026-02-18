# Use official Ultralytics image (has CUDA, PyTorch, OpenCV)
FROM ultralytics/ultralytics:latest

# Install missing system library for OpenCV (fixes libGL.so.1 error)
RUN apt-get update && apt-get install -y \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /workspace

# Copy your code into container
COPY train.py .
COPY kitti.yaml .

# (Optional) If you have extra Python dependencies:
COPY requirements.txt .
RUN pip install -r requirements.txt

# Default command when container starts
CMD ["python", "train.py"]








