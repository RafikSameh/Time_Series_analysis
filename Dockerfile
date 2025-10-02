# Use NVIDIA CUDA runtime as base so system CUDA libraries are available
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Noninteractive + utf-8
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8

# Hint to NVIDIA runtime (documentary; honored only if container is run with --gpus)
# This does not "enable" GPU by itself — the host + docker run flags must allow GPUs.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# Install Python and basic build tools
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

# Set working dir
WORKDIR /app

# Copy only requirements first to take advantage of Docker cache
COPY requirements.txt .

# Upgrade pip and install Python deps
RUN python3 -m pip install --upgrade pip
RUN pip install  -r requirements.txt

# Copy rest of your project (app.py, modules, etc.)
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Default command to start the Streamlit app. The app will be reachable on 0.0.0.0:8501
#CMD ["streamlit", "run", "webpage.py", "--server.address=0.0.0.0"]