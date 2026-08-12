FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel
ENV DEBIAN_FRONTEND=non-interactive
WORKDIR /workspace
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/deveringham/moe-experiments.git /workspace
RUN pip install --no-cache-dir -r /moe-experiments/requirements.txt
CMD ["bash"]