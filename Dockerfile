FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# install git
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# get repo
RUN git clone https://github.com/deveringham/moe-experiments.git
WORKDIR /workspace/moe-experiments

# install dependencies from pyproject.toml
RUN uv sync --locked

ENV VIRTUAL_ENV=/workspace/moe-experiments/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

CMD ["bash"]
