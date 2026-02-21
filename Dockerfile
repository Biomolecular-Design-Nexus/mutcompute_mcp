# =============================================================================
# MutCompute MCP Docker Image
# =============================================================================
# Multi-stage build:
#   Stage 1 (legacy): Python 3.6 environment with Theano, Cython box_builder,
#                     pdb2pqr, freesasa, and MutCompute neural network weights
#   Stage 2 (mcp):    Python 3.10 MCP server that calls into the legacy env
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Legacy Python 3.6 environment for MutCompute neural network
# ---------------------------------------------------------------------------
FROM continuumio/miniconda3:4.12.0 AS legacy

# System dependencies for building Cython extensions, freesasa, and pdb2pqr
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    python2 \
    autoconf \
    automake \
    libtool \
    libc++-dev \
    libjson-c-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create the legacy conda environment with Python 3.6
RUN conda create -p /opt/env_py36 python=3.6.12 -y && conda clean -afy

# Clone the MutCompute repository (includes model weights ~144MB)
ARG MUTCOMPUTE_REPO=https://github.com/charlesxu90/MutCompute.git
ARG MUTCOMPUTE_BRANCH=main
RUN git clone --depth 1 -b ${MUTCOMPUTE_BRANCH} ${MUTCOMPUTE_REPO} /opt/mutcompute

# Install Python dependencies into the legacy environment
RUN /opt/env_py36/bin/pip install --no-cache-dir -r /opt/mutcompute/requirements.txt

# Install additional conda packages needed by MutCompute
RUN conda install -p /opt/env_py36 -y \
    numpy=1.19.2 \
    scipy=1.5.2 \
    theano=1.0.4 \
    pygpu=0.7.6 \
    libgpuarray=0.7.6 \
    biopython=1.70 \
    pandas=1.1.5 \
    cython=0.29.21 \
    && conda clean -afy

# Build Cython box_builder extension
WORKDIR /opt/mutcompute/src/box_builder
RUN /opt/env_py36/bin/python setup.py build_ext --inplace

# Build and install freesasa from source
WORKDIR /tmp
RUN git clone https://github.com/mittinatten/freesasa.git && \
    cd freesasa && \
    git checkout 2.0.3 && \
    autoreconf -i && \
    ./configure --disable-json --disable-xml --prefix=/usr/local && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd / && rm -rf /tmp/freesasa

# ---------------------------------------------------------------------------
# Stage 2: MCP server with Python 3.10
# ---------------------------------------------------------------------------
FROM continuumio/miniconda3:latest

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python2 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the legacy Python 3.6 environment from stage 1
COPY --from=legacy /opt/env_py36 /opt/env_py36

# Copy the MutCompute repo (with compiled Cython extensions and model weights)
COPY --from=legacy /opt/mutcompute /opt/mutcompute

# Copy freesasa binary and libraries from stage 1
COPY --from=legacy /usr/local/bin/freesasa /usr/local/bin/freesasa
COPY --from=legacy /usr/local/lib/libfreesasa* /usr/local/lib/
RUN ldconfig

# Create the main MCP environment with Python 3.10
RUN conda create -p /opt/env python=3.10 pip -y && conda clean -afy

WORKDIR /app

# Install MCP server dependencies
RUN /opt/env/bin/pip install --no-cache-dir \
    fastmcp \
    loguru \
    pandas \
    numpy \
    scipy \
    tqdm \
    matplotlib \
    seaborn

# Force-reinstall fastmcp to ensure clean state
RUN /opt/env/bin/pip install --no-cache-dir --ignore-installed fastmcp

# Copy MCP server source code
COPY src/ ./src/
RUN chmod -R a+r /app/src/

# Create working directories
RUN mkdir -p tmp/inputs tmp/outputs logs

# Set up symlinks so the MCP server can find the legacy environment and repo
# The MCP tools reference PROJECT_DIR / "env_py36" and PROJECT_DIR / "repo" / "mutcompute"
RUN ln -s /opt/env_py36 /app/env_py36 && \
    mkdir -p /app/repo && \
    ln -s /opt/mutcompute /app/repo/mutcompute

# Ensure freesasa and python2 are on PATH for the legacy environment
ENV PATH="/usr/local/bin:${PATH}"
ENV PYTHONPATH=/app
# Default to CPU mode for Theano (override with THEANO_FLAGS for GPU)
ENV THEANO_FLAGS="device=cpu,floatX=float32"

CMD ["/opt/env/bin/python", "src/server.py"]
