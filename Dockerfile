# =========================================================
# Stage 1: Python Builder (Multi Arch)
# =========================================================
FROM registry.access.redhat.com/ubi9/ubi AS python-builder

ARG TARGETOS=linux
ARG TARGETARCH

WORKDIR /workspace

# Install system deps
RUN dnf install -y \
    python3.12 python3.12-devel python3.12-pip \
    gcc gcc-c++ gfortran make cmake git  \
    openblas openblas-devel numactl \
    zlib-devel libjpeg-devel \
    clang llvm-devel \
    openssl-devel \
    && dnf clean all

# Fix openblas symlink
RUN ln -sf /usr/lib64/libopenblas.so.0 /usr/lib64/libopenblas.so || true

# -----------------------------
# Create virtualenv (ALL arch)
# -----------------------------
RUN python3.12 -m venv /workspace/build/venv 
ENV PATH="/workspace/build/venv/bin:$PATH"

# Upgrade base tooling
RUN python3.12 -m pip install --upgrade pip setuptools wheel

# -----------------------------
# Python build tools for s390x
# -----------------------------
RUN if [ "$TARGETARCH" = "s390x" ]; then \
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable && \
        . "$HOME/.cargo/env" && \
        python3.12 -m pip install \
            "setuptools>=77.0.3,<81.0.0" \
            "setuptools-scm>=8.0" \
            "packaging>=24.2" \
            "pillow==10.4.0" \
            "cryptography==42.0.8" \
            "setuptools-rust" nanobind wheel cffi maturin \
            "numpy==2.4.4" \
            cmake pybind11 cython ninja scikit-build-core meson-python && \
        python3.12 -m pip install \
        "torch==2.10.0+cpu" \
        --index-url https://download.pytorch.org/whl/cpu ; \
    fi

# Check if pytorch is installed successfully
RUN if [ "$TARGETARCH" = "s390x" ]; then \
        python3.12 -c "import torch; print(torch.__version__)" ; \
    fi

# Env flags
ENV VLLM_USE_TRITON=0
ENV BINDGEN_EXTRA_CLANG_ARGS="-I/usr/include"
ENV GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
ENV GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1

ARG VLLM_BENCHMARK_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_BENCHMARK_BRANCH=v0.14.0
ARG OPENCV_VERSION=90
ARG OUTLINES_CORE_VERSION=0.2.11
ENV ENABLE_HEADLESS=1

# ---------------------------------------------
# Build OpenCV + outlines-core + vLLM on s390x 
# ---------------------------------------------
RUN if [ "$TARGETARCH" = "s390x" ]; then \
        . "$HOME/.cargo/env" && \

        echo "=== OpenCV ===" && \
        git clone --recursive https://github.com/opencv/opencv-python.git -b ${OPENCV_VERSION} && \
        cd opencv-python && \
        python3.12 -m pip install build scikit-build-core && \
        python3.12 -m build --wheel --outdir /tmp/wheels && \
        python3.12 -m pip install /tmp/wheels/opencv*.whl && \

        echo "=== outlines-core ===" && \
        cd /workspace && \
        git clone https://github.com/dottxt-ai/outlines-core.git && \
        cd outlines-core && \
        git checkout tags/${OUTLINES_CORE_VERSION} && \
        sed -i 's/version = "0.0.0"/version = "'"${OUTLINES_CORE_VERSION}"'"/' Cargo.toml && \
        python3.12 -m maturin build --release --out /tmp/wheels && \
        python3.12 -m pip install /tmp/wheels/outlines_core*.whl && \

        echo "=== vLLM ===" && \
        cd /workspace && \
        git clone --branch ${VLLM_BENCHMARK_BRANCH} ${VLLM_BENCHMARK_REPO} /tmp/vllm && \
        cd /tmp/vllm && \

        echo "numpy==2.4.4" > /tmp/constraints.txt && \
        echo "opencv-python-headless==4.13.0.90" >> /tmp/constraints.txt && \

        sed -i '/"torch == 2.10.0"/d' pyproject.toml && \
        sed -i '/^license\s*=.*/d' pyproject.toml && \
        sed -i '/^\[project\.license\]/,/^\[/d' pyproject.toml && \
        sed -i '/license-files/d' pyproject.toml && \
        sed -i '/^\[project\]/a license = { text = "Apache-2.0" }' pyproject.toml && \

        python3.12 -m pip install /tmp/wheels/opencv*.whl --force-reinstall && \

        VLLM_TARGET_DEVICE=empty python3.12 -m pip install . \
            --no-build-isolation \
            -c /tmp/constraints.txt && \

        rm -rf /tmp/vllm ; \
    else \
        echo "Skipping s390x builds for $TARGETARCH" ; \
    fi

RUN if [ "$TARGETARCH" = "s390x" ]; then \
        python3.12 -c "import torch, vllm, numpy; print('VLLM OK')" ; \
    fi

COPY Makefile Makefile
COPY pkg/preprocessing/chat_completions/ pkg/preprocessing/chat_completions/

RUN if [ "$TARGETARCH" != "s390x" ]; then \
        TARGETOS=${TARGETOS} TARGETARCH=${TARGETARCH} make install-python-deps; \
    else \
        echo "Skipping install-python-deps for s390x (vLLM already installed)"; \
    fi


RUN python3.12 -c "import torch, vllm, numpy; print('VLLM OK')"

# Build Stage: using Go 1.24.1 image
FROM registry.access.redhat.com/ubi9/go-toolset:1.24 AS builder

ARG TARGETOS
ARG TARGETARCH

WORKDIR /workspace

# Install system-level dependencies first. This layer is very stable.
USER root
# Install EPEL repository directly and then ZeroMQ, as epel-release is not in default repos.
# Install all necessary dependencies including Python 3.12 for chat-completions templating.
# The builder is based on UBI8, so we need epel-release-8.
RUN dnf install -y 'https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm' && \
    dnf install -y gcc-c++ libstdc++ libstdc++-devel clang zeromq-devel pkgconfig \ 
    python3.12-devel python3.12-pip openblas openblas-devel && \
    dnf install -y gcc-toolset-14 && \
    dnf clean all 

# Fix ld version mismatch
RUN if [ "$TARGETARCH" = "s390x" ]; then \
        ln -sf /usr/bin/ld \
        /opt/rh/gcc-toolset-14/root/usr/libexec/gcc/s390x-redhat-linux/14/ld; \
    fi

# Fix openblas symlink
RUN ln -sf /usr/lib64/libopenblas.so.0 /usr/lib64/libopenblas.so || true
ENV PATH=/opt/rh/gcc-toolset-14/root/usr/bin:$PATH

# Copy the Go Modules manifests
COPY go.mod go.mod
COPY go.sum go.sum
# cache deps before building and copying source so that we don't need to re-download as much
# and so that source changes don't invalidate our downloaded layer
RUN go mod download

# Copy the source code.
COPY . .

# Copy this project's own Python source code into the final image
COPY --from=python-builder /workspace/pkg/preprocessing/chat_completions /workspace/pkg/preprocessing/chat_completions
COPY --from=python-builder /workspace/build/venv /workspace/build/venv
# Fix python symlink for UBI environment
RUN ln -sf /usr/bin/python3.12 /workspace/build/venv/bin/python && \
    ln -sf /usr/bin/python3.12 /workspace/build/venv/bin/python3

ENV PATH=/workspace/build/venv/bin:$PATH

# Set the PYTHONPATH. This mirrors the Makefile's export, ensuring both this project's
# Python code and the installed libraries (site-packages) are found at runtime.
ENV PYTHONPATH=/workspace/pkg/preprocessing/chat_completions:/workspace/build/venv/lib/python3.12/site-packages:\
/workspace/build/venv/lib64/python3.12/site-packages

RUN /workspace/build/venv/bin/python -c "import tokenizer_wrapper" && \
    echo "tokenizer_wrapper OK"
RUN make build

# Use distroless as minimal base image to package the manager binary
# Refer to https://github.com/GoogleContainerTools/distroless for more details
FROM registry.access.redhat.com/ubi9/ubi:latest
WORKDIR /
USER root

RUN dnf install -y 'https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm' && \
    dnf install -y zeromq libxcrypt-compat python3.12 python3.12-pip \
    openblas openblas-devel libgomp \
    gcc-toolset-14 libjpeg-turbo && \
    dnf clean all

ENV LD_LIBRARY_PATH=/opt/rh/gcc-toolset-14/root/usr/lib64:$LD_LIBRARY_PATH
ENV PATH=/opt/rh/gcc-toolset-14/root/usr/bin:$PATH

# Fix openblas symlink
RUN ln -sf /usr/lib64/libopenblas.so /usr/lib64/libopenblas.so.0 || true

# Copy Python artifacts from python-builder
COPY --from=python-builder /workspace/pkg/preprocessing/chat_completions \
    /app/pkg/preprocessing/chat_completions
COPY --from=python-builder /workspace/build/venv /workspace/build/venv

# Fix python symlink AFTER venv is copied ← IMPORTANT
RUN ln -sf /usr/bin/python3.12 /workspace/build/venv/bin/python && \
    ln -sf /usr/bin/python3.12 /workspace/build/venv/bin/python3

ENV PATH=/workspace/build/venv/bin:$PATH
ENV PYTHONPATH=/app/pkg/preprocessing/chat_completions:/workspace/build/venv/lib64/python3.12/site-packages

RUN /workspace/build/venv/bin/python -c "import tokenizer_wrapper"

# Copy the compiled Go application
COPY --from=builder /workspace/bin/llm-d-kv-cache /app/kv-cache-manager

USER 65532:65532
ENTRYPOINT ["/app/kv-cache-manager"]
