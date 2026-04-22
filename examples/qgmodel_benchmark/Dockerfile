FROM debian:bookworm-slim AS builder

# Stage 1: build environment
# Install basic build tools, MPICH, FFTW, Python, and dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    bash wget git gnupg2 ca-certificates build-essential \
    mpich libmpich-dev libfftw3-dev\
    python3 python3-pip python3-dev python3-venv && \
    rm -rf /var/lib/apt/lists/*

# Add Intel's repository and install just the IFX compiler
RUN wget -qO - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | apt-key add - && \
    echo "deb https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/oneAPI.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    intel-oneapi-compiler-fortran && \
    rm -rf /var/lib/apt/lists/*

# Install NEDAS and required Python packages
RUN git clone https://github.com/nansencenter/NEDAS.git /opt/NEDAS
WORKDIR /opt/NEDAS
RUN git checkout -b develop origin/develop

RUN python3 -m venv /opt/pyenv
ENV PATH=/opt/pyenv/bin:$PATH
RUN pip3 install --upgrade --no-cache-dir pip && \
    pip3 install  --no-cache-dir -r requirements.txt && \
    pip3 install  --no-cache-dir numba pyFFTW && \
    MPICC=mpicc pip3 install --no-cache-dir --no-binary=mpi4py mpi4py

# Build qg model
ENV PATH=/opt/intel/oneapi/compiler/latest/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/latest/lib
ENV FFTW_DIR=/lib/x86_64-linux-gnu
ENV LIBM_DIR=/lib/x86_64-linux-gnu
WORKDIR /opt/NEDAS/NEDAS/models/qg/src
RUN make

# Stage 2: runtime environment
FROM debian:bookworm-slim

# install the dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    mpich python3 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/NEDAS /app/NEDAS
COPY --from=builder /opt/pyenv /opt/pyenv
COPY --from=builder \
    /opt/intel/oneapi/compiler/latest/lib/libimf.so* \
    /opt/intel/oneapi/compiler/latest/lib/libintlc.so* \
    /usr/local/lib
COPY --from=builder /lib/x86_64-linux-gnu/libfftw3.so* /lib/x86_64-linux-gnu

ENV PYTHONPATH=/app/NEDAS
ENV PATH=/opt/pyenv/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/lib:/lib/x86_64-linux-gnu

WORKDIR /app
COPY setup_docker.src /app
COPY config_docker.yml /app/config.yml
COPY *sh /app

# Create a non-root user with the same UID/GID as the host user
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} appgroup && \
    useradd -u ${USER_ID} -g appgroup -d /app -s /bin/bash appuser && \
    chown -R ${USER_ID}:${GROUP_ID} /app

# Switch to the new user
USER appuser

# Create working directory
WORKDIR /app

# Default command
CMD ["bash"]
