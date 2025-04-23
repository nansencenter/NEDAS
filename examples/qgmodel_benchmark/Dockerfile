FROM ubuntu:22.04

# Install basic build tools, MPICH, Python, and MPI4Py dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    vim git wget gnupg2 ca-certificates build-essential \
    mpich libmpich-dev \
    python3 python3-pip python3-dev python3-venv && \
    rm -rf /var/lib/apt/lists/*

# Add Intel's repository and install just the IFX compiler
RUN wget -qO - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | apt-key add - && \
    echo "deb https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/oneAPI.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    intel-oneapi-compiler-fortran && \
    rm -rf /var/lib/apt/lists/*

# Accept build arguments (fallback to 1000 if not provided)
ARG USER_ID=1000
ARG GROUP_ID=1000

# Create a non-root user with the same UID/GID as the host user
RUN groupadd -g ${GROUP_ID} appgroup && \
    useradd -u ${USER_ID} -g appgroup -d /app -s /bin/bash appuser && \
    mkdir -p /app && \
    chown -R ${USER_ID}:${GROUP_ID} /app

# Switch to the new user
USER appuser

# Python environment
RUN python3 -m venv /app/pyenv
RUN echo '. /app/pyenv/bin/activate' > /app/python.src
ENV PATH=/app/pyenv/bin:$PATH

# Install NEDAS and required Python packages
RUN git clone https://github.com/nansencenter/NEDAS.git /app/NEDAS
WORKDIR /app/NEDAS
RUN git checkout -b other_features origin/other_features
ENV PYTHONPATH=/app/NEDAS:$PYTHONPATH

RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt && \
    pip3 install numba pyFFTW cmocean jupyter && \
    MPICC=mpicc pip3 install --no-binary=mpi4py mpi4py

# Set up environment variables for IFX
ENV PATH=/opt/intel/oneapi/compiler/latest/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/latest/lib:${LD_LIBRARY_PATH}

# Create working directory
WORKDIR /work

# Default command
CMD ["/bin/bash"]
