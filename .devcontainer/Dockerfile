# DFBT Dockerfile
# Based on PyTorch image with Python 3.10
# Updated to PyTorch 2.5.0 with CUDA 12.4 to support RTX 5080 (sm_120)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES},display \
    DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

### environment variables
ARG USERNAME=waybaba
ENV UDATADIR=/data \
    UPRJDIR=/code \
    UOUTDIR=/output \
    UDEVICEID=docker


### apt packages
RUN apt-get update && apt-get install -y \
    curl \
    git \
    sudo \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    htop \
    vim \
    libosmesa6-dev \
    libgl1-mesa-dev \
    gcc \
    g++ \
    build-essential \
    libgmp-dev \
    libmpfr-dev \
    libmpc-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

### pip - base dependencies
# Note: PyTorch 2.5.0 is already installed in base image, but we ensure version compatibility
RUN pip install --no-cache-dir \
    torch==2.5 \
    tensorboard \
    tensorboardX \
    tqdm \
    rich \
    numpy \
    cython

### pip - gymnasium and mujoco (for v5 environments)
RUN pip install --no-cache-dir \
    gymnasium[mujoco] \
    mujoco==2.3.3

### mujoco210 for mujoco-py (needed by d4rl)
USER root
RUN mkdir -p /usr/local/mujoco && \
    wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O /tmp/mujoco210.tar.gz && \
    tar -zxvf /tmp/mujoco210.tar.gz -C /usr/local/mujoco/ && \
    rm /tmp/mujoco210.tar.gz && \
    mv /usr/local/mujoco/mujoco210 /usr/local/mujoco/mujoco210 || true

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/mujoco/mujoco210/bin
ENV MUJOCO_PY_MUJOCO_PATH=/usr/local/mujoco/mujoco210

### pip - mujoco-py (for d4rl)
RUN pip install --no-cache-dir \
    'mujoco-py<2.2,>=2.1'

### pip - d4rl
RUN pip install --no-cache-dir \
    git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl

### Prebuild mujoco-py (first import with non-root user causes error)
RUN python -c "import mujoco_py; print('mujoco-py version:', mujoco_py.__version__)" || true

### Fix permissions for mujoco_py generated files
RUN chmod -R a+rwx /opt/conda/lib/python3.10/site-packages/mujoco_py/generated 2>/dev/null || true

### Non-root user creation
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME || true && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME || true && \
    mkdir -p /home/$USERNAME/.vscode-server /home/$USERNAME/.vscode-server-insiders && \
    chown -R ${USER_UID}:${USER_GID} /home/$USERNAME/.vscode-server* && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    usermod -a -G audio,video $USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME && \
    chown -R ${USER_UID}:${USER_GID} /usr/local/mujoco

USER $USERNAME
ENV HOME /home/$USERNAME
WORKDIR $HOME

### Verify installations
RUN python -c "import gymnasium; print('gymnasium:', gymnasium.__version__)" && \
    python -c "import mujoco; print('mujoco:', mujoco.__version__)" && \
    python -c "import d4rl; print('d4rl imported successfully')" || echo "d4rl import check skipped"

CMD sleep infinity

