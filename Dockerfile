ARG IMGAE=nvidia/cuda:11.0.3-devel-ubuntu20.04
FROM ${IMGAE}

SHELL ["bash", "-c"]

LABEL maintainer="ktro310115@gmail.com" \
        description="wrapper of https://github.com/PruneTrunong/DenseMatching"

# setup timezone
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# install base packages
RUN apt-get update && apt-get install -y \
    apt-utils \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    curl \
    lsb-release \
    ninja-build \
    git \
    python3 \
    python3-pip

# set up PruneTruong/DenseMatching
RUN git clone --recursive https://github.com/PruneTruong/DenseMatching
WORKDIR /DenseMatching

ARG TORCH=1.7.1+cu110
ARG TORCHVISION=0.8.2+cu110

RUN pip install \
    torch==${TORCH} \
    torchvision==${TORCHVISION} \
    -f https://download.pytorch.org/whl/torch_stable.html

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY scripts/api.py .
COPY scripts/utils.py .
COPY scripts/schemas.py .
COPY scripts/logger.py .

RUN git submodule update --init --recursive && \
    git submodule update --recursive --remote && \
    python3 -c "from admin.environment import create_default_local_file; create_default_local_file()" && \
    bash assets/download_pre_trained_models.sh

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
  ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
  ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

ENV HOST=0.0.0.0
ENV PORT=8000
ENV GUNICORN_CMD_ARGS="--keep-alive 0"

CMD uvicorn api:app --host=${HOST} --port=${PORT} --reload
