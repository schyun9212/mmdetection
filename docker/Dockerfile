ARG PYTORCH="1.9.0"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get install -y tmux vim wget sudo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV
RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

# create a non-root user
ARG USER_ID=1000
ARG USER=appuser
RUN useradd -m --no-log-init --system  --uid ${USER_ID} ${USER} -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER ${USER}

RUN sudo chsh ${USER} -s $(which bash)

# Install tmux
RUN cd \
    && git clone https://github.com/gpakosz/.tmux.git \
    && ln -s -f .tmux/.tmux.conf \
    && cp .tmux/.tmux.conf.local .

# Install MMDetection
RUN conda clean --all
RUN git clone https://github.com/schyun9212/mmdetection.git /home/${USER}/mmdetection
WORKDIR /home/${USER}/mmdetection
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .
