FROM liny2/pytorch:1.13.1-py3.10.15-cuda11.7.1-devel-ubuntu20.04

# setup timezone
ENV DEBIAN_FRONTEND=noninteractive
RUN echo 'Etc/UTC' > /etc/timezone && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub \
    && apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && rm -rf /var/lib/apt/lists/*


# Install OpenMMLab projects
RUN pip install --no-deps \
    mmengine==0.7.3 \
    mmdet==3.0.0 \
    mmsegmentation==1.0.0 \
    git+https://github.com/open-mmlab/mmdetection3d.git@22aaa47fdb53ce1870ff92cb7e3f96ae38d17f61
RUN pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html --no-deps

# Install MinkowskiEngine
# Feel free to skip nvidia-cuda-dev if minkowski installation is fine
RUN apt-get update \
    && apt-get -y install libopenblas-dev   -o Acquire::Retries=5
RUN TORCH_CUDA_ARCH_LIST="7.0 8.6" \
    pip install git+https://github.com/NVIDIA/MinkowskiEngine.git@02fc608bea4c0549b0a7b00ca1bf15dee4a0b228 -v --no-deps \
    --config-settings="--build-option=--blas=openblas" \
    --config-settings="--build-option=--force_cuda" \
    --no-build-isolation

# Install torch-scatter 
RUN pip install pyg_lib torch-scatter==2.1.1  torch_cluster==1.6.1 -f https://data.pyg.org/whl/torch-1.13.1+cu117.html --no-deps

# Install ScanNet superpoint segmentator
RUN git clone https://github.com/Karbo123/segmentator.git \
    && cd segmentator/csrc \
    && git reset --hard 76efe46d03dd27afa78df972b17d07f2c6cfb696 \
    && mkdir build \
    && cd build \
    && cmake .. \
    -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
    -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
    -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
    -DCMAKE_INSTALL_PREFIX=`python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())'` \
    && make \
    && make install \
    && cd ../../..

# 安装项目中dependencies文件夹下的包
COPY dependencies/ /dependencies/



# Install remaining python packages
RUN pip install --no-deps  --retries 5 \
    spconv-cu117==2.3.6 \
    addict==2.4.0 \
    yapf==0.33.0 \
    termcolor==2.3.0 \
    packaging==23.1 \
    numpy==1.24.1 \
    rich==13.3.5 \
    opencv-python==4.7.0.72 \
    pycocotools==2.0.6 \
    Shapely==1.8.5 \
    scipy==1.10.1 \
    terminaltables==3.1.10 \
    numba==0.57.0 \
    llvmlite==0.40.0 \
    pccm==0.4.7 \
    ccimport==0.4.2 \
    pybind11==2.10.4 \
    ninja==1.11.1 \
    lark==1.1.5 \
    cumm-cu116==0.4.9 \
    pyquaternion==0.9.9 \
    lyft-dataset-sdk==0.0.8 \
    pandas==2.0.1 \
    python-dateutil==2.8.2 \
    matplotlib==3.5.2 \
    pyparsing==3.0.9 \
    cycler==0.11.0 \
    kiwisolver==1.4.4 \
    scikit-learn==1.2.2 \
    joblib==1.2.0 \
    threadpoolctl==3.1.0 \
    cachetools==5.3.0 \
    nuscenes-devkit==1.1.10 \
    trimesh==3.21.6 \
    open3d==0.17.0 \
    plotly==5.18.0 \
    dash==2.14.2 \
    plyfile==1.0.2 \
    flask==3.0.0 \
    werkzeug==3.0.1 \
    click==8.1.7 \
    blinker==1.7.0 \
    itsdangerous==2.1.2 \
    importlib_metadata==2.1.2 \
    zipp==3.17.0 \
    h5py \
    torchmetrics==0.11.4 \
    pgeof \
    omegaconf \
    colorhash \
    hydra-core \
    hydra-colorlog \
    pytorch-lightning==2.2 


RUN apt-get update && apt-get install -q -y --no-install-recommends \
    dirmngr \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# setup keys
RUN set -eux; \
    key='C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654'; \
    export GNUPGHOME="$(mktemp -d)"; \
    gpg --batch --keyserver keyserver.ubuntu.com --recv-keys "$key"; \
    mkdir -p /usr/share/keyrings; \
    gpg --batch --export "$key" > /usr/share/keyrings/ros1-latest-archive-keyring.gpg; \
    gpgconf --kill all; \
    rm -rf "$GNUPGHOME"

# setup sources.list
RUN echo "deb [ signed-by=/usr/share/keyrings/ros1-latest-archive-keyring.gpg ] http://mirrors.ustc.edu.cn/ros/ubuntu/ focal main" > /etc/apt/sources.list.d/ros1-latest.list

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ENV ROS_DISTRO noetic

# install ros packages
RUN apt-get update && apt-get install -o Acquire::Retries=10 -y --no-install-recommends \
    ros-noetic-ros-core  build-essential\
    && rm -rf /var/lib/apt/lists/*