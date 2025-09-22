# CUDA 11.8 + cuDNN8 on Ubuntu 20.04 (Python 3.8 default)
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps (includes nano)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 python3.8-dev python3.8-venv python3-pip \
    git git-lfs nano wget curl ca-certificates \
    build-essential ninja-build cmake \
    ffmpeg libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
    && rm -rf /var/lib/apt/lists/* && git lfs install

# Make "python" and "pip" point to Python 3.8 explicitly
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Keep pip tooling fresh
RUN pip install --upgrade pip setuptools wheel

# PyTorch 1.13.1 + cu117 (works on Ada inside CUDA 11.8 container)
ARG TORCH_VER=1.13.1+cu117
ARG TV_VER=0.14.1+cu117
ARG TA_VER=0.13.1
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu117 \
        torch==${TORCH_VER} torchvision==${TV_VER} torchaudio==${TA_VER}

# Headless OpenCV for servers/containers
RUN pip install --no-cache-dir "opencv-python-headless>=4.7.0"

# --- Fix for Python 3.8: stable Numba/llvmlite pair ---
RUN pip install --no-cache-dir "llvmlite==0.39.1" "numba==0.56.4"

# Small libs needed by StreamPETR EVA-ViT & utilities
RUN pip install --no-cache-dir \
    einops==0.6.1 \
    fvcore==0.1.5.post20221221 \
    iopath==0.1.10 \
    yacs==0.1.8 \
    timm==0.6.13 \
    tqdm==4.66.4 \
    pycocotools==2.0.7

# OpenMMLab pins expected by StreamPETR
RUN pip install --no-cache-dir \
    'mmcv-full==1.6.0' -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html && \
    pip install --no-cache-dir mmdet==2.28.2 mmsegmentation==0.30.0

# nuScenes eval deps
RUN pip install --no-cache-dir nuscenes-devkit==1.1.10 pyquaternion shapely

# Clone StreamPETR + mmdet3d v1.0.0rc6 and install (editable)
WORKDIR /workspace
RUN git clone https://github.com/exiawsh/StreamPETR && \
    cd StreamPETR && \
    git clone https://github.com/open-mmlab/mmdetection3d.git && \
    cd mmdetection3d && git checkout v1.0.0rc6 && \
    pip install --no-cache-dir -U pip setuptools wheel cython pybind11 && \
    pip install --no-cache-dir -e .

# FlashAttention for Torch 1.13 + cu117 (build from source)
ENV CUDA_HOME=/usr/local/cuda
RUN pip install --no-cache-dir packaging && \
    pip install --no-build-isolation --no-cache-dir "flash-attn==0.2.8"

# Environment & QoL
# Put mmdetection3d FIRST to avoid converter shadowing; no dangling $PYTHONPATH
ENV PYTHONUNBUFFERED=1 \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9" \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONPATH=/workspace/StreamPETR/mmdetection3d:/workspace/StreamPETR

WORKDIR /workspace/StreamPETR

# Build-time sanity print (optional)
RUN python - <<'PY'
import sys, torch, mmcv, mmdet
import importlib
mmdet3d = importlib.import_module("mmdet3d")
print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, "GPU?", torch.cuda.is_available())
print("mmcv:", mmcv.__version__, "mmdet:", mmdet.__version__, "mmdet3d:", getattr(mmdet3d,"__version__","installed"))
PY
