# 1. Base Image: NVIDIA CUDA 11.8 + Ubuntu 22.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 2. 시스템 환경 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# 3. 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    libsndfile1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 4. python 명령어를 python3로 연결
RUN ln -s /usr/bin/python3 /usr/bin/python

# 5. 작업 디렉토리 설정
WORKDIR /app

# 6. 의존성 설치
COPY requirements.txt /app/

# pip 업그레이드 및 패키지 설치
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip install -r requirements.txt

# 7. 소스코드 복사
COPY . /app

# 8. 실행
CMD ["bash", "evaluation/eval_recon.sh"]
