FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Kiev

WORKDIR /app-recognition

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    postgresql-client \
    build-essential \
    libpq-dev \
    dos2unix \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    git \
    tzdata \
    && ln -fs /usr/share/zoneinfo/Europe/Kiev /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

RUN pip install torch torchvision
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

COPY . .

RUN dos2unix scripts/wait-for-db.sh && chmod +x scripts/wait-for-db.sh
RUN dos2unix scripts/wait-for-sonar.sh && chmod +x scripts/wait-for-sonar.sh

EXPOSE 8000