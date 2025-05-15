FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /recognition

RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client \
    build-essential \
    libpq-dev \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

RUN dos2unix scripts/wait-for-db.sh && chmod +x scripts/wait-for-db.sh
RUN dos2unix scripts/wait-for-sonar.sh && chmod +x scripts/wait-for-sonar.sh

EXPOSE 8000

