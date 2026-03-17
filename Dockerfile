FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY utils/ utils/
COPY init/ init/
COPY app.py .
COPY setup_f5tts.sh .

# F5TTS: копируем локальную папку если есть, иначе скачиваем с Google Drive
RUN --mount=type=bind,source=.,target=/ctx \
    if [ -d "/ctx/F5TTS" ] && [ -f "/ctx/F5TTS/ckpts/model_v4.safetensors" ]; then \
        echo "Копируем локальный F5TTS..." && cp -r /ctx/F5TTS /app/F5TTS; \
    else \
        echo "Скачиваем F5TTS..." && bash setup_f5tts.sh; \
    fi

RUN mkdir -p uploads

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
