#!/bin/bash
# Скачивание F5TTS (код + чекпоинты модели) с Google Drive
# Использование: ./setup_f5tts.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
F5TTS_DIR="${SCRIPT_DIR}/F5TTS"
ARCHIVE="F5TTS.tar.gz"

# Google Drive file ID — вставь сюда свой ID один раз
GDRIVE_ID="https://drive.google.com/file/d/1j1i_L4el5xDGOfkMSrJBX2UWlDcIrqu3/view?usp=sharing"

if [ -d "$F5TTS_DIR" ] && [ -f "$F5TTS_DIR/ckpts/model_v4.safetensors" ]; then
    echo "F5TTS уже существует. Пропускаем загрузку."
    exit 0
fi

if [ "$GDRIVE_ID" = "YOUR_GDRIVE_FILE_ID_HERE" ]; then
    echo "ОШИБКА: Укажи Google Drive file ID в setup_f5tts.sh (переменная GDRIVE_ID)"
    exit 1
fi

echo "Скачиваем F5TTS с Google Drive..."

if ! command -v gdown &> /dev/null; then
    echo "Устанавливаем gdown..."
    pip install --quiet gdown
fi

cd "$SCRIPT_DIR"
gdown "$GDRIVE_ID" -O "$ARCHIVE"

echo "Распаковываем F5TTS..."
tar -xzf "$ARCHIVE"
rm -f "$ARCHIVE"

echo "F5TTS готов."
