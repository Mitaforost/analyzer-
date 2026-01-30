import os
import logging
import gc
import configparser
import tempfile
import threading
from datetime import datetime

from flask import Flask, request, jsonify
import torch

from core.whisper_engine import WhisperEngine
from core.analyzers import (
    analyze_script_presence,
    InterestsPlugin,
    is_informational_call
)
from bitrix_client import BitrixClient


# ========================
# Конфигурация
# ========================
cfg = configparser.ConfigParser()
cfg.read("config.ini", encoding="utf-8")

WHISPER_MODEL = cfg.get("WHISPER", "model")
REQUIRED_PHRASES = [
    p.strip()
    for p in cfg.get("SCRIPT", "required_phrases").split(",")
    if p.strip()
]

TMP_DIR = cfg.get("BITRIX", "tmp_dir", fallback=tempfile.gettempdir())
os.makedirs(TMP_DIR, exist_ok=True)

# токен исходящего вебхука (ИМЕННО application_token)
EXPECTED_APP_TOKEN = cfg.get("BITRIX", "outgoing_app_token")


# ========================
# Логгер
# ========================
logger = logging.getLogger("bitrix_analyzer")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
)
logger.addHandler(handler)


# ========================
# Flask + Bitrix
# ========================
app = Flask(__name__)
client = BitrixClient()


# ========================
# Whisper
# ========================
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_engine = WhisperEngine(
    model_name=WHISPER_MODEL,
    device=device
)


# ========================
# Fallback Diarizer
# ========================
class SimpleFallbackDiarizer:
    def diarize(self, wav_path):
        import wave
        import contextlib

        try:
            with contextlib.closing(wave.open(wav_path, 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)
        except Exception:
            duration = 0.0

        segments = [{
            "start": 0.0,
            "end": duration,
            "speaker": "SPEAKER"
        }]
        return segments, duration


diarizer = SimpleFallbackDiarizer()


# ========================
# Основная обработка по activity_id
# ========================
def process_call(activity_id: int):
    start_time = datetime.now()
    logger.info(f"[{activity_id}] Начата обработка звонка")

    try:
        activity = client.get_call_activity(activity_id)
        if not activity:
            logger.error(f"[{activity_id}] CRM activity не найдена")
            return

        # --- Получаем FILE_ID
        file_id = None

        if activity.get("FILES"):
            file_id = activity["FILES"][0].get("FILE_ID")
        elif activity.get("FILE_ID"):
            file_id = activity.get("FILE_ID")

        if not file_id:
            logger.warning(f"[{activity_id}] FILE_ID ещё не появился — пропуск")
            return

        # --- Скачиваем аудио
        audio_path = client.download_audio(file_id)
        if not audio_path:
            logger.error(f"[{activity_id}] Ошибка скачивания аудио")
            return

        logger.info(f"[{activity_id}] Аудио скачано")

        # --- Whisper
        try:
            segments = whisper_engine.transcribe_segments(audio_path)
            full_text = " ".join(
                s.get("text", "") for s in segments
            ).strip()
        except Exception:
            logger.exception(f"[{activity_id}] Ошибка Whisper")
            full_text = ""
            segments = []

        # --- Диаризация (fallback)
        diar_segments, _ = diarizer.diarize(audio_path)

        # --- Анализ
        script_result = analyze_script_presence(full_text, REQUIRED_PHRASES)
        interests = InterestsPlugin().analyze(full_text, diar_segments)
        informative = is_informational_call(full_text)

        # --- Комментарий в CRM
        owner_id = activity.get("OWNER_ID")
        owner_type = activity.get("OWNER_TYPE_ID")  # 1=lead, 2=deal

        comment_text = (
            "📞 *Автоматический анализ звонка*\n\n"
            f"Процент выполнения скрипта: {script_result.get('percent', 0)}%\n"
            f"Информативный звонок: {'Да' if informative else 'Нет'}\n"
            f"Пропущенные фразы: "
            f"{', '.join(script_result.get('missing', [])) or '—'}"
        )

        client.add_comment(
            owner_type_id=owner_type,
            owner_id=owner_id,
            text=comment_text
        )

        logger.info(f"[{activity_id}] Анализ завершён")

    except Exception:
        logger.exception(f"[{activity_id}] Критическая ошибка")

    finally:
        try:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass

        gc.collect()
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"[{activity_id}] Завершено за {elapsed:.2f} сек")


# ========================
# Поиск activity по CALL_ID
# ========================
def process_call_by_call_id(call_id: str):
    logger.info(f"[CALL_ID={call_id}] Поиск CRM activity")
    activity = client.find_activity_by_call_id(call_id)

    if not activity:
        logger.warning(f"[CALL_ID={call_id}] Activity не найдена")
        return

    process_call(int(activity["ID"]))


# ========================
# WEBHOOK
# ========================
@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        data = request.get_json(silent=True) or {}
        # --- Проверка токена
        app_token = data.get("auth", {}).get("application_token")
        if app_token != EXPECTED_APP_TOKEN:
            logger.warning("Неверный application_token")
            return jsonify({"status": "forbidden"}), 403

        event = data.get("event")
        payload = data.get("data", {})

        logger.info(f"Событие: {event}")

        # --- CRM Activity UPDATE (ключевое)
        if event == "ONCRMACTIVITYUPDATE":
            activity_id = payload.get("FIELDS", {}).get("ID")
            if activity_id:
                threading.Thread(
                    target=process_call,
                    args=(int(activity_id),),
                    daemon=True
                ).start()

        # --- Завершение звонка (fallback)
        elif event == "ONVOXIMPLANTCALLEND":
            call_id = payload.get("CALL_ID")
            if call_id:
                threading.Thread(
                    target=process_call_by_call_id,
                    args=(call_id,),
                    daemon=True
                ).start()

        return jsonify({"status": "ok"}), 200

    except Exception:
        logger.exception("Ошибка обработки вебхука")
        return jsonify({"status": "error"}), 500


# ========================
# RUN
# ========================
if __name__ == "__main__":
    logger.info("Flask сервер запущен: 0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000)
