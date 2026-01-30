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

# ==================================================
# CONFIG
# ==================================================
cfg = configparser.ConfigParser()
cfg.read("config.ini", encoding="utf-8")

WHISPER_MODEL = cfg.get("WHISPER", "model")
TMP_DIR = cfg.get("BITRIX", "tmp_dir", fallback=tempfile.gettempdir())
EXPECTED_APP_TOKEN = cfg.get("BITRIX", "outgoing_app_token")
AUDIO_BASE_URL = cfg.get("BITRIX", "audio_base_url")

REQUIRED_PHRASES = [
    p.strip()
    for p in cfg.get("SCRIPT", "required_phrases").split(",")
    if p.strip()
]

os.makedirs(TMP_DIR, exist_ok=True)

# ==================================================
# LOGGING
# ==================================================
logger = logging.getLogger("bitrix_analyzer")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(handler)

# ==================================================
# APP / CLIENT
# ==================================================
app = Flask(__name__)
client = BitrixClient()

# ==================================================
# WHISPER
# ==================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_engine = WhisperEngine(model_name=WHISPER_MODEL, device=device)

# ==================================================
# FALLBACK DIARIZER
# ==================================================
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
        return [{"start": 0.0, "end": duration, "speaker": "SPEAKER"}], duration

diarizer = SimpleFallbackDiarizer()

# ==================================================
# INTERNAL STATE
# ==================================================
PROCESSING_LOCK = set()
MAX_RETRY = 6           # ~6 минут ожидания MP3
RETRY_DELAY = 60        # секунд

# ==================================================
# MAIN CALL PROCESSOR
# ==================================================
def process_call(activity_id: int, attempt: int = 1):
    if activity_id in PROCESSING_LOCK:
        logger.info(f"[{activity_id}] Уже в обработке — пропуск")
        return

    PROCESSING_LOCK.add(activity_id)
    start_time = datetime.now()
    audio_path = None

    try:
        logger.info(f"[{activity_id}] Попытка {attempt}: получение активности")
        activity = client.get_call_activity(activity_id)
        if not activity:
            logger.error(f"[{activity_id}] CRM activity не найдена")
            return

        # -------- FILE_ID / URL --------
        file_id = None
        file_url = None

        if activity.get("FILES") and len(activity["FILES"]) > 0:
            file_entry = activity["FILES"][0]
            file_id = str(file_entry.get("id") or file_entry.get("FILE_ID"))
            file_url = file_entry.get("url")  # используем готовый URL, если есть
        elif activity.get("FILE_ID"):
            file_id = str(activity.get("FILE_ID"))
            file_url = AUDIO_BASE_URL + file_id

        if not file_id:
            if attempt < MAX_RETRY:
                logger.info(f"[{activity_id}] FILE_ID нет, повтор через {RETRY_DELAY} сек")
                threading.Timer(RETRY_DELAY, process_call, args=(activity_id, attempt + 1)).start()
            else:
                logger.warning(f"[{activity_id}] FILE_ID так и не появился — отказ")
            return

        logger.info(f"[{activity_id}] FILE_ID найден: {file_id}, URL: {file_url}")

        # -------- DOWNLOAD AUDIO --------
        audio_path = client.download_audio(file_id, file_url=file_url)
        if not audio_path:
            logger.error(f"[{activity_id}] Ошибка скачивания аудио")
            return

        logger.info(f"[{activity_id}] Аудио скачано: {audio_path}")

        # -------- WHISPER --------
        try:
            segments = whisper_engine.transcribe_segments(audio_path)
            full_text = " ".join(s.get("text", "") for s in segments).strip()
        except Exception:
            logger.exception(f"[{activity_id}] Ошибка Whisper")
            full_text = ""
            segments = []

        # -------- DIARIZATION --------
        diar_segments, _ = diarizer.diarize(audio_path)

        # -------- ANALYSIS --------
        script_result = analyze_script_presence(full_text, REQUIRED_PHRASES)
        interests = InterestsPlugin().analyze(full_text, diar_segments)
        informative = is_informational_call(full_text)

        # -------- CRM COMMENT --------
        owner_id = activity.get("OWNER_ID")
        owner_type = activity.get("OWNER_TYPE_ID")

        comment_text = (
            "📞 <b>Автоматический анализ звонка</b><br><br>"
            f"<b>Скрипт:</b> {script_result.get('percent', 0)}%<br>"
            f"<b>Информативный:</b> {'Да' if informative else 'Нет'}<br>"
            f"<b>Пропущенные фразы:</b> "
            f"{', '.join(script_result.get('missing', [])) or '—'}"
        )

        client.add_comment(owner_type_id=owner_type, owner_id=owner_id, text=comment_text)
        logger.info(f"[{activity_id}] Анализ успешно завершён")

    except Exception:
        logger.exception(f"[{activity_id}] Критическая ошибка")

    finally:
        PROCESSING_LOCK.discard(activity_id)
        try:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass
        gc.collect()
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"[{activity_id}] Завершено за {elapsed:.1f} сек")

# ==================================================
# WEBHOOK
# ==================================================
@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        data = request.get_json(silent=True) or {}
        logger.info(f"RAW WEBHOOK: {data}")

        # ---- AUTH ----
        app_token = data.get("auth", {}).get("application_token")
        if app_token != EXPECTED_APP_TOKEN:
            logger.warning(f"Неверный application_token: {app_token}")
            return jsonify({"status": "forbidden"}), 403

        event = data.get("event")
        fields = data.get("data", {}).get("FIELDS", {})
        logger.info(f"EVENT: {event}")

        if event in ("ONCRMACTIVITYADD", "ONCRMACTIVITYUPDATE"):
            activity_id = fields.get("ID")
            provider = fields.get("PROVIDER_ID")
            if activity_id and provider == "VOXIMPLANT_CALL":
                threading.Thread(target=process_call, args=(int(activity_id),), daemon=True).start()

        return jsonify({"status": "ok"}), 200

    except Exception:
        logger.exception("Webhook error")
        return jsonify({"status": "error"}), 500

# ==================================================
# RUN
# ==================================================
if __name__ == "__main__":
    logger.info("🚀 Flask webhook: 0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000)
