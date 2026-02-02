import os
import gc
import logging
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
EXPECTED_APP_TOKEN = cfg.get("BITRIX", "outgoing_app_token")
TMP_DIR = cfg.get("BITRIX", "tmp_dir", fallback=tempfile.gettempdir())

REQUIRED_PHRASES = [
    p.strip()
    for p in cfg.get("SCRIPT", "required_phrases").split(",")
    if p.strip()
]

os.makedirs(TMP_DIR, exist_ok=True)

# ==================================================
# LOGGING
# ==================================================
logger = logging.getLogger("bitrix_main")
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
MAX_RETRY = 6
RETRY_DELAY = 60

# ==================================================
# CALL PROCESSOR
# ==================================================
def process_call(activity_id: int, attempt: int = 1):
    if activity_id in PROCESSING_LOCK:
        logger.info("[%s] Already processing", activity_id)
        return

    PROCESSING_LOCK.add(activity_id)
    start_time = datetime.now()
    audio_path = None

    try:
        logger.info("[%s] Attempt %s: get activity", activity_id, attempt)
        activity = client.get_call_activity(activity_id)

        if not activity:
            logger.error("[%s] Activity not found", activity_id)
            return

        # -------- FILE_ID --------
        file_id = None
        files = activity.get("FILES") or []
        if files:
            file_id = str(files[0].get("id") or files[0].get("FILE_ID"))

        if not file_id:
            if attempt < MAX_RETRY:
                logger.info("[%s] FILE_ID not ready, retry in %s sec", activity_id, RETRY_DELAY)
                threading.Timer(RETRY_DELAY, process_call, args=(activity_id, attempt + 1)).start()
            else:
                logger.warning("[%s] FILE_ID not found after retries", activity_id)
            return

        logger.info("[%s] FILE_ID: %s", activity_id, file_id)

        # -------- DOWNLOAD AUDIO --------
        audio_path = client.download_audio(file_id)
        if not audio_path:
            logger.error("[%s] Audio download failed", activity_id)
            return

        # -------- WHISPER --------
        try:
            segments = whisper_engine.transcribe_segments(audio_path)
            full_text = " ".join(s.get("text", "") for s in segments).strip()
        except Exception:
            logger.exception("[%s] Whisper failed", activity_id)
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

        comment_lines = [
            "📞 Автоматический анализ звонка",
            f"Скрипт: {len(script_result.get('found', []))}/{len(REQUIRED_PHRASES)} фраз найдено",
            f"Информативный: {'Да' if informative else 'Нет'}",
            f"Пропущенные фразы: {', '.join(script_result.get('missed', [])) or '—'}"
        ]
        comment_text = "\n".join(comment_lines)

        logger.info("[%s] Sending comment to Bitrix...", activity_id)
        client.add_comment(owner_type, owner_id, comment_text)
        logger.info("[%s] Analysis completed", activity_id)

    except Exception:
        logger.exception("[%s] Critical error", activity_id)

    finally:
        PROCESSING_LOCK.discard(activity_id)

        try:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass

        gc.collect()
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info("[%s] Finished in %.1f sec", activity_id, elapsed)

# ==================================================
# WEBHOOK
# ==================================================
@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        data = request.get_json(silent=True) or {}
        logger.info("RAW WEBHOOK: %s", data)

        token = data.get("auth", {}).get("application_token")
        if token != EXPECTED_APP_TOKEN:
            logger.warning("Invalid application_token: %s", token)
            return jsonify({"status": "forbidden"}), 403

        event = data.get("event")
        fields = data.get("data", {}).get("FIELDS", {})
        logger.info("EVENT: %s", event)

        activity_id = fields.get("ID")
        provider = fields.get("PROVIDER_ID")

        # Обрабатываем все звонки
        if activity_id and provider in ("VOXIMPLANT_CALL", "ASTERISK_CALL", "CALL"):
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
