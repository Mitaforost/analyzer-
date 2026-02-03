import os
import gc
import time
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
logger.handlers.clear()
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
# INTERNAL SETTINGS
# ==================================================
MAX_RETRY = 15
RETRY_DELAY = 60  # секунд
PROCESSING = set()
LOCK = threading.Lock()

# ==================================================
# HELPERS
# ==================================================
def extract_file_id(activity: dict) -> str | None:
    files = activity.get("FILES") or []
    if files:
        fid = files[0].get("id") or files[0].get("FILE_ID")
        if fid:
            return str(fid)

    storage_ids = activity.get("STORAGE_ELEMENT_IDS") or []
    if storage_ids:
        return str(storage_ids[0])

    settings = activity.get("SETTINGS") or {}
    for key in ("FILE_ID", "RECORD_ID"):
        if key in settings:
            return str(settings[key])

    return None

# ==================================================
# CALL PROCESSOR
# ==================================================
def process_call(activity_id: int):
    with LOCK:
        if activity_id in PROCESSING:
            logger.info("[%s] Already processing", activity_id)
            return
        PROCESSING.add(activity_id)

    start_time = datetime.now()
    audio_path = None

    try:
        logger.info("[%s] Processing started", activity_id)
        activity = None
        file_id = None

        # ---------- WAIT FOR FILE ----------
        for attempt in range(1, MAX_RETRY + 1):
            activity = client.get_call_activity(activity_id)
            if not activity:
                logger.info("[%s] Activity not ready (%s/%s)", activity_id, attempt, MAX_RETRY)
                time.sleep(RETRY_DELAY)
                continue

            file_id = extract_file_id(activity)
            if file_id:
                break

            logger.info("[%s] MP3 not ready (%s/%s)", activity_id, attempt, MAX_RETRY)
            time.sleep(RETRY_DELAY)

        if not file_id:
            logger.error("[%s] MP3 not found after retries", activity_id)
            return

        logger.info("[%s] FILE_ID: %s", activity_id, file_id)

        # ---------- DOWNLOAD ----------
        audio_path = client.download_audio(file_id)
        if not audio_path:
            logger.error("[%s] Audio download failed", activity_id)
            return

        # ---------- WHISPER ----------
        try:
            segments = whisper_engine.transcribe_segments(audio_path)
            full_text = " ".join(s.get("text", "") for s in segments).strip()
        except Exception:
            logger.exception("[%s] Whisper error", activity_id)
            segments = []
            full_text = ""

        # ---------- DIARIZATION ----------
        diar_segments, _ = diarizer.diarize(audio_path)

        # ---------- ANALYSIS ----------
        script_result = analyze_script_presence(full_text, REQUIRED_PHRASES)
        InterestsPlugin().analyze(full_text, diar_segments)
        informative = is_informational_call(full_text)

        owner_id = activity.get("OWNER_ID")
        owner_type = activity.get("OWNER_TYPE_ID")

        if not owner_id or not owner_type:
            logger.warning("[%s] No CRM owner, skipping comment", activity_id)
            return

        comment_lines = [
            "📞 Автоматический анализ звонка",
            f"Скрипт: {len(script_result.get('found', []))}/{len(REQUIRED_PHRASES)}",
            f"Информативный: {'Да' if informative else 'Нет'}",
            f"Пропущенные фразы: {', '.join(script_result.get('missed', [])) or '—'}"
        ]
        comment_text = "\n".join(comment_lines)

        logger.info("[%s] Sending comment", activity_id)
        client.add_comment(owner_type, owner_id, comment_text)
        logger.info("[%s] Done", activity_id)

    except Exception:
        logger.exception("[%s] Critical error", activity_id)

    finally:
        with LOCK:
            PROCESSING.discard(activity_id)
        if audio_path and os.path.exists(audio_path):
            try:
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
        # Bitrix шлёт form-data
        data = request.form.to_dict()
        logger.info("RAW WEBHOOK: %s", data)

        # --- Проверка токена ---
        token = data.get("auth[application_token]")
        if token != EXPECTED_APP_TOKEN:
            logger.warning("Invalid token: %s", token)
            return jsonify({"status": "forbidden"}), 403

        # --- Получаем ID активности ---
        activity_id = data.get("data[FIELDS][ID]")
        event = data.get("event")

        if not activity_id:
            return jsonify({"status": "ignored"}), 200

        # Нам интересны только события по звонкам
        if event not in ("ONCRMACTIVITYADD", "ONCRMACTIVITYUPDATE"):
            return jsonify({"status": "ignored"}), 200

        logger.info("Activity ID received: %s", activity_id)

        threading.Thread(
            target=process_call,
            args=(int(activity_id),),
            daemon=True
        ).start()

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
