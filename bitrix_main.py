import os
import gc
import time
import logging
import configparser
import tempfile
import threading
from datetime import datetime
import glob
import json

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

os.makedirs(TMP_DIR, exist_ok=True)

# ==================================================
# LOGGING
# ==================================================
logger = logging.getLogger("bitrix_main")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
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

# 🔒 ВАЖНО: защита Whisper от параллельного доступа
WHISPER_LOCK = threading.Lock()

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
RETRY_DELAY = 20
PROCESSING = set()
LOCK = threading.Lock()

# ==================================================
# LOAD SALES SCRIPTS
# ==================================================
SCRIPTS = []
for fpath in glob.glob("scripts/*.json"):
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            SCRIPTS.append(json.load(f))
    except Exception:
        logger.exception("Failed to load script %s", fpath)

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
    for key in ("FILE_ID", "RECORD_ID", "CALL_RECORD_ID", "CALL_RECORD_FILE_ID"):
        val = settings.get(key)
        if val:
            return str(val)

    return None


def choose_best_script(full_text: str):
    best_score = 0.0
    best_script = None
    best_result = None

    for script in SCRIPTS:
        result = analyze_script_presence(full_text, script.get("phrases", [])) \
                 or {"found": [], "missed": []}

        score = len(result["found"]) / max(len(script.get("phrases", [])), 1)

        if score > best_score:
            best_score = score
            best_script = script
            best_result = result

    return best_script, best_result


def download_call_audio(activity: dict, file_id: str) -> str | None:
    if not file_id:
        logger.info(f"[{activity.get('ID')}] Нет FILE_ID")
        return None

    audio_path = client.download_audio(file_id)

    if not audio_path:
        logger.info(f"[{activity.get('ID')}] Ошибка скачивания FILE_ID {file_id}")

    return audio_path


# ==================================================
# PROCESS CALL
# ==================================================
def process_call(activity_id: int):

    # защита от повторной обработки
    with LOCK:
        if activity_id in PROCESSING:
            logger.info(f"[{activity_id}] Already processing")
            return
        PROCESSING.add(activity_id)

    start_time = datetime.now()
    audio_path = None

    try:
        activity = None
        file_id = None
        call_type = "Не определён"

        # ==============================
        # Ждём появления записи звонка
        # ==============================
        for attempt in range(1, MAX_RETRY + 1):

            activity = client.get_call_activity(activity_id)

            if not activity:
                time.sleep(RETRY_DELAY)
                continue

            if str(activity.get("TYPE_ID")) != "2":
                logger.info(f"[{activity_id}] Not a call activity")
                return

            direction = str(activity.get("DIRECTION"))
            call_type = "Входящий" if direction == "1" else "Исходящий"

            duration = int(activity.get("DURATION") or 0)
            minutes = duration // 60
            seconds = duration % 60

            logger.info(
                f"[{activity_id}] {call_type} | {minutes}м {seconds:02d}с"
            )

            file_id = extract_file_id(activity)

            if file_id:
                logger.info(f"[{activity_id}] Найден FILE_ID: {file_id}")
                break

            time.sleep(RETRY_DELAY)

        if not file_id:
            logger.warning(f"[{activity_id}] Файл так и не появился")
            return

        # ==============================
        # Скачиваем аудио
        # ==============================
        audio_path = download_call_audio(activity, file_id)
        if not audio_path:
            return

        if os.path.getsize(audio_path) < 5000:
            logger.info(f"[{activity_id}] Файл слишком маленький")
            return

        # ==============================
        # WHISPER (с LOCK!)
        # ==============================
        try:
            with WHISPER_LOCK:
                segments = whisper_engine.transcribe_segments(audio_path)

            full_text = " ".join(
                s.get("text", "") for s in segments
            ).strip()

        except Exception as e:
            logger.warning(f"[{activity_id}] Whisper failed: {e}")
            segments = []
            full_text = ""

        # ==============================
        # АНАЛИЗ
        # ==============================
        diar_segments, _ = diarizer.diarize(audio_path)

        best_script, script_result = choose_best_script(full_text)

        if not script_result:
            script_result = {"found": [], "missed": []}

        script_name = best_script.get("name") if best_script else "Не определено"

        interests = InterestsPlugin().analyze(full_text, diar_segments) or {}
        informative = is_informational_call(full_text)

        # ==============================
        # ПОИСК CRM-СУЩНОСТИ (НАДЁЖНЫЙ)
        # ==============================
        owner_id = None
        owner_type = None

        # 1️⃣ BINDINGS
        bindings = activity.get("BINDINGS") or []
        for b in bindings:
            if b.get("OWNER_ID") and b.get("OWNER_TYPE_ID"):
                owner_id = b["OWNER_ID"]
                owner_type = b["OWNER_TYPE_ID"]
                break

        # 2️⃣ OWNER_ID / OWNER_TYPE_ID
        if not owner_id:
            if activity.get("OWNER_ID") and activity.get("OWNER_TYPE_ID"):
                owner_id = activity["OWNER_ID"]
                owner_type = activity["OWNER_TYPE_ID"]

        # 3️⃣ SETTINGS
        if not owner_id:
            settings = activity.get("SETTINGS") or {}
            if settings.get("CRM_ENTITY_ID") and settings.get("CRM_ENTITY_TYPE"):
                owner_id = settings["CRM_ENTITY_ID"]
                owner_type = settings["CRM_ENTITY_TYPE"]

        # 4️⃣ COMMUNICATIONS
        if not owner_id:
            communications = activity.get("COMMUNICATIONS") or []
            for c in communications:
                if c.get("ENTITY_ID") and c.get("ENTITY_TYPE_ID"):
                    owner_id = c["ENTITY_ID"]
                    owner_type = c["ENTITY_TYPE_ID"]
                    break

        if not owner_id or not owner_type:
            logger.warning(
                f"[{activity_id}] Не удалось определить CRM-сущность для комментария"
            )
            return

        owner_type = int(owner_type)

        logger.info(
            f"[{activity_id}] Добавляем комментарий в "
            f"OWNER_TYPE_ID={owner_type} OWNER_ID={owner_id}"
        )

        # ==============================
        # Формируем комментарий
        # ==============================
        total_phrases = len(script_result["found"]) + len(script_result["missed"])

        comment_text = "\n".join([
            f"Тип звонка: {call_type}",
            f"Скрипт: {script_name}",
            f"Скрипт выполнен: {len(script_result['found'])}/{total_phrases}",
            f"Информативный: {'Да' if informative else 'Нет'}",
            f"Пропущенные фразы: "
            f"{', '.join(script_result['missed']) or '—'}",
            f"Интересы клиента: "
            f"{', '.join(f'{k}({v})' for k, v in interests.items()) or 'не выявлены'}"
        ])

        result = client.add_comment(owner_type, owner_id, comment_text)

        logger.info(f"[{activity_id}] Ответ Bitrix: {result}")

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"[{activity_id}] ✔ Анализ завершён за {elapsed:.1f} сек")

    except Exception as e:
        logger.exception(f"[{activity_id}] Critical error: {e}")

    finally:
        with LOCK:
            PROCESSING.discard(activity_id)

        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception:
                pass

        gc.collect()


# ==================================================
# WEBHOOK
# ==================================================
@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        data = request.form.to_dict()

        token = data.get("auth[application_token]")
        if token != EXPECTED_APP_TOKEN:
            return jsonify({"status": "forbidden"}), 403

        activity_id = data.get("data[FIELDS][ID]")
        event = data.get("event")

        if not activity_id or event not in (
                "ONCRMACTIVITYADD",
                "ONCRMACTIVITYUPDATE"
        ):
            return jsonify({"status": "ignored"}), 200

        threading.Thread(
            target=process_call,
            args=(int(activity_id),),
            daemon=True
        ).start()

        return jsonify({"status": "ok"}), 200

    except Exception:
        return jsonify({"status": "error"}), 500


# ==================================================
# RUN SERVER
# ==================================================
if __name__ == "__main__":
    logger.info("Webhook server started on 0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000)
