import os
import time
import torch
import configparser
from pathlib import Path
from utils.logger import setup_logger
from core.whisper_engine import WhisperEngine
# NOTE: Diarizer импортируем динамически ниже (чтобы поддерживать fallback без pyannote)
from core.reporter import save_transcript, save_summary
from core.analyzers import analyze_script_presence, InterestsPlugin, is_informational_call
from pydub import AudioSegment
import gc
from typing import List, Dict, Any, Optional

# ------------------------
# Загрузка конфигурации
# ------------------------
cfg = configparser.ConfigParser()
cfg.read("config.ini", encoding="utf-8")

BASE_PATH = cfg.get("PATHS", "base_path")
FFMPEG = cfg.get("PATHS", "ffmpeg_bin")
FFPROBE = cfg.get("PATHS", "ffprobe_bin")
INPUT_FOLDER = cfg.get("PATHS", "input_folder")
WAV_FOLDER = cfg.get("PATHS", "wav_folder")
TRANSCRIPTS = cfg.get("PATHS", "transcripts")
OUTPUT_DIARIZATION = cfg.get("PATHS", "output_diarization")
LOG_FILE = cfg.get("PATHS", "log_file")
WHISPER_MODEL = cfg.get("WHISPER", "model")
REQUIRED_PHRASES = [p.strip() for p in cfg.get("SCRIPT", "required_phrases").split(",") if p.strip()]
HF_TOKEN = cfg.get("PYANNOTE", "hf_token", fallback=None)
PYANNOTE_DEVICE = cfg.get("PYANNOTE", "device", fallback="cpu")
POLL_INTERVAL = cfg.getint("GENERAL", "poll_interval", fallback=10)

# ------------------------
# Настройка ffmpeg для pydub
# ------------------------
AudioSegment.converter = FFMPEG
AudioSegment.ffprobe = FFPROBE

# ------------------------
# Логгер
# ------------------------
logger = setup_logger(LOG_FILE)
logger.info("Analyzer starting...")

# ------------------------
# Создание необходимых папок
# ------------------------
for p in [INPUT_FOLDER, WAV_FOLDER, TRANSCRIPTS, OUTPUT_DIARIZATION]:
    Path(p).mkdir(parents=True, exist_ok=True)

# ------------------------
# Инициализация движков
# ------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

whisper_engine = WhisperEngine(model_name=WHISPER_MODEL, device=device)

# Динамический импорт Diarizer — если по какой-то причине pyannote/diarizer отсутствует или некорректен,
# мы подставим fallback-диаризацию (один сегмент на весь файл).
DiarizerClass = None
use_diarization = False
try:
    from core.diarization import Diarizer as _Diarizer
    DiarizerClass = _Diarizer
    # пытаемся инициализировать (может упасть при отсутствии токена/несовместимой версии)
    try:
        diar_device = PYANNOTE_DEVICE if PYANNOTE_DEVICE != "cpu" else "cpu"
        diarizer = DiarizerClass(hf_token=HF_TOKEN, device=diar_device)
        use_diarization = True
        logger.info("Diarizer initialized successfully (pyannote).")
    except Exception as e:
        logger.warning("Diarizer import succeeded but initialization failed — falling back to simple diarization. %s", e)
        diarizer = None
        use_diarization = False
except Exception as e:
    logger.warning("Pyannote Diarizer not available, using fallback diarizer. %s", e)
    diarizer = None
    use_diarization = False

processed_files = set()

# ------------------------
# Fallback Diarizer (простая заглушка, если нет pyannote)
# ------------------------
class SimpleFallbackDiarizer:
    """
    Возвращает один сегмент на весь файл (без разделения спикеров).
    Используется как безопасный fallback, чтобы остальной pipeline работал.
    """
    def __init__(self):
        pass

    def diarize(self, wav_path: str, save_path: Optional[str] = None):
        # пытаемся узнать длительность через pydub (без лишних зависимостей)
        try:
            audio = AudioSegment.from_file(wav_path)
            duration = round(len(audio) / 1000.0, 2)
        except Exception:
            duration = 0.0
        segments = [{"start": 0.0, "end": duration, "speaker": "Speaker_0"}]
        return segments, 0.0

if not use_diarization:
    diarizer = SimpleFallbackDiarizer()
    logger.info("Using SimpleFallbackDiarizer (no speaker separation).")

# ------------------------
# Хелперы
# ------------------------
def convert_to_wav(file_path: str) -> Optional[str]:
    wav_path = os.path.join(WAV_FOLDER, os.path.basename(file_path).rsplit('.', 1)[0] + ".wav")
    if os.path.exists(wav_path):
        return wav_path
    try:
        AudioSegment.from_file(file_path).export(wav_path, format="wav")
        logger.info(f"Converted to WAV: {wav_path}")
        return wav_path
    except Exception as e:
        logger.exception(f"Conversion error for {file_path}: {e}")
        return None

def _overlap(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))

def map_whisper_to_diar(diar_segments: List[Dict[str, Any]], whisper_segments: List[Dict[str, Any]]):
    """
    Для каждого whisper-сегмента находим диар-сегмент с максимальным перекрытием и присваиваем текст.
    """
    if not diar_segments:
        return []

    for ds in diar_segments:
        ds.setdefault("text", "")

    for ws in whisper_segments:
        best = None
        best_overlap = 0.0
        for ds in diar_segments:
            ov = _overlap(ws.get("start", 0), ws.get("end", 0), ds.get("start", 0), ds.get("end", 0))
            if ov > best_overlap:
                best_overlap = ov
                best = ds
        if best and best_overlap > 0.0:
            if best.get("text"):
                best["text"] = best["text"].rstrip() + " " + ws.get("text", "").strip()
            else:
                best["text"] = ws.get("text", "").strip()
    return diar_segments

def detect_autoanswer(segments: List[Dict[str, Any]], whisper_segments: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Ищем автоответчик по характерным фразам в whisper_segments.
    Возвращаем сам whisper-сегмент если найден, иначе None.
    """
    auto_patterns = ["вы позвонили", "оставьте сообщение", "после сигнала", "автоответчик", "сообщение оставьте",
                     "оставьте голосовое", "добро пожаловать", "вы набрали"]
    for ws in whisper_segments:
        txt = (ws.get("text") or "").lower()
        for p in auto_patterns:
            if p in txt:
                return ws
    return None

def assign_roles_by_policy(diar_segments: List[Dict[str, Any]], whisper_segments: List[Dict[str, Any]]):
    """
    Назначаем роли:
    - если найден автоответчик — помечаем как 'Автоответчик'
    - затем два наиболее говорящих — Менеджер и Клиент
    - остальные — Собеседник N
    """
    if not diar_segments:
        return diar_segments

    ws_auto = detect_autoanswer(diar_segments, whisper_segments)
    auto_speaker = None
    if ws_auto:
        best = None
        best_ov = 0.0
        for ds in diar_segments:
            ov = _overlap(ws_auto.get("start", 0), ws_auto.get("end", 0), ds.get("start", 0), ds.get("end", 0))
            if ov > best_ov:
                best_ov = ov
                best = ds
        if best and best_ov > 0.0:
            auto_speaker = best.get("speaker")
            best["role"] = "Автоответчик"
            logger.debug("Autoanswer detected -> speaker %s", auto_speaker)

    speaker_times = {}
    for s in diar_segments:
        sp = s.get("speaker") or s.get("role") or "Спикер"
        speaker_times[sp] = speaker_times.get(sp, 0.0) + max(0.0, (s.get("end", 0) - s.get("start", 0)))

    sorted_sp = sorted(speaker_times.items(), key=lambda x: x[1], reverse=True)

    roles_map = {}
    idx = 0
    for sp, _ in sorted_sp:
        if sp == auto_speaker:
            continue
        if idx == 0:
            roles_map[sp] = "Менеджер"
            idx += 1
        elif idx == 1:
            roles_map[sp] = "Клиент"
            idx += 1
        else:
            break

    other_idx = 1
    for sp, _ in sorted_sp:
        if sp in roles_map or sp == auto_speaker:
            continue
        roles_map[sp] = f"Собеседник {other_idx}"
        other_idx += 1

    for s in diar_segments:
        if s.get("role") == "Автоответчик":
            continue
        sp = s.get("speaker")
        s["role"] = roles_map.get(sp, _normalize_speaker_label(sp))

    return diar_segments

def _normalize_speaker_label(sp):
    if not sp:
        return "Спикер"
    if isinstance(sp, str) and sp.upper().startswith("SPEAKER_"):
        return sp
    return sp

# ------------------------
# Основная обработка одного файла
# ------------------------
def process_file(file_path: str):
    file_name = os.path.basename(file_path).rsplit('.', 1)[0]
    if file_name in processed_files:
        return
    logger.info(f"Processing {file_name}")

    wav_path = convert_to_wav(file_path)
    if not wav_path:
        logger.error("Conversion failed, skipping file.")
        return

    # транскрибация
    try:
        whisper_segments = whisper_engine.transcribe_segments(wav_path) or []
        if whisper_segments:
            full_text = " ".join([s.get("text", "") for s in whisper_segments]).strip()
        else:
            full_text, _ = whisper_engine.transcribe(wav_path)
    except Exception as e:
        logger.exception("Whisper failed: %s", e)
        full_text = ""
        whisper_segments = []

    # диаризация (попытка через диаризатор; может быть SimpleFallbackDiarizer)
    try:
        diar_segments, _ = diarizer.diarize(wav_path)
    except Exception as e:
        logger.exception("Diarization failed, using single-segment fallback: %s", e)
        diar_segments = [{"start": 0.0, "end": 0.0, "speaker": "Unknown", "role": "Спикер", "text": full_text}]

    if not diar_segments:
        diar_segments = [{"start": 0.0, "end": 0.0, "speaker": "Unknown", "role": "Спикер", "text": full_text}]

    # сопоставление текста с сегментами
    diar_segments = map_whisper_to_diar(diar_segments, whisper_segments)

    # назначение ролей
    diar_segments = assign_roles_by_policy(diar_segments, whisper_segments)

    # если тексты пусты — распределяем общий текст по сегментам пропорционально длительности
    if diar_segments and all(not (s.get("text") or "").strip() for s in diar_segments) and full_text:
        total = sum((s.get("end", 0) - s.get("start", 0)) for s in diar_segments) or 1.0
        words = full_text.split()
        idx = 0
        for s in diar_segments:
            dur = (s.get("end", 0) - s.get("start", 0))
            part_ratio = dur / total if total > 0 else 1.0 / len(diar_segments)
            take = max(1, int(len(words) * part_ratio))
            part = " ".join(words[idx: idx + take])
            s["text"] = part
            idx += take
        if idx < len(words):
            diar_segments[-1]["text"] = (diar_segments[-1].get("text", "") + " " + " ".join(words[idx:])).strip()

    # анализ
    script_res = analyze_script_presence(full_text, REQUIRED_PHRASES)
    interests = InterestsPlugin().analyze(full_text, diar_segments)
    metrics = {
        "total_duration": round(sum(max(0.0, (s.get("end", 0) - s.get("start", 0))) for s in diar_segments), 2),
        "speaker_times": {}
    }
    for s in diar_segments:
        role = s.get("role", s.get("speaker", "Спикер"))
        metrics["speaker_times"][role] = metrics["speaker_times"].get(role, 0) + max(0.0, (s.get("end", 0) - s.get("start", 0)))
    metrics["speaker_times"] = {k: round(v, 2) for k, v in metrics["speaker_times"].items()}

    informative = is_informational_call(full_text)
    if not informative:
        logger.info("Call marked as NOT informational (still saving summary & transcript).")

    # сохранение
    try:
        transcript_path = os.path.join(TRANSCRIPTS, f"full_{file_name}.txt")
        summary_path = os.path.join(TRANSCRIPTS, f"summary_{file_name}.txt")
        save_transcript(transcript_path, diar_segments, fallback_text=full_text)
        save_summary(summary_path, full_text, diar_segments, script_res, interests, metrics)
    except Exception as e:
        logger.exception("Failed to save outputs: %s", e)

    # удаляем WAV (MP3 оставляем)
    try:
        if os.path.exists(wav_path):
            os.remove(wav_path)
            logger.info(f"Removed WAV: {wav_path}")
    except Exception as e:
        logger.exception(f"Error removing WAV: {e}")

    # очистка памяти
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

    processed_files.add(file_name)
    logger.info(f"Finished processing {file_name}")

# ------------------------
# Слежение за папкой
# ------------------------
def watch_folder():
    logger.info(f"Watching folder: {INPUT_FOLDER}")
    while True:
        try:
            files = [os.path.join(INPUT_FOLDER, f) for f in os.listdir(INPUT_FOLDER)
                     if f.lower().endswith(".mp3")]
            for f in files:
                try:
                    process_file(f)
                except Exception:
                    logger.exception("Error processing file %s", f)
            time.sleep(POLL_INTERVAL)
        except Exception as e:
            logger.exception(f"Error in watch loop: {e}")
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    watch_folder()
