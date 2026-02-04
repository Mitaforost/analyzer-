import time
import logging
from typing import List, Dict, Any
import torch
import whisper
import os
import soundfile as sf

logger = logging.getLogger("analyzer")


class WhisperEngine:
    def __init__(self, model_name: str = "small", device: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading Whisper model '{model_name}' on {self.device}")
        # model loading may take long
        self.model = whisper.load_model(model_name, device=self.device)

    def _check_audio(self, wav_path: str) -> bool:
        """Проверка аудиофайла на существование, размер и длину"""
        if not os.path.exists(wav_path):
            logger.warning(f"Audio file not found: {wav_path}")
            return False
        try:
            info = sf.info(wav_path)
            if info.frames == 0:
                logger.warning(f"Audio file is empty: {wav_path}")
                return False
            if info.frames / info.samplerate < 0.5:
                logger.warning(f"Audio too short (<0.5s): {wav_path}")
                return False
        except Exception as e:
            logger.warning(f"Failed to read audio {wav_path}: {e}")
            return False
        return True

    def transcribe(self, wav_path: str) -> (str, float):
        """Возвращает полный текст и время обработки"""
        if not self._check_audio(wav_path):
            return "", 0.0

        start = time.perf_counter()
        try:
            result = self.model.transcribe(wav_path)
            text = result.get("text", "") or ""
            duration = time.perf_counter() - start
            return text.strip(), duration
        except Exception as e:
            logger.exception(f"Whisper transcription failed for {wav_path}: {e}")
            return "", 0.0

    def transcribe_segments(self, wav_path: str) -> List[Dict[str, Any]]:
        """Возвращает список сегментов с безопасной обработкой пустых файлов"""
        if not self._check_audio(wav_path):
            return []

        try:
            result = self.model.transcribe(wav_path, verbose=False)
            segments = result.get("segments", []) or []
            out = []
            for s in segments:
                out.append({
                    "start": float(s.get("start", 0.0)),
                    "end": float(s.get("end", 0.0)),
                    "text": (s.get("text") or "").strip()
                })
            return out
        except Exception as e:
            logger.exception(f"Whisper segmentation failed for {wav_path}: {e}")
            return []
