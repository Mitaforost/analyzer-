import time
import logging
from typing import List, Dict, Any
import torch
import whisper

logger = logging.getLogger("analyzer")


class WhisperEngine:
    def __init__(self, model_name: str = "small", device: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading Whisper model '{model_name}' on {self.device}")
        # model loading may take long
        self.model = whisper.load_model(model_name, device=self.device)

    def transcribe(self, wav_path: str) -> (str, float):
        """
        Возвращает полный текст и время обработки.
        """
        start = time.perf_counter()
        try:
            result = self.model.transcribe(wav_path)
            text = result.get("text", "") or ""
            duration = time.perf_counter() - start
            return text.strip(), duration
        except Exception:
            logger.exception("Whisper transcription failed for %s", wav_path)
            return "", 0.0

    def transcribe_segments(self, wav_path: str) -> List[Dict[str, Any]]:
        """
        Возвращает список сегментов вида:
        [{'start': float, 'end': float, 'text': str}, ...]
        Если модель поддерживает 'segments' в результате, используем их.
        """
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
        except Exception:
            logger.exception("Whisper segmentation failed for %s", wav_path)
            return []
