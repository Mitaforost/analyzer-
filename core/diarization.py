import os
import time
import json
import logging
import torch
from pyannote.audio import Pipeline

logger = logging.getLogger("analyzer")

class Diarizer:
    def __init__(self, hf_token, device=None):
        logger.debug("Initializing Pyannote Pipeline...")
        try:
            # Новое API: token= вместо use_auth_token=
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.0",
                token=hf_token
            )

            # Определение устройства
            if device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            elif isinstance(device, int):
                self.device = torch.device(f"cuda:{device}")
            else:
                self.device = torch.device(device)

            # новое API устройства
            self.pipeline.to(self.device)

            logger.debug(f"Pyannote pipeline loaded on device {self.device}")
        except Exception as e:
            logger.exception("Failed to load Pyannote pipeline.")
            raise e

    def diarize(self, wav_path, save_path=None):
        start_time = time.perf_counter()
        logger.info(f"Running diarization for {os.path.basename(wav_path)}")

        try:
            diarization = self.pipeline(wav_path)

            # Новое API itertracks (возвращает segment, _, label)
            segments = [
                {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "speaker": label
                }
                for segment, _, label in diarization.itertracks(yield_label=True)
            ]

            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(segments, f, ensure_ascii=False, indent=2)
                logger.debug(f"Diarization JSON saved: {save_path}")

            duration = time.perf_counter() - start_time
            logger.debug(f"Diarization took {duration:.2f}s for {wav_path}")

            return segments, duration

        except Exception as e:
            logger.exception(f"Diarization failed for {wav_path}: {e}")
            return [], 0.0
