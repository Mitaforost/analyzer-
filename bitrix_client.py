import os
import requests
import logging
import configparser
from typing import Dict, List, Optional
import base64

# ========================
# CONFIG
# ========================
cfg = configparser.ConfigParser()
cfg.read("config.ini", encoding="utf-8")

BITRIX_WEBHOOK = cfg.get("BITRIX", "webhook_url").rstrip("/") + "/"
TMP_DIR = cfg.get("BITRIX", "tmp_dir", fallback=os.path.join(os.getcwd(), "tmp"))

os.makedirs(TMP_DIR, exist_ok=True)

logger = logging.getLogger("bitrix_client")


class BitrixClient:
    ENTITY_MAP = {
        1: "lead",
        2: "deal",
        3: "contact",
        4: "company",
        5: "order",
        # dynamic объекты можно добавить по необходимости
    }

    def __init__(self):
        self.webhook_url = BITRIX_WEBHOOK

    # -------------------------------------------------
    # Получение CRM активности
    # -------------------------------------------------
    def get_call_activity(self, activity_id: int) -> Dict:
        url = self.webhook_url + "crm.activity.get.json"
        try:
            resp = requests.get(url, params={"id": activity_id}, timeout=30)
            resp.raise_for_status()
            return resp.json().get("result", {}) or {}
        except Exception:
            logger.exception("crm.activity.get failed %s", activity_id)
            return {}

    # -------------------------------------------------
    # Получение DOWNLOAD_URL через Disk API
    # -------------------------------------------------
    def get_disk_download_url(self, file_id: str) -> str:
        url = self.webhook_url + "disk.file.get.json"
        try:
            resp = requests.get(url, params={"id": file_id}, timeout=30)
            resp.raise_for_status()
            result = resp.json().get("result", {}) or {}
            return result.get("DOWNLOAD_URL", "")
        except Exception:
            logger.exception("disk.file.get failed %s", file_id)
            return ""

    # -------------------------------------------------
    # Скачивание аудио
    # -------------------------------------------------
    def download_audio(self, file_id: str) -> str:
        local_path = os.path.join(TMP_DIR, f"{file_id}.mp3")
        try:
            download_url = self.get_disk_download_url(file_id)
            if not download_url:
                logger.error("No DOWNLOAD_URL for file %s", file_id)
                return ""

            with requests.get(download_url, stream=True, timeout=60) as r:
                r.raise_for_status()
                content_type = r.headers.get("Content-Type", "")
                if "audio" not in content_type and "octet-stream" not in content_type:
                    logger.error("Invalid content-type for %s: %s", file_id, content_type)
                    return ""

                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(8192):
                        if chunk:
                            f.write(chunk)

            if os.path.getsize(local_path) < 1024:
                logger.error("Downloaded file too small: %s", local_path)
                return ""

            logger.info("Audio downloaded: %s", local_path)
            return local_path

        except Exception:
            logger.exception("Error downloading audio %s", file_id)
            return ""

    # -------------------------------------------------
    # Добавление комментария в таймлайн
    # -------------------------------------------------
    def add_comment(
        self,
        owner_type_id: int,
        owner_id: int,
        text: str,
        files: Optional[List[List[str]]] = None
    ):
        """
        Добавляет комментарий к сущности (Lead/Deal/Contact и т.д.) через crm.timeline.comment.add.
        files: список [ [название файла, base64 содержимое], ...]
        """
        url = self.webhook_url + "crm.timeline.comment.add.json"
        entity_type = self.ENTITY_MAP.get(owner_type_id, "lead")

        payload = {
            "fields": {
                "ENTITY_TYPE": entity_type,
                "ENTITY_ID": owner_id,
                "COMMENT": text,
                "FILES": files or []
            }
        }

        try:
            resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code != 200:
                logger.error("Failed to add comment %s %s: %s", entity_type, owner_id, resp.text)
            else:
                logger.info("Timeline comment added to %s %s", entity_type, owner_id)
                logger.debug("Bitrix response: %s", resp.json())
        except Exception:
            logger.exception("Exception adding comment to %s %s", entity_type, owner_id)

    # -------------------------------------------------
    # Утилита для прикрепления локального файла в base64
    # -------------------------------------------------
    @staticmethod
    def encode_file_base64(file_path: str) -> str:
        try:
            with open(file_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            logger.exception("Failed to encode file %s", file_path)
            return ""
