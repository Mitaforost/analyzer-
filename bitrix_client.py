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
    # Типы сущностей для Bitrix (верхний регистр!)
    ENTITY_MAP = {
        1: "LEAD",
        2: "DEAL",
        3: "CONTACT",
        4: "COMPANY",
        5: "ORDER",
    }

    def __init__(self):
        self.webhook_url = BITRIX_WEBHOOK

    # -------------------------------------------------
    # Универсальный вызов API Bitrix
    # -------------------------------------------------
    def call_api(self, method: str, params: dict, post: bool = False) -> dict:
        """
        Универсальный вызов REST API.
        Если post=True, делаем POST, иначе GET.
        """
        url = self.webhook_url + method + ".json"
        try:
            if post:
                resp = requests.post(url, json=params, timeout=30)
            else:
                resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                logger.error("Bitrix API error %s: %s", method, data)
            return data.get("result", {}) or {}
        except Exception:
            logger.exception("Bitrix API call failed: %s %s", method, params)
            return {}

    # -------------------------------------------------
    # Получение CRM активности
    # -------------------------------------------------
    def get_call_activity(self, activity_id: int):
        result = self.call_api(
            "crm.activity.list",
            {
                "filter": {"ID": activity_id},
                "select": ["*", "BINDINGS", "COMMUNICATIONS"]
            },
            post=True  # 🔥 ВАЖНО: ТОЛЬКО POST
        )

        if not result:
            return None

        if isinstance(result, list) and result:
            return result[0]

        return None

    # -------------------------------------------------
    # Получение DOWNLOAD_URL через Disk API
    # -------------------------------------------------
    def get_disk_download_url(self, file_id: str) -> str:
        result = self.call_api("disk.file.get", {"id": file_id})
        return result.get("DOWNLOAD_URL", "")

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
        Добавляет комментарий через crm.timeline.comment.add
        """

        payload = {
            "fields": {
                "ENTITY_TYPE_ID": owner_type_id,  # <-- ВАЖНО
                "ENTITY_ID": owner_id,
                "COMMENT": text
            }
        }

        if files:
            payload["fields"]["FILES"] = files

        return self.call_api(
            "crm.timeline.comment.add",
            payload,
            post=True
        )

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
