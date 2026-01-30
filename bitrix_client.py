import os
import requests
import logging
import configparser
from typing import Dict

# ------------------------
# Конфиг
# ------------------------
cfg = configparser.ConfigParser()
cfg.read("config.ini", encoding="utf-8")

BITRIX_WEBHOOK = cfg.get("BITRIX", "webhook_url")
AUDIO_BASE_URL = cfg.get("BITRIX", "audio_base_url")
TMP_DIR = cfg.get("BITRIX", "tmp_dir", fallback=os.path.join(os.getcwd(), "tmp"))

os.makedirs(TMP_DIR, exist_ok=True)
logger = logging.getLogger("analyzer")

class BitrixClient:
    def __init__(self):
        self.webhook_url = BITRIX_WEBHOOK.rstrip("/") + "/"

    # ------------------------
    # Получение CRM активности по ID
    # ------------------------
    def get_call_activity(self, activity_id: int) -> Dict:
        url = self.webhook_url + "crm.activity.get.json"
        resp = requests.get(url, params={"id": activity_id})
        if resp.status_code != 200:
            logger.error("Failed to get activity %s: %s", activity_id, resp.text)
            return {}
        return resp.json().get("result", {})

    # ------------------------
    # Поиск активности по CALL_ID (Origin ID телефонии)
    # ------------------------
    def find_activity_by_call_id(self, call_id: str) -> Dict:
        url = self.webhook_url + "crm.activity.list.json"
        params = {
            "filter": {"TYPE_ID": 2, "ORIGIN_ID": call_id},
            "select": ["ID","OWNER_ID","OWNER_TYPE_ID","FILES"]
        }
        resp = requests.post(url, json=params)
        if resp.status_code != 200:
            logger.error("Failed to find activity by CALL_ID %s: %s", call_id, resp.text)
            return {}
        items = resp.json().get("result", [])
        return items[0] if items else {}

    # ------------------------
    # Скачивание аудио по fileId
    # ------------------------
    def download_audio(self, file_id: str, file_url: str = None) -> str:
        url = file_url if file_url else f"{AUDIO_BASE_URL}{file_id}"
        local_path = os.path.join(TMP_DIR, f"{file_id}.mp3")
        try:
            resp = requests.get(url, stream=True)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info("Downloaded audio: %s", local_path)
            return local_path
        except requests.HTTPError as e:
            logger.error("Error downloading MP3 %s: %s", file_id, e)
            return ""
        except Exception as e:
            logger.error("Unexpected error downloading MP3 %s: %s", file_id, e)
            return ""

    # ------------------------
    # Добавление комментария в карточку лида/сделки
    # ------------------------
    def add_comment(self, owner_type_id: int, owner_id: int, text: str):
        url = self.webhook_url + "crm.timeline.comment.add.json"
        data = {"fields": {"ENTITY_TYPE_ID": owner_type_id, "ENTITY_ID": owner_id, "COMMENT": text}}
        resp = requests.post(url, json=data)
        if resp.status_code != 200:
            logger.error("Failed to add timeline comment to %s %s: %s", owner_type_id, owner_id, resp.text)
        else:
            logger.info("Added timeline comment to %s %s", owner_type_id, owner_id)

    # ------------------------
    # Обновление пользовательских полей
    # ------------------------
    def update_fields(self, entity_id: int, fields: Dict, entity_type="lead"):
        method = "crm.lead.update.json" if entity_type=="lead" else "crm.deal.update.json"
        url = self.webhook_url + method
        data = {"id": entity_id, "fields": fields}
        resp = requests.post(url, json=data)
        if resp.status_code != 200:
            logger.error("Failed to update %s %s: %s", entity_type, entity_id, resp.text)
        else:
            logger.info("Updated %s %s fields: %s", entity_type, entity_id, fields)
