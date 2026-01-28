import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger("analyzer")


def analyze_script_presence(text: str, required_phrases: List[str]) -> Dict[str, List[str]]:
    """
    Проверяет, какие фразы из скрипта встречаются в тексте.
    Возвращает словарь: {"found": [...], "missed": [...]}
    """
    found, missed = [], []
    text = (text or "").strip()
    for phrase in required_phrases:
        p = (phrase or "").strip()
        if not p:
            continue
        try:
            pattern = re.compile(re.escape(p), flags=re.IGNORECASE)
            if pattern.search(text):
                found.append(p)
            else:
                missed.append(p)
        except re.error:
            if p.lower() in text.lower():
                found.append(p)
            else:
                missed.append(p)
    logger.debug("Script check — found: %s, missed: %s", found, missed)
    return {"found": found, "missed": missed}


class InterestsPlugin:
    """
    Простейший плагин для анализа интересов клиента.
    Возвращает словарь {keyword: count}.
    """
    DEFAULT_KEYWORDS = ["цена", "срок", "доставка", "гарантия", "купить", "стоимость", "скидка"]

    def __init__(self, keywords: List[str] = None):
        self.keywords = keywords or self.DEFAULT_KEYWORDS

    def analyze(self, text: str, segments: List[Dict[str, Any]] = None) -> Dict[str, int]:
        text = (text or "")
        interests = {}
        for k in self.keywords:
            try:
                cnt = len(re.findall(rf"\b{re.escape(k)}\b", text, flags=re.IGNORECASE))
            except re.error:
                cnt = text.lower().count(k.lower())
            if cnt:
                interests[k] = cnt
        logger.debug("Interests detected: %s", interests)
        return interests


def is_informational_call(text: str, min_words: int = 6) -> bool:
    """
    Простая эвристика: считаем звонок информативным, если >= min_words слов.
    Поддерживает кириллицу.
    """
    if not text:
        return False
    clean = re.sub(r"[^\w\u0400-\u04FF\s]", " ", text)
    words = [w for w in clean.split() if len(w) > 1]
    informative = len(words) >= min_words
    logger.debug("is_informational_call: words=%d, min=%d -> %s", len(words), min_words, informative)
    return informative
