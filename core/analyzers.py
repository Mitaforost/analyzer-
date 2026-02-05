import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger("analyzer")

# ==========================================================
# SCRIPT PHRASE CHECK
# ==========================================================

def analyze_script_presence(text: str, required_phrases: List[str]) -> Dict[str, List[str]]:
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

    return {"found": found, "missed": missed}


# ==========================================================
# INTERESTS
# ==========================================================

class InterestsPlugin:

    DEFAULT_KEYWORDS = [
        "цена", "срок", "доставка", "гарантия",
        "купить", "стоимость", "скидка"
    ]

    def __init__(self, keywords: List[str] = None):
        self.keywords = keywords or self.DEFAULT_KEYWORDS

    # 👇 ВАЖНО — добавили segments=None
    def analyze(self, text: str, segments: list = None) -> Dict[str, int]:
        text = text or ""
        interests = {}
        for k in self.keywords:
            try:
                cnt = len(re.findall(rf"\b{re.escape(k)}\b", text, flags=re.IGNORECASE))
            except re.error:
                cnt = text.lower().count(k.lower())
            if cnt:
                interests[k] = cnt
        return interests

# ==========================================================
# INFORMATIONAL CALL CHECK
# ==========================================================

def is_informational_call(text: str, min_words: int = 6) -> bool:
    """
    Считаем звонок информативным,
    если в нём >= min_words слов.
    Поддерживает кириллицу.
    """
    if not text:
        return False

    clean = re.sub(r"[^\w\u0400-\u04FF\s]", " ", text)
    words = [w for w in clean.split() if len(w) > 1]

    return len(words) >= min_words


# ==========================================================
# MANAGER PERFORMANCE ANALYZER
# ==========================================================

class ManagerPerformanceAnalyzer:

    POLITE_WORDS = [
        "спасибо", "пожалуйста", "добрый день",
        "здравствуйте", "хорошего дня",
        "извините", "рад помочь"
    ]

    PROMISE_WORDS = [
        "доставка", "скидка", "коммерческое предложение",
        "отправлю", "перезвоню", "подготовим",
        "согласуем", "заключить сделку"
    ]

    def __init__(self, script: dict | None):
        self.script = script or {}

    def _calculate_speaker_times(self, segments: List[Dict[str, Any]]) -> Dict[str, float]:
        speaker_times = {}

        for seg in segments:
            role = seg.get("role") or seg.get("speaker") or "Спикер"
            start = float(seg.get("start", 0))
            end = float(seg.get("end", 0))
            duration = max(0.0, end - start)

            speaker_times[role] = speaker_times.get(role, 0.0) + duration

        return speaker_times

    def analyze(self, full_text: str, segments: List[Dict[str, Any]]) -> Dict[str, Any]:

        full_text = full_text or ""
        text_lower = full_text.lower()

        # === SCRIPT PERCENT ===
        phrases = self.script.get("phrases", [])
        script_check = analyze_script_presence(full_text, phrases)

        total_phrases = max(len(phrases), 1)
        script_percent = int(len(script_check["found"]) / total_phrases * 100)

        # === POLITENESS ===
        polite_count = sum(text_lower.count(w) for w in self.POLITE_WORDS)
        polite_percent = min(int(polite_count * 20), 100)

        # === PROMISES ===
        promises = [w for w in self.PROMISE_WORDS if w in text_lower]

        # === SALES SCORE ===
        sales_score = script_percent

        # === SPEAKER TIMES ===
        speaker_times = self._calculate_speaker_times(segments)

        return {
            "script_percent": script_percent,
            "polite_percent": polite_percent,
            "sales_score": sales_score,
            "promises": promises,
            "speaker_times": speaker_times
        }
