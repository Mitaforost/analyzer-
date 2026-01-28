import os
import logging
from datetime import datetime
from typing import List, Dict, Any

logger = logging.getLogger("analyzer")


def _normalize_role(role: str) -> str:
    if not role:
        return "Спикер"
    r = role.strip()
    lower = r.lower()
    if lower in ("менеджер", "клиент", "автоответчик", "робот", "автоответчик/робот"):
        return r
    if r.upper().startswith("SPEAKER_"):
        return "Спикер"
    if "auto" in lower or "ответ" in lower or "answer" in lower:
        return "Автоответчик"
    return r


def _merge_consecutive_segments_by_role(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not segments:
        return []

    merged = []
    for seg in sorted(segments, key=lambda s: s.get("start", 0.0)):
        role = _normalize_role(seg.get("role") or seg.get("speaker") or "Спикер")
        text = (seg.get("text") or "").strip()
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))

        if not merged:
            merged.append({"role": role, "text": text, "start": start, "end": end})
            continue

        last = merged[-1]
        gap = start - last["end"]
        if role == last["role"] and gap <= 1.0:
            last["text"] = (last["text"] + " " + text).strip() if last["text"] else text
            last["end"] = max(last["end"], end)
        else:
            merged.append({"role": role, "text": text, "start": start, "end": end})
    return merged


def save_transcript(path: str, segments: List[Dict[str, Any]], fallback_text: str = None) -> None:
    try:
        merged = _merge_consecutive_segments_by_role(segments)
        lines = []
        for m in merged:
            text = (m.get("text") or "").strip()
            if not text:
                continue
            role = m.get("role", "Спикер")
            text = " ".join(text.split())
            lines.append(f"{role}: {text}")

        if not lines:
            if fallback_text:
                lines = [" ".join(fallback_text.split())]
            else:
                collected = [s.get("text", "").strip() for s in segments if s.get("text")]
                lines = [" ".join(collected).strip()] if collected else [""]

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.debug("Saved transcript: %s", path)
    except Exception as e:
        logger.exception("Error saving transcript %s: %s", path, e)


def save_summary(path: str,
                 full_text: str,
                 segments: List[Dict[str, Any]],
                 script_res: Dict[str, List[str]],
                 interests: Dict[str, int],
                 metrics: Dict[str, Any]) -> None:
    try:
        lines = [f"Дата отчёта: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]
        total = metrics.get("total_duration", 0)
        lines.append(f"Общая длительность: {total} сек")
        lines.append("Время по ролям:")
        for role, t in metrics.get("speaker_times", {}).items():
            lines.append(f"{role}: {t} сек")

        found = script_res.get("found", [])
        missed = script_res.get("missed", [])
        lines.append(f"Найденные ключевые фразы: {', '.join(found) if found else '-'}")
        lines.append(f"Пропущенные ключевые фразы: {', '.join(missed) if missed else '-'}")

        interests_str = ", ".join([f"{k}({v})" for k, v in interests.items()]) if interests else "-"
        lines.append(f"Интересы клиента: {interests_str}")

        snippet = (full_text or "").replace("\n", " ").strip()
        if snippet:
            snippet = snippet[:397].rstrip() + "..." if len(snippet) > 400 else snippet
            lines.append("")
            lines.append("Краткий фрагмент разговора:")
            lines.append(snippet)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.debug("Saved summary: %s", path)
    except Exception as e:
        logger.exception("Error saving summary %s: %s", path, e)
