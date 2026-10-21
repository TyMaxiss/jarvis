"""Нечёткое сравнение названий приложений и окон.

Пользователь говорит «телеграм», «дискорд», «вижуал студио код», а приложения называются
Telegram, Discord, Visual Studio Code. Поэтому обе строки транслитерируются в латиницу
и сводятся к упрощённому «фонетическому скелету», после чего сравниваются через rapidfuzz.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

from rapidfuzz import fuzz

_CYR_TO_LAT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh",
    "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "ts",
    "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu",
    "я": "ya",
}  # fmt: skip

# Замены применяются по порядку к латинской строке. Они грубо сближают английское
# написание с русским произношением: steam ≈ «стим», discord ≈ «дискорд».
_SKELETON_RULES: tuple[tuple[str, str], ...] = (
    ("you", "yu"),
    ("ph", "f"),
    ("wh", "v"),
    ("ck", "k"),
    ("x", "ks"),
    ("q", "k"),
    ("w", "v"),
    ("j", "dzh"),
    ("ee", "i"),
    ("ea", "i"),
    ("oo", "u"),
    ("c", "k"),
    ("kh", "h"),
    ("y", "i"),
)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_DOUBLE_RE = re.compile(r"([a-z])\1+")
_SILENT_E_RE = re.compile(r"(?<=[a-z]{2})e\b")


def to_latin(text: str) -> str:
    """Транслитерирует кириллицу в латиницу, остальные символы оставляет как есть."""
    return "".join(_CYR_TO_LAT.get(ch, ch) for ch in text.lower())


def skeleton(text: str) -> str:
    """Возвращает упрощённый латинский «скелет» строки для нечёткого сравнения.

    Учитывает и кириллицу, и латиницу: skeleton("Ютуб") == skeleton("YouTube").
    """
    result = to_latin(text)
    # «кх» после транслитерации даёт kh, а «ch» в chrome читается как «к» —
    # такие случаи покрываются алиасами в конфиге, а не правилами.
    result = result.replace("ch", "\0")
    for old, new in _SKELETON_RULES:
        result = result.replace(old, new)
    result = result.replace("\0", "ch")
    result = _NON_ALNUM_RE.sub(" ", result)
    result = _DOUBLE_RE.sub(r"\1", result)
    result = _SILENT_E_RE.sub("", result)
    return " ".join(result.split())


def name_score(query: str, name: str) -> float:
    """Оценивает (0–100), насколько название name подходит под запрос query.

    Сравниваются и строки целиком, и отдельные слова: «хром» находит «Google Chrome»,
    «студио код» — «Visual Studio Code». Совпадение по части слов чуть штрафуется,
    чтобы точное название побеждало длинное («Steam» против «Steamworks Shared»).
    """
    q_plain, n_plain = query.lower().strip(), name.lower().strip()
    if not q_plain or not n_plain:
        return 0.0
    if q_plain == n_plain:
        return 100.0
    q, n = skeleton(query), skeleton(name)
    if not q or not n:
        return 0.0
    full = max(fuzz.ratio(q, n), fuzz.ratio(q_plain, n_plain))

    q_words, n_words = q.split(), n.split()
    # Каждое слово запроса должно хорошо совпасть с каким-то словом названия.
    per_word = [max(fuzz.ratio(qw, nw) for nw in n_words) for qw in q_words]
    partial = min(per_word) * 0.95 if len(n_words) > len(q_words) else 0.0
    return max(full, partial)


def best_matches[T](
    query: str,
    items: Iterable[T],
    key: Callable[[T], str] = str,
    threshold: float = 80.0,
    limit: int = 5,
) -> list[tuple[T, float]]:
    """Возвращает до limit элементов, чьё название подходит под запрос, по убыванию оценки.

    При равной оценке выше стоит более короткое название.
    """
    scored = [(item, name_score(query, key(item))) for item in items]
    scored = [pair for pair in scored if pair[1] >= threshold]
    scored.sort(key=lambda pair: (-pair[1], len(key(pair[0]))))
    return scored[:limit]
