"""Нормализация распознанного текста перед поиском команды.

Порядок обработки:
1. нижний регистр, «ё» → «е», пунктуация → пробелы (числа вида 1.5 и 14:30 сохраняются);
2. русские числительные 0–1000 → цифры («пятьдесят пять» → 55, «половина» → 50);
3. «%» и слова «процент/процента/процентов» после числа удаляются;
4. удаляются имя ассистента (wake word) и слова-паразиты.

Одна и та же нормализация применяется и к фразам пользователя, и к фразам из YAML,
поэтому они сравниваются в одинаковом виде.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

# --- Числительные -------------------------------------------------------------
# Формы даны после замены «ё» → «е». Покрыты основные падежи: «до двадцати»,
# «на пятьдесят», «с тремя», «двухсот».

_UNITS: dict[int, tuple[str, ...]] = {
    0: ("ноль", "нуль", "ноля", "нуля", "нолю", "нулю", "нолем", "нулем", "ноле", "нуле"),
    1: ("один", "одна", "одно", "одного", "одной", "одному", "одним", "одном", "одну", "одною"),
    2: ("два", "две", "двух", "двум", "двумя"),
    3: ("три", "трех", "трем", "тремя"),
    4: ("четыре", "четырех", "четырем", "четырьмя"),
    5: ("пять", "пяти", "пятью"),
    6: ("шесть", "шести", "шестью"),
    7: ("семь", "семи", "семью"),
    8: ("восемь", "восьми", "восемью", "восьмью"),
    9: ("девять", "девяти", "девятью"),
}
_TEENS: dict[int, tuple[str, ...]] = {
    10: ("десять", "десяти", "десятью"),
    11: ("одиннадцать", "одиннадцати", "одиннадцатью"),
    12: ("двенадцать", "двенадцати", "двенадцатью"),
    13: ("тринадцать", "тринадцати", "тринадцатью"),
    14: ("четырнадцать", "четырнадцати", "четырнадцатью"),
    15: ("пятнадцать", "пятнадцати", "пятнадцатью"),
    16: ("шестнадцать", "шестнадцати", "шестнадцатью"),
    17: ("семнадцать", "семнадцати", "семнадцатью"),
    18: ("восемнадцать", "восемнадцати", "восемнадцатью"),
    19: ("девятнадцать", "девятнадцати", "девятнадцатью"),
}
_TENS: dict[int, tuple[str, ...]] = {
    20: ("двадцать", "двадцати", "двадцатью"),
    30: ("тридцать", "тридцати", "тридцатью"),
    40: ("сорок", "сорока"),
    50: ("пятьдесят", "пятидесяти", "пятьюдесятью"),
    60: ("шестьдесят", "шестидесяти", "шестьюдесятью"),
    70: ("семьдесят", "семидесяти", "семьюдесятью"),
    80: ("восемьдесят", "восьмидесяти", "восемьюдесятью", "восьмьюдесятью"),
    90: ("девяносто", "девяноста"),
}
_HUNDREDS: dict[int, tuple[str, ...]] = {
    100: ("сто", "ста"),
    200: ("двести", "двухсот", "двумстам", "двумястами", "двухстах"),
    300: ("триста", "трехсот", "тремстам", "тремястами", "трехстах"),
    400: ("четыреста", "четырехсот", "четыремстам", "четырьмястами", "четырехстах"),
    500: ("пятьсот", "пятисот", "пятистам", "пятьюстами", "пятистах"),
    600: ("шестьсот", "шестисот", "шестистам", "шестьюстами", "шестистах"),
    700: ("семьсот", "семисот", "семистам", "семьюстами", "семистах"),
    800: ("восемьсот", "восьмисот", "восьмистам", "восемьюстами", "восьмьюстами", "восьмистах"),
    900: ("девятьсот", "девятисот", "девятистам", "девятьюстами", "девятистах"),
}
_THOUSAND = frozenset({"тысяча", "тысячи", "тысячу", "тысячей", "тысячею", "тысяче"})
_HALF = frozenset({"половина", "половину", "половины", "половине", "половиной", "наполовину"})
_PERCENT_WORDS = frozenset({"процент", "процента", "процентов", "проценту", "процентам"})


def _invert(table: dict[int, tuple[str, ...]]) -> dict[str, int]:
    """Превращает {число: формы} в {форма: число}."""
    return {form: value for value, forms in table.items() for form in forms}


_UNIT_WORDS = _invert(_UNITS)
_TEEN_WORDS = _invert(_TEENS)
_TEN_WORDS = _invert(_TENS)
_HUNDRED_WORDS = _invert(_HUNDREDS)

# Пунктуация заменяется пробелом. Исключение — разделитель внутри числа или времени
# (1.5, 1,5, 14:30): он сохраняется, а дробная запятая становится точкой.
_PUNCT_RE = re.compile(r"(?P<sep>(?<=\d)[.,:](?=\d))|[^\w\s]|_")
_SPACES_RE = re.compile(r"\s+")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def _replace_punct(match: re.Match[str]) -> str:
    """Возвращает замену для найденного знака препинания."""
    sep = match.group("sep")
    if sep is None:
        return " "
    return ":" if sep == ":" else "."


def _parse_number(tokens: Sequence[str], start: int) -> tuple[int, int] | None:
    """Пытается прочитать число из слов, начиная с позиции start.

    Числительные читаются по разрядам: сотни → десятки (или 10–19) → единицы.
    Если следующее слово нарушает порядок разрядов, начинается новое число:
    «пять десять» → 5 и 10.

    Returns:
        Пара (значение, позиция после числа) или None, если в start нет числительного.
    """
    if tokens[start] in _HALF:
        return 50, start + 1

    value: int | None = None
    pos = start
    if pos < len(tokens) and tokens[pos] in _HUNDRED_WORDS:
        value = _HUNDRED_WORDS[tokens[pos]]
        pos += 1
    if pos < len(tokens) and tokens[pos] in _TEEN_WORDS:
        return (value or 0) + _TEEN_WORDS[tokens[pos]], pos + 1
    if pos < len(tokens) and tokens[pos] in _TEN_WORDS:
        value = (value or 0) + _TEN_WORDS[tokens[pos]]
        pos += 1
    if pos < len(tokens) and tokens[pos] in _UNIT_WORDS:
        unit = _UNIT_WORDS[tokens[pos]]
        # «ноль» допустим только как самостоятельное число, а не «двадцать ноль».
        if unit != 0 or value is None:
            value = (value or 0) + unit
            pos += 1
            # «одна тысяча» → 1000. Большие числа («две тысячи») не поддерживаются.
            if value == 1 and pos < len(tokens) and tokens[pos] in _THOUSAND:
                return 1000, pos + 1
    if value is None:
        return None
    return value, pos


def numerals_to_digits(tokens: Sequence[str]) -> list[str]:
    """Заменяет русские числительные в списке слов на цифры.

    >>> numerals_to_digits("громкость на пятьдесят пять".split())
    ['громкость', 'на', '55']
    """
    result: list[str] = []
    i = 0
    while i < len(tokens):
        if tokens[i] in _THOUSAND:
            # «тысяча» → 1000, но «две тысячи» не поддерживается: после числа слово остаётся.
            after_number = bool(result) and _NUMBER_RE.fullmatch(result[-1]) is not None
            result.append(tokens[i] if after_number else "1000")
            i += 1
            continue
        parsed = _parse_number(tokens, i)
        if parsed is None:
            result.append(tokens[i])
            i += 1
            continue
        value, i = parsed
        result.append(str(value))
    return result


def _strip_percent_words(tokens: Sequence[str]) -> list[str]:
    """Убирает слово «процент(ов)» после числа: «50 процентов» → «50»."""
    result: list[str] = []
    for token in tokens:
        if token in _PERCENT_WORDS and result and _NUMBER_RE.fullmatch(result[-1]):
            continue
        result.append(token)
    return result


def basic_clean(text: str) -> str:
    """Нижний регистр, «ё» → «е», «%» → пробел, пунктуация → пробелы, схлопывание пробелов."""
    text = text.lower().replace("ё", "е").replace("%", " ")
    text = _PUNCT_RE.sub(_replace_punct, text)
    return _SPACES_RE.sub(" ", text).strip()


def _remove_sequences(tokens: list[str], sequences: Iterable[tuple[str, ...]]) -> list[str]:
    """Удаляет из списка слов все вхождения заданных последовательностей слов."""
    ordered = sorted(sequences, key=len, reverse=True)
    result: list[str] = []
    i = 0
    while i < len(tokens):
        for seq in ordered:
            if tuple(tokens[i : i + len(seq)]) == seq:
                i += len(seq)
                break
        else:
            result.append(tokens[i])
            i += 1
    return result


class Normalizer:
    """Приводит фразу к каноническому виду для сравнения с командами.

    Args:
        wake_words: имя ассистента и варианты его написания распознавателем.
        filler_words: слова-паразиты; допускаются словосочетания («будь добр»).
    """

    def __init__(self, wake_words: Iterable[str] = (), filler_words: Iterable[str] = ()) -> None:
        self._removable = {
            tuple(basic_clean(phrase).split())
            for phrase in (*wake_words, *filler_words)
            if basic_clean(phrase)
        }

    def tokens(self, text: str) -> list[str]:
        """Возвращает нормализованную фразу в виде списка слов."""
        tokens = numerals_to_digits(basic_clean(text).split())
        tokens = _strip_percent_words(tokens)
        return _remove_sequences(tokens, self._removable)

    def normalize(self, text: str) -> str:
        """Возвращает нормализованную фразу одной строкой."""
        return " ".join(self.tokens(text))
