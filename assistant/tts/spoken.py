"""Текст для голоса Silero: числа и латиница — русскими словами.

Silero читает только кириллицу: цифры, проценты и названия вроде «Telegram» модель
пропустила бы. Piper и edge-tts справляются с ними сами, им эта обработка не нужна.
"""

from __future__ import annotations

import re

_UNITS = ("ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять")
_TEENS = (
    "десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать",
    "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать",
)  # fmt: skip
_TENS = ("", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят",
         "девяносто")  # fmt: skip
_HUNDREDS = ("", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот", "восемьсот",
             "девятьсот")  # fmt: skip
# Разряды: формы для 1, 2–4, 5+ и род (тысяча — женского).
_SCALES = (
    (10**12, ("триллион", "триллиона", "триллионов"), False),
    (10**9, ("миллиард", "миллиарда", "миллиардов"), False),
    (10**6, ("миллион", "миллиона", "миллионов"), False),
    (10**3, ("тысяча", "тысячи", "тысяч"), True),
)
# Число месяца — порядковым в среднем роде: «21 октября» → «двадцать первое октября».
_ORDINAL_UNITS = ("", "первое", "второе", "третье", "четвёртое", "пятое", "шестое", "седьмое", "восьмое",
                  "девятое")  # fmt: skip
_ORDINAL_TENS = {10: "десятое", 20: "двадцатое", 30: "тридцатое"}
_MONTHS = (
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
)  # fmt: skip
_FRACTIONS = {1: ("десятая", "десятых"), 2: ("сотая", "сотых"), 3: ("тысячная", "тысячных")}
# Единицы после чисел: формы для 1, 2–4 (и дробей), 5+.
_UNIT_WORDS = {
    "%": ("процент", "процента", "процентов"),
    "°c": ("градус", "градуса", "градусов"),
    "°": ("градус", "градуса", "градусов"),
    "тб": ("терабайт", "терабайта", "терабайт"),
    "гб": ("гигабайт", "гигабайта", "гигабайт"),
    "мб": ("мегабайт", "мегабайта", "мегабайт"),
    "кб": ("килобайт", "килобайта", "килобайт"),
    "ч": ("час", "часа", "часов"),
    "мин": ("минута", "минуты", "минут"),
    "с": ("секунда", "секунды", "секунд"),
    "сек": ("секунда", "секунды", "секунд"),
    "мс": ("миллисекунда", "миллисекунды", "миллисекунд"),
}
_FEMININE_UNITS = frozenset({"мин", "с", "сек", "мс"})
# Названия, которые по буквам не прочитать: как их говорят по-русски.
_WORDS = {
    "youtube": "ютуб", "google": "гугл", "chrome": "хром", "telegram": "телеграм", "discord": "дискорд",
    "steam": "стим", "spotify": "спотифай", "windows": "виндоус", "microsoft": "майкрософт", "edge": "эдж",
    "word": "ворд", "excel": "эксель", "powerpoint": "пауэрпоинт", "outlook": "аутлук", "teams": "тимс",
    "visual": "вижуал", "studio": "студио", "code": "код", "vscode": "ви эс код", "notepad": "ноутпад",
    "explorer": "эксплорер", "yandex": "яндекс", "whatsapp": "вотсап", "zoom": "зум", "skype": "скайп",
    "twitch": "твич", "github": "гитхаб", "python": "пайтон", "firefox": "файрфокс", "opera": "опера",
    "netflix": "нетфликс", "ok": "окей", "okay": "окей", "wi-fi": "вай-фай", "wifi": "вай-фай",
    "bluetooth": "блютус", "chatgpt": "чат джи пи ти", "deepseek": "дипсик", "claude": "клод",
    "openai": "оупен эй ай", "ollama": "оллама", "jarvis": "джарвис", "whisper": "виспер", "piper": "пайпер",
    "silero": "силеро", "vk": "вэ ка", "email": "имейл", "e-mail": "имейл", "online": "онлайн",
    "offline": "офлайн", "play": "плей", "store": "стор", "office": "офис", "photoshop": "фотошоп",
    "minecraft": "майнкрафт", "obs": "о би эс", "nvidia": "энвидиа", "intel": "интел", "amd": "эй эм ди",
    "apple": "эппл", "android": "андроид", "iphone": "айфон", "telegram's": "телеграма",
}  # fmt: skip
_LETTERS = {
    "a": "эй", "b": "би", "c": "си", "d": "ди", "e": "и", "f": "эф", "g": "джи", "h": "эйч", "i": "ай",
    "j": "джей", "k": "кей", "l": "эл", "m": "эм", "n": "эн", "o": "оу", "p": "пи", "q": "кью", "r": "ар",
    "s": "эс", "t": "ти", "u": "ю", "v": "ви", "w": "дабл ю", "x": "экс", "y": "уай", "z": "зед",
}  # fmt: skip
# Транслитерация остального: сначала сочетания букв, потом буквы.
_COMBOS = (
    ("tion", "шн"), ("sch", "ш"), ("igh", "ай"), ("sh", "ш"), ("ch", "ч"), ("zh", "ж"), ("kh", "х"),
    ("ts", "ц"), ("ph", "ф"), ("th", "т"), ("ck", "к"), ("qu", "кв"), ("ee", "и"), ("oo", "у"), ("ea", "и"),
    ("ou", "ау"), ("ya", "я"), ("yo", "ё"), ("yu", "ю"), ("ye", "е"), ("ai", "ай"), ("ay", "ей"),
    ("ey", "ей"), ("oy", "ой"), ("oi", "ой"), ("au", "о"), ("aw", "о"), ("ew", "ью"), ("ow", "оу"),
    ("x", "кс"), ("j", "дж"), ("w", "в"), ("q", "к"),
)  # fmt: skip
_SINGLE = {
    "a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф", "g": "г", "h": "х", "i": "и", "k": "к",
    "l": "л", "m": "м", "n": "н", "o": "о", "p": "п", "r": "р", "s": "с", "t": "т", "u": "у", "v": "в",
    "y": "и", "z": "з",
}  # fmt: skip

_DATE_RE = re.compile(rf"\b(\d{{1,2}})\s+({'|'.join(_MONTHS)})\b", re.IGNORECASE)
_TIME_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_UNIT_ALT = "|".join(sorted((re.escape(unit) for unit in _UNIT_WORDS), key=len, reverse=True))
_NUMBER_RE = re.compile(
    rf"(?<![\w.,])([-−]?)(\d+)(?:[.,](\d{{1,3}}))?(?![\w.,]\d)(?:\s?({_UNIT_ALT})(?![а-яёa-z]))?",
    re.IGNORECASE,
)
_LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'’+\-]*")
_ALNUM_PARTS_RE = re.compile(r"[A-Za-z]+|\d+")
_SYMBOLS = {"&": " и ", "+": " плюс ", "=": " равно ", "№": " номер ", "…": "...", "—": " - ", "–": " - "}
_DISALLOWED_RE = re.compile(r"[^а-яё .,!?:;\-\n]+", re.IGNORECASE)
_SPACES_RE = re.compile(r"[ \t]+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r" +([.,!?:;])")


def plural(number: int, forms: tuple[str, str, str]) -> str:
    """Форма слова для числа: 1 процент, 2 процента, 5 процентов."""
    tens, ones = number % 100, number % 10
    if ones == 1 and tens != 11:
        return forms[0]
    if 2 <= ones <= 4 and not 12 <= tens <= 14:
        return forms[1]
    return forms[2]


def _triple(number: int, feminine: bool) -> list[str]:
    """Слова для числа 1–999."""
    words = [_HUNDREDS[number // 100]] if number >= 100 else []
    rest = number % 100
    if 10 <= rest < 20:
        return [*words, _TEENS[rest - 10]]
    if rest >= 20:
        words.append(_TENS[rest // 10])
    ones = rest % 10
    if ones:
        words.append(("одна", "две")[ones - 1] if feminine and ones <= 2 else _UNITS[ones])
    return words


def number_words(number: int, feminine: bool = False) -> str:
    """Целое число словами: 2026 → «две тысячи двадцать шесть»; feminine — «одна», «две»."""
    if number < 0:
        return f"минус {number_words(-number, feminine)}"
    if number == 0:
        return _UNITS[0]
    words: list[str] = []
    for scale, forms, scale_feminine in _SCALES:
        count, number = divmod(number, scale)
        if count:
            words += [*_triple(count, scale_feminine), plural(count, forms)]
    if number:
        words += _triple(number, feminine)
    return " ".join(words)


def ordinal_neuter(day: int) -> str:
    """Число месяца порядковым в среднем роде: 21 → «двадцать первое»."""
    if day in _ORDINAL_TENS:
        return _ORDINAL_TENS[day]
    if day < 10:
        return _ORDINAL_UNITS[day]
    if day < 20:
        return _TEENS[day - 10][:-1] + "ое"  # одиннадцать → одиннадцатое
    return f"{_TENS[day // 10]} {_ORDINAL_UNITS[day % 10]}"


def _date(match: re.Match[str]) -> str:
    day = int(match.group(1))
    if not 1 <= day <= 31:
        return match.group(0)
    return f"{ordinal_neuter(day)} {match.group(2)}"


def _time(match: re.Match[str]) -> str:
    hours, minutes = int(match.group(1)), int(match.group(2))
    if minutes == 0:
        return f"{number_words(hours)} ноль ноль"
    spoken_minutes = number_words(minutes) if minutes >= 10 else f"ноль {number_words(minutes)}"
    return f"{number_words(hours)} {spoken_minutes}"


def _fraction(integer: int, fraction: str) -> str:
    """Дробь словами: 2,5 → «две целых пять десятых»."""
    value = int(fraction)
    singular, many = _FRACTIONS[len(fraction)]
    whole = "целая" if plural(integer, ("1", "2", "5")) == "1" else "целых"
    return (
        f"{number_words(integer, feminine=True)} {whole} "
        f"{number_words(value, feminine=True)} {plural(value, (singular, many, many))}"
    )


def _number_only(sign: str, whole: str, fraction: str | None, feminine: bool = False) -> str:
    spoken = _fraction(int(whole), fraction) if fraction else number_words(int(whole), feminine)
    return f"минус {spoken}" if sign else spoken


def _number(match: re.Match[str]) -> str:
    sign, whole, fraction, unit = match.groups()
    unit_key = (unit or "").lower()
    if unit_key == "с" and re.match(r"\s*[а-яё]", match.string[match.end() :], re.IGNORECASE):
        # «с» перед словом — предлог, а не секунды: «в 5 с друзьями».
        return f"{_number_only(sign, whole, fraction)} {unit}"
    forms = _UNIT_WORDS.get(unit_key)
    spoken = _number_only(sign, whole, fraction, feminine=unit_key in _FEMININE_UNITS)
    if forms is None:
        return f" {spoken} "  # пробелы — чтобы не слиплось с соседним словом: «2GIS»
    # После дроби — форма для 2–4: «две целых пять десятых процента».
    return f" {spoken} {forms[1] if fraction else plural(int(whole), forms)} "


def transliterate(word: str) -> str:
    """Латинское слово русскими буквами — приблизительно, как читается по-английски."""
    word = word.lower()
    result: list[str] = []
    index = 0
    while index < len(word):
        if index == 0 and word[0] == "e":
            result.append("э")  # edge → эдж
            index += 1
            continue
        if word[index] == "y" and index == 0 and len(word) > 1 and word[1] in "aeiou":
            result.append("й")
            index += 1
            continue
        if word[index] == "e" and index == len(word) - 1 and len(word) > 3:
            index += 1  # немое e на конце: game → гейм
            continue
        if word[index] == "c" and index + 1 < len(word) and word[index + 1] in "eiy":
            result.append("с")
            index += 1
            continue
        for combo, sound in _COMBOS:
            if word.startswith(combo, index):
                result.append(sound)
                index += len(combo)
                break
        else:
            result.append(_SINGLE.get(word[index], ""))
            index += 1
    return "".join(result)


def _latin_part(part: str) -> str:
    if part.isdigit():
        return number_words(int(part))
    lowered = part.lower()
    if lowered in _WORDS:
        return _WORDS[lowered]
    if len(part) <= 3 or (part.isupper() and len(part) <= 5):
        return " ".join(_LETTERS[letter] for letter in lowered if letter in _LETTERS)  # USB → ю эс би
    return transliterate(part)


def _latin(match: re.Match[str]) -> str:
    token = match.group(0)
    lowered = token.lower().strip("'’")
    if lowered in _WORDS:
        return f" {_WORDS[lowered]} "
    pieces = [piece for piece in re.split(r"[-+'’]", token) if piece]
    return f" {' '.join(_latin_part(part) for piece in pieces for part in _ALNUM_PARTS_RE.findall(piece))} "


def spoken_text(text: str) -> str:
    """Текст, который Silero прочитает целиком: числа, единицы и латиница — словами.

    >>> spoken_text("Громкость 50%")
    'Громкость пятьдесят процентов'
    """
    text = _DATE_RE.sub(_date, text)
    text = _TIME_RE.sub(_time, text)
    text = _NUMBER_RE.sub(_number, text)  # раньше латиницы: «−3°C» — градусы, а не буква C
    text = _LATIN_RE.sub(_latin, text)
    for symbol, words in _SYMBOLS.items():
        text = text.replace(symbol, words)
    text = _DISALLOWED_RE.sub(" ", text)
    lines = (
        _SPACE_BEFORE_PUNCT_RE.sub(r"\1", _SPACES_RE.sub(" ", line)).strip() for line in text.splitlines()
    )
    return "\n".join(line for line in lines if line)
