"""Тесты конвертера русских числительных в цифры."""

import pytest

from assistant.nlu.normalize import numerals_to_digits

_UNITS = ["", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
_TEENS = ["десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать",
          "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать"]  # fmt: skip
_TENS = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят",
         "восемьдесят", "девяносто"]  # fmt: skip
_HUNDREDS = ["", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот",
             "восемьсот", "девятьсот"]  # fmt: skip


def spell(number: int) -> str:
    """Число 0–1000 прописью (именительный падеж) — эталон для проверки."""
    if number == 0:
        return "ноль"
    if number == 1000:
        return "тысяча"
    hundreds, rest = divmod(number, 100)
    words = [_HUNDREDS[hundreds]]
    if 10 <= rest < 20:
        words.append(_TEENS[rest - 10])
    else:
        tens, units = divmod(rest, 10)
        words += [_TENS[tens], _UNITS[units]]
    return " ".join(word for word in words if word)


@pytest.mark.parametrize("number", range(0, 1001))
def test_every_number_from_0_to_1000(number: int) -> None:
    assert numerals_to_digits(spell(number).split()) == [str(number)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("пятьдесят пять", "55"),
        ("сто двадцать", "120"),
        ("двести пять", "205"),
        ("половина", "50"),
        ("половину", "50"),
        ("наполовину", "50"),
        ("одна тысяча", "1000"),
        ("одну", "1"),
        ("две", "2"),
    ],
)
def test_spec_examples(text: str, expected: str) -> None:
    assert " ".join(numerals_to_digits(text.split())) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("убавь до двадцати", "убавь до 20"),
        ("с тремя окнами", "с 3 окнами"),
        ("до пятидесяти пяти", "до 55"),
        ("около двухсот", "около 200"),
        ("девяноста девяти", "99"),
        ("сорока двух", "42"),
        ("четырехсот", "400"),
    ],
)
def test_oblique_cases(text: str, expected: str) -> None:
    assert " ".join(numerals_to_digits(text.split())) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("пять десять", "5 10"),  # разряды не по порядку — два числа
        ("двадцать ноль", "20 0"),  # «ноль» не прибавляется к десяткам
        ("две тысячи", "2 тысячи"),  # больше 1000 не поддерживается
        ("сто сто", "100 100"),
        ("громкость на пятьдесят", "громкость на 50"),
        ("открой ютуб", "открой ютуб"),
        ("громкость 50", "громкость 50"),
    ],
)
def test_sequences_and_text(text: str, expected: str) -> None:
    assert " ".join(numerals_to_digits(text.split())) == expected
