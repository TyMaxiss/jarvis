"""Тесты нормализации фраз."""

import pytest

from assistant.nlu.normalize import Normalizer, basic_clean


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Ёлка, ёжик!", "елка ежик"),
        ("  Много   пробелов\t", "много пробелов"),
        ("по-русски", "по русски"),
        ("через 1,5 минуты", "через 1.5 минуты"),
        ("в 14:30", "в 14:30"),
        ("громкость 50%", "громкость 50"),
        ("snake_case", "snake case"),
    ],
)
def test_basic_clean(text: str, expected: str) -> None:
    assert basic_clean(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Джарвис, открой ютуб", "открой ютуб"),
        ("Jarvis, открой ютуб", "открой ютуб"),
        ("открой ютуб, джарвис", "открой ютуб"),
        ("Ну-ка, открой мне ютуб, пожалуйста", "открой ютуб"),
        ("давай сделай громче", "сделай громче"),
        ("будь добр, сделай тише", "сделай тише"),
        ("Эй, Джарвис! Слушай, пауза", "пауза"),
        ("Сделай громкость на пятьдесят пять процентов!", "сделай громкость на 55"),
        ("громкость на половину", "громкость на 50"),
        ("Громкость 30 %", "громкость 30"),
        ("Джарвис", ""),
        ("", ""),
    ],
)
def test_normalize(normalizer: Normalizer, text: str, expected: str) -> None:
    assert normalizer.normalize(text) == expected


def test_words_inside_other_words_are_kept(normalizer: Normalizer) -> None:
    # «ну» и «мне» удаляются только как отдельные слова.
    assert normalizer.normalize("открой нужный менеджер") == "открой нужный менеджер"


def test_custom_wake_word() -> None:
    normalizer = Normalizer(wake_words=["пятница"], filler_words=[])
    assert normalizer.normalize("Пятница, который час?") == "который час"
