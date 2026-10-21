"""Экран для нейросети: описание окна, масштаб снимка, выбор элемента по названию.

Настоящий UI Automation проверяется в tests/test_windows.py (на Windows-раннере CI),
здесь — логика на поддельном рабочем столе.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from assistant.actions.apps import AppIndex
from assistant.actions.registry import ActionError, ActionRegistry
from assistant.actions.screen import (
    Element,
    Rect,
    Screen,
    Shot,
    choose_element,
    fit_width,
    register,
    screen_report,
    split_type_words,
)
from assistant.actions.winapi import EMPTY_WINDOW, WindowInfo
from assistant.commands.loader import CommandStore
from assistant.config import AppConfig
from assistant.main import build_registry
from assistant.nlu.matcher import Matcher
from assistant.nlu.normalize import Normalizer
from tests.conftest import NO_WINDOW, ROOT, RecordingSpeaker

NOTEPAD = WindowInfo(42, "Без имени — Блокнот", "Notepad", 7, "notepad.exe")
# Второй монитор слева от основного: координаты экрана отрицательные.
MONITOR = Rect(-2560, 0, 0, 1440)


def element(
    kind: str, name: str, left: int, top: int, width: int = 80, height: int = 24, **extra: Any
) -> Element:
    return Element(kind, name, Rect(left, top, left + width, top + height), **extra)


SAVE = element("Button", "Сохранить", -2000, 100)
SAVE_AS = element("MenuItem", "Сохранить как…", -2400, 60)
CANCEL = element("Button", "Отмена", -1900, 100)
LOGIN = element("Hyperlink", "Войти", -1500, 300)
REMEMBER = element("CheckBox", "Запомнить меня", -1500, 340)
DISABLED = element("Button", "Удалить", -1800, 100, enabled=False)
NAMELESS_EDIT = element("Edit", "", -2500, 200, 1000, 30)
NAMELESS_PANE = element("Pane", "", -2560, 0, 2560, 1440)
ELEMENTS = [NAMELESS_PANE, SAVE_AS, SAVE, CANCEL, DISABLED, NAMELESS_EDIT, LOGIN, REMEMBER]


class FakeDesktop:
    """Окно с элементами и снимок монитора; нажатия и щелчки записываются."""

    def __init__(self, window: WindowInfo = NOTEPAD, elements: list[Element] | None = None) -> None:
        self.window = window
        self.elements = ELEMENTS if elements is None else elements
        self.pressed: list[Element] = []
        self.clicks: list[tuple[int, int, str, bool]] = []

    def target_window(self) -> WindowInfo:
        return self.window

    def look(self, hwnd: int) -> tuple[Rect, Image.Image, list[Element]]:
        assert hwnd == self.window.hwnd
        return MONITOR, Image.new("RGB", (MONITOR.width, MONITOR.height), "white"), list(self.elements)

    def press(self, hwnd: int, choose: Callable[[list[Element]], Element]) -> tuple[Element, str]:
        chosen = choose(list(self.elements))
        self.pressed.append(chosen)
        return chosen, {"CheckBox": "toggle", "TabItem": "select"}.get(chosen.kind, "invoke")

    def click(self, x: int, y: int, button: str, double: bool) -> None:
        self.clicks.append((x, y, button, double))


def test_shot_is_scaled_to_1280_and_maps_back() -> None:
    assert fit_width(2560, 1440) == (1280, 720)
    assert fit_width(1920, 1080) == (1280, 720)
    assert fit_width(1024, 768) == (1024, 768)  # меньше — не увеличивается
    shot = Shot(MONITOR, 1280, 720)
    assert shot.scale == 2.0
    assert shot.to_shot(-2000, 100) == (280, 50)
    assert shot.to_screen(280, 50) == (-2000, 100)
    assert shot.to_screen(*shot.to_shot(-1, 1439)) == (-2, 1438)  # погрешность — не больше масштаба


def test_report_lists_elements_in_shot_coordinates() -> None:
    text = screen_report(NOTEPAD, Shot(MONITOR, 1280, 720), ELEMENTS)
    lines = text.splitlines()
    assert lines[0] == "Окно «Без имени — Блокнот» (notepad.exe)."
    assert "снимка экрана 1280×720 (сам экран 2560×1440)" in lines[1]
    assert "click_ui_element" in lines[1] and "click_screen" in lines[1]
    assert lines[2] == "Элементы окна:"
    # Центр «Сохранить»: (-1960, 112) на экране → (300, 56) на снимке; размер 80×24 → 40×12.
    assert "- Button «Сохранить» (300, 56) 40×12" in lines
    assert "- Button «Удалить» (400, 56) 40×12 (недоступен)" in lines
    assert "- Edit (280, 108) 500×15" in lines  # поле без подписи — показано
    assert not any("Pane" in line for line in lines)  # контейнер без имени — нет


def test_report_without_window_and_with_limit() -> None:
    text = screen_report(EMPTY_WINDOW, Shot(MONITOR, 1280, 720), ELEMENTS, limit=2)
    assert text.startswith("Активного окна нет: виден рабочий стол.")
    assert text.endswith("…и ещё 5 (показаны первые 2).")
    empty = screen_report(NOTEPAD, Shot(MONITOR, 1280, 720), [NAMELESS_PANE])
    assert empty.endswith("Видимых элементов UI Automation в окне нет (программа их не сообщает).")


@pytest.mark.parametrize(
    ("text", "kind", "expected"),
    [
        ("сохранить", "", SAVE),  # точное название важнее «Сохранить как…»
        ("Сохранить как", "", SAVE_AS),
        ("сохранит", "", SAVE),  # опечатка
        ("сохранить", "MenuItem", SAVE_AS),  # тип из describe_screen
        ("войти", "", LOGIN),
        ("ссылке войти", "", LOGIN),  # «кликни по ссылке войти»
        ("кнопку отмена", "", CANCEL),
        ("запомнить", "", REMEMBER),  # начало названия
        ("флажок запомнить меня", "", REMEMBER),
    ],
)
def test_choose_element_by_name(text: str, kind: str, expected: Element) -> None:
    assert choose_element(ELEMENTS, text, kind) == expected


def test_choose_element_explains_failures() -> None:
    with pytest.raises(ActionError, match="Не нашёл «Печать» в окне$"):
        choose_element(ELEMENTS, "Печать")
    with pytest.raises(ActionError, match="Похожие: «Отмена»"):
        choose_element(ELEMENTS, "отменить")
    with pytest.raises(ActionError, match="кнопку «Удалить» — элемент сейчас неактивен"):
        choose_element(ELEMENTS, "удалить")
    twins = [element("Button", "OK", 0, 0), element("Button", "OK", 100, 0)]
    assert choose_element(twins, "ok", number=2) == twins[1]
    with pytest.raises(ActionError, match="Элементов «ok» в окне 2, а нужен 3-й"):
        choose_element(twins, "ok", number=3)


def test_type_words_are_split_off() -> None:
    assert split_type_words("ссылке войти") == ({"Hyperlink"}, "войти")
    assert split_type_words("пункт меню Файл") == ({"MenuItem"}, "Файл")
    assert split_type_words("кнопку") == (set(), "кнопку")  # одно слово — это и есть название
    assert split_type_words("Сохранить") == (set(), "Сохранить")


def test_describe_keeps_image_for_agent_once() -> None:
    screen = Screen(FakeDesktop())
    text = screen.describe()
    assert "Button «Сохранить» (300, 56)" in text
    image = screen.take_image()
    assert image is not None and image.startswith(b"\xff\xd8")  # JPEG
    assert Image.open(io.BytesIO(image)).size == (1280, 720)
    assert screen.take_image() is None  # снимок отдаётся один раз


def test_click_point_maps_shot_to_screen() -> None:
    desktop = FakeDesktop()
    screen = Screen(desktop)
    with pytest.raises(ActionError, match="Сначала опишите экран"):
        screen.click_point(10, 10)
    screen.describe()
    assert screen.click_point(300, 56) == "Щелчок в точке 300, 56 снимка (на экране -1960, 112)"
    assert screen.click_point(300, 56, "right", double=True).startswith("Двойной щелчок")
    assert desktop.clicks == [(-1960, 112, "left", False), (-1960, 112, "right", True)]
    with pytest.raises(ActionError, match="за пределами снимка 1280×720"):
        screen.click_point(1280, 10)


def test_click_element_says_what_was_done() -> None:
    desktop = FakeDesktop()
    screen = Screen(desktop)
    assert screen.click_element("сохранить") == "Нажал кнопку «Сохранить»"
    assert screen.click_element("запомнить меня") == "Переключил флажок «Запомнить меня»"
    assert desktop.pressed == [SAVE, REMEMBER]
    with pytest.raises(ActionError, match="Нет активного окна"):
        Screen(FakeDesktop(EMPTY_WINDOW)).click_element("сохранить")


def test_actions_are_registered() -> None:
    registry = ActionRegistry()
    desktop = FakeDesktop()
    register(registry, Screen(desktop))
    describe, click, point = (
        registry.get(n) for n in ("describe_screen", "click_ui_element", "click_screen")
    )
    assert describe.run(describe.parse_args({})).startswith("Окно «Без имени — Блокнот»")
    assert click.run(click.parse_args({"text": "Войти", "kind": "Hyperlink"})) == "Нажал ссылку «Войти»"
    assert point.run(point.parse_args({"x": 300, "y": 56})).startswith("Щелчок в точке 300, 56")
    assert set(click.parameters_schema()["properties"]) == {"text", "kind", "number"}


@pytest.mark.parametrize(
    ("phrase", "text"),
    [
        ("нажми на кнопку сохранить", "сохранить"),
        ("джарвис, кликни по ссылке Войти", "ссылке Войти"),
        ("щёлкни на сохранить как", "сохранить как"),
    ],
)
def test_click_phrases(phrase: str, text: str, config: AppConfig, tmp_path: Path) -> None:
    registry = build_registry(config, AppIndex(config.apps, scanner=list), RecordingSpeaker())
    store = CommandStore(ROOT / "commands", "my.yaml", registry)
    assert store.reload().issues == []
    matcher = Matcher(Normalizer(config.wake_words, config.matcher.filler_words), config.matcher)
    matcher.set_commands(store.commands)
    result = matcher.match(phrase, NO_WINDOW)
    assert [m.command.id for m in result.matches] == ["click_element"]
    assert result.matches[0].slots == {"text": text}
    # «нажми энтер» — клавиша, а не элемент: такие фразы уходят нейросети.
    assert matcher.match("нажми энтер", NO_WINDOW).status == "not_found"
