"""Ввод через SendInput: сочетания клавиш, печать текста (в том числе кириллицы), мышь."""

from __future__ import annotations

import time
from typing import Annotated, Literal

from pydantic import Field

from assistant.actions import winapi
from assistant.actions.registry import ActionError, ActionRegistry

# Виртуальные коды клавиш Windows по названиям (регистр не важен).
_NAMED_KEYS: dict[str, int] = {
    "ctrl": 0x11, "control": 0x11, "ктрл": 0x11, "lctrl": 0xA2, "rctrl": 0xA3,
    "alt": 0x12, "альт": 0x12, "lalt": 0xA4, "ralt": 0xA5, "altgr": 0xA5,
    "shift": 0x10, "шифт": 0x10, "lshift": 0xA0, "rshift": 0xA1,
    "win": 0x5B, "windows": 0x5B, "вин": 0x5B, "lwin": 0x5B, "rwin": 0x5C,
    "enter": 0x0D, "return": 0x0D, "ввод": 0x0D, "энтер": 0x0D,
    "esc": 0x1B, "escape": 0x1B, "эскейп": 0x1B,
    "space": 0x20, "пробел": 0x20, "tab": 0x09, "таб": 0x09, "backspace": 0x08,
    "delete": 0x2E, "del": 0x2E, "insert": 0x2D, "ins": 0x2D,
    "home": 0x24, "end": 0x23, "pageup": 0x21, "pgup": 0x21, "pagedown": 0x22, "pgdn": 0x22,
    "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
    "влево": 0x25, "вверх": 0x26, "вправо": 0x27, "вниз": 0x28,
    "capslock": 0x14, "numlock": 0x90, "scrolllock": 0x91,
    "printscreen": 0x2C, "prtsc": 0x2C, "pause": 0x13, "apps": 0x5D, "menu": 0x5D,
    "volume_mute": 0xAD, "volume_down": 0xAE, "volume_up": 0xAF,
    "media_next": 0xB0, "media_prev": 0xB1, "media_stop": 0xB2, "media_play_pause": 0xB3,
    "browser_back": 0xA6, "browser_forward": 0xA7, "browser_refresh": 0xA8,
    "browser_home": 0xAC,
    "plus": 0xBB, "minus": 0xBD, "comma": 0xBC, "period": 0xBE,
    ";": 0xBA, "=": 0xBB, ",": 0xBC, "-": 0xBD, ".": 0xBE, "/": 0xBF, "`": 0xC0,
    "[": 0xDB, "\\": 0xDC, "]": 0xDD, "'": 0xDE,
    "multiply": 0x6A, "add": 0x6B, "subtract": 0x6D, "decimal": 0x6E, "divide": 0x6F,
}  # fmt: skip
_NAMED_KEYS.update({f"f{n}": 0x6F + n for n in range(1, 25)})
_NAMED_KEYS.update({f"num{n}": 0x60 + n for n in range(10)})

# Русская раскладка ЙЦУКЕН: буква → та же физическая клавиша в латинской раскладке.
# Так «ctrl+с» (кириллица) работает как Ctrl+C.
_CYRILLIC_KEYS = dict(
    zip("йцукенгшщзхъфывапролджэячсмитьбюё", "qwertyuiop[]asdfghjkl;'zxcvbnm,.`", strict=True)
)

VK_RETURN = 0x0D
VK_TAB = 0x09


def parse_key(name: str) -> int:
    """Возвращает виртуальный код клавиши по названию: "ctrl", "f5", "d", "пробел".

    Raises:
        ActionError: неизвестная клавиша.
    """
    key = name.strip().lower()
    if key in _NAMED_KEYS:
        return _NAMED_KEYS[key]
    if len(key) == 1:
        key = _CYRILLIC_KEYS.get(key, key)
        if "a" <= key <= "z" or "0" <= key <= "9":
            return ord(key.upper())
        if key in _NAMED_KEYS:
            return _NAMED_KEYS[key]
    raise ActionError(f"Неизвестная клавиша «{name}»")


def parse_hotkey(keys: str) -> list[int]:
    """Разбирает сочетание вида "ctrl+shift+esc" в список виртуальных кодов.

    Клавиша «+» записывается словом plus: "ctrl+plus".

    Raises:
        ActionError: пустое сочетание или неизвестная клавиша.
    """
    parts = keys.replace(" ", "").split("+")
    if not keys.strip():
        raise ActionError("Не указано сочетание клавиш")
    if any(not part for part in parts):
        raise ActionError(f"Ошибка в сочетании «{keys}»: клавиша «+» пишется словом plus")
    return [parse_key(part) for part in parts]


def press_hotkey(keys: str) -> None:
    """Нажимает сочетание: клавиши зажимаются по порядку и отпускаются в обратном."""
    codes = parse_hotkey(keys)
    winapi.wait_modifiers_released()
    events = [winapi.key_event(vk, up=False) for vk in codes]
    events += [winapi.key_event(vk, up=True) for vk in reversed(codes)]
    winapi.send_inputs(events)


def tap_key(vk: int) -> None:
    """Нажимает и отпускает одну клавишу."""
    winapi.send_inputs([winapi.key_event(vk, up=False), winapi.key_event(vk, up=True)])


def _utf16_units(char: str) -> list[int]:
    """Кодовые единицы UTF-16 символа (у эмодзи их две — суррогатная пара)."""
    data = char.encode("utf-16-le")
    return [int.from_bytes(data[i : i + 2], "little") for i in range(0, len(data), 2)]


def text_events(text: str) -> list[winapi.INPUT]:
    """События для печати текста: Unicode-ввод, перевод строки — Enter, табуляция — Tab."""
    events: list[winapi.INPUT] = []
    for char in text:
        if char == "\r":
            continue
        if char in "\n\t":
            vk = VK_RETURN if char == "\n" else VK_TAB
            events += [winapi.key_event(vk, up=False), winapi.key_event(vk, up=True)]
            continue
        for unit in _utf16_units(char):
            events += [winapi.unicode_event(unit, up=False), winapi.unicode_event(unit, up=True)]
    return events


def type_unicode(text: str, chunk: int = 64) -> None:
    """Печатает текст в активное окно порциями (некоторые программы теряют быстрый ввод)."""
    winapi.wait_modifiers_released()
    events = text_events(text)
    for start in range(0, len(events), chunk):
        winapi.send_inputs(events[start : start + chunk])
        time.sleep(0.01)


def register(registry: ActionRegistry) -> None:
    """Регистрирует действия клавиатурного ввода."""

    @registry.action(
        "hotkey",
        "Нажать сочетание клавиш в активном окне, например win+d, ctrl+shift+esc, alt+f4, k",
    )
    def hotkey(
        keys: Annotated[str, Field(description="Клавиши через +, например ctrl+c или win+d")],
    ) -> str:
        press_hotkey(keys)
        return f"Нажато {keys}"

    @registry.action("type_text", "Напечатать текст в активном окне (поддерживается кириллица)")
    def type_text(
        text: Annotated[str, Field(min_length=1, max_length=5000, description="Текст для ввода")],
    ) -> str:
        type_unicode(text)
        return "Текст напечатан"

    @registry.action(
        "mouse_click", "Щёлкнуть мышью в точке экрана (координаты в пикселях от левого верхнего угла)"
    )
    def mouse_click(
        x: Annotated[int, Field(ge=0, le=20000, description="Координата X в пикселях")],
        y: Annotated[int, Field(ge=0, le=20000, description="Координата Y в пикселях")],
        button: Annotated[Literal["left", "right", "middle"], Field(description="Кнопка мыши")] = "left",
        double: Annotated[bool, Field(description="Двойной щелчок")] = False,
    ) -> str:
        winapi.wait_modifiers_released()
        if not winapi.api().SetCursorPos(x, y):
            raise ActionError(f"Не удалось переместить курсор в точку {x}, {y}")
        winapi.send_inputs(winapi.mouse_button_events(button, double=double))
        kind = "Двойной щелчок" if double else "Щелчок"
        return f"{kind} ({button}) в точке {x}, {y}"

    @registry.action("scroll", "Прокрутить колесо мыши там, где курсор: плюс — вверх, минус — вниз")
    def scroll(
        amount: Annotated[int, Field(ge=-50, le=50, description="Сколько щелчков колеса; плюс — вверх")],
    ) -> str:
        if amount == 0:
            return "Прокручивать нечего"
        winapi.send_inputs([winapi.wheel_event(amount)])
        return f"Прокрутил {'вверх' if amount > 0 else 'вниз'} на {abs(amount)}"
