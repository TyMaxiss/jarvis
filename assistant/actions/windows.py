"""Работа с окнами: активное окно, переключение, свернуть/развернуть/закрыть."""

from __future__ import annotations

import logging
from typing import Annotated, Literal

from pydantic import Field

from assistant.actions import winapi
from assistant.actions.input import press_hotkey
from assistant.actions.registry import ActionError, ActionRegistry
from assistant.nlu.names import name_score

log = logging.getLogger(__name__)


def active_window() -> winapi.WindowInfo:
    """Активное окно; вне Windows — пустое описание (условия when не выполняются)."""
    if not winapi.IS_WINDOWS:
        return winapi.EMPTY_WINDOW
    try:
        return winapi.foreground_window()
    except OSError:
        log.exception("Не удалось получить активное окно")
        return winapi.EMPTY_WINDOW


def window_score(query: str, window: winapi.WindowInfo) -> float:
    """Насколько окно подходит под запрос: по заголовку или по имени процесса."""
    return max(name_score(query, window.title), name_score(query, window.process_stem))


def find_windows(query: str, threshold: float) -> list[tuple[winapi.WindowInfo, float]]:
    """Окна, подходящие под запрос, по убыванию оценки."""
    scored = [(w, window_score(query, w)) for w in winapi.list_windows()]
    matched = [(w, s) for w, s in scored if s >= threshold]
    matched.sort(key=lambda pair: -pair[1])
    return matched


def _describe(window: winapi.WindowInfo) -> str:
    return window.title or window.process_stem or "окно"


def _target_window() -> winapi.WindowInfo:
    """Активное окно, с которым можно работать (не рабочий стол и не панель задач)."""
    window = winapi.foreground_window()
    if not window.hwnd or window.class_name in winapi.SHELL_WINDOW_CLASSES:
        raise ActionError("Нет активного окна")
    return window


def register(registry: ActionRegistry, match_threshold: float = 80.0) -> None:
    """Регистрирует действия с окнами.

    Args:
        registry: реестр действий.
        match_threshold: минимальная оценка при поиске окна по названию.
    """

    @registry.action("get_active_window", "Узнать заголовок и программу активного окна")
    def get_active_window() -> str:
        window = winapi.foreground_window()
        if not window.hwnd:
            return "Активного окна нет"
        return f"Активное окно: «{window.title}» ({window.process})"

    @registry.action(
        "focus_window", "Переключиться на открытое окно по части заголовка или названию программы"
    )
    def focus_window(
        title: Annotated[str, Field(min_length=1, description="Часть заголовка или название программы")],
    ) -> str:
        matches = find_windows(title, match_threshold)
        if not matches:
            raise ActionError(f"Не нашёл окно «{title}»")
        window = matches[0][0]
        if not winapi.activate_window(window.hwnd):
            raise ActionError(f"Windows не дал переключиться на «{_describe(window)}»")
        return f"Переключился на «{_describe(window)}»"

    @registry.action(
        "window_action",
        "Действие с активным окном: свернуть, развернуть, восстановить, закрыть; "
        "minimize_all сворачивает все окна",
    )
    def window_action(
        action: Annotated[
            Literal["minimize", "maximize", "restore", "close", "minimize_all"],
            Field(description="minimize, maximize, restore, close или minimize_all"),
        ],
    ) -> str:
        if action == "minimize_all":
            press_hotkey("win+m")
            return "Все окна свёрнуты"
        window = _target_window()
        user = winapi.api()
        if action == "minimize":
            user.ShowWindow(window.hwnd, winapi.SW_MINIMIZE)
            return "Окно свёрнуто"
        if action == "maximize":
            user.ShowWindow(window.hwnd, winapi.SW_MAXIMIZE)
            return "Окно развёрнуто"
        if action == "restore":
            user.ShowWindow(window.hwnd, winapi.SW_RESTORE)
            return "Окно восстановлено"
        if window.pid in winapi.own_process_tree():
            raise ActionError("Это окно самого ассистента, закрывать его не буду")
        user.PostMessageW(window.hwnd, winapi.WM_CLOSE, 0, 0)
        return f"Закрываю «{_describe(window)}»"
