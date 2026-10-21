"""Методы Python, которые интерфейс вызывает через window.pywebview.api (js_api в pywebview).

У каждого окна свой объект Api: так понятно, какое окно загрузилось и куда слать ему
события. pywebview выполняет каждый вызов в новом потоке и показывает интерфейсу все
публичные атрибуты объекта, поэтому здесь только методы, а ссылки — в «_»-атрибутах.
Имена и аргументы совпадают с AssistantApi в ui/src/lib/types.ts.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from assistant.ui.controller import UiController

log = logging.getLogger(__name__)


def _logged[F: Callable[..., Any]](method: F) -> F:
    """Ошибка в методе попадает в лог (интерфейс получит её как отклонённый Promise)."""

    @functools.wraps(method)
    def wrapper(*args: Any) -> Any:
        try:
            return method(*args)
        except Exception:
            log.exception("Ошибка в методе интерфейса %s", method.__name__)
            raise

    return wrapper  # type: ignore[return-value]


class Api:
    """Мост интерфейса к UiController для одного окна.

    Args:
        controller: всё, что умеет интерфейс.
        window: имя окна: main, overlay или chat.
    """

    def __init__(self, controller: UiController, window: str) -> None:
        self._controller = controller
        self._window = window

    @_logged
    def get_state(self) -> dict[str, Any]:
        # Страница окна загрузилась и спрашивает состояние — с этого момента ей шлются события.
        self._controller.window_ready(self._window)
        return self._controller.state()

    @_logged
    def get_settings(self) -> dict[str, Any]:
        return self._controller.settings()

    @_logged
    def save_settings(self, patch: dict[str, Any]) -> dict[str, Any]:
        return self._controller.save_settings(patch)

    @_logged
    def llm_test(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._controller.llm_test(request)

    @_logged
    def llm_models(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._controller.llm_models(request)

    @_logged
    def llm_save(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._controller.llm_save(request)

    @_logged
    def llm_forget_key(self, profile: str) -> dict[str, Any]:
        return self._controller.llm_forget_key(profile)

    @_logged
    def open_key_page(self, profile: str) -> None:
        self._controller.open_key_page(profile)

    @_logged
    def voice_pause(self, paused: bool) -> dict[str, Any]:
        return self._controller.voice_pause(bool(paused))

    @_logged
    def voice_listen(self) -> None:
        self._controller.voice_listen()

    @_logged
    def send_text(self, text: str) -> None:
        self._controller.send_text(str(text))

    @_logged
    def confirm_answer(self, question_id: str, yes: bool) -> None:
        self._controller.confirm_answer(str(question_id), bool(yes))

    @_logged
    def set_volume(self, percent: float) -> dict[str, Any]:
        return self._controller.set_volume(percent)

    @_logged
    def toggle_mute(self) -> dict[str, Any]:
        return self._controller.toggle_mute()

    @_logged
    def run_scenario(self, command_id: str) -> None:
        self._controller.run_scenario(str(command_id))

    @_logged
    def preview_voice(self, tts: dict[str, Any]) -> None:
        self._controller.preview_voice(tts)

    @_logged
    def level_meter(self, on: bool) -> None:
        self._controller.level_meter(bool(on))

    @_logged
    def pause_hotkeys(self, paused: bool) -> None:
        self._controller.pause_hotkeys(bool(paused))

    @_logged
    def window_action(self, window: str, action: str) -> None:
        self._controller.window_action(str(window), str(action))

    @_logged
    def window_resize(self, width: float, height: float, seq: int) -> None:
        self._controller.window_resize(width, height, seq)

    @_logged
    def open_path(self, kind: str) -> None:
        self._controller.open_path(str(kind))

    @_logged
    def reload_commands(self) -> dict[str, Any]:
        return self._controller.reload_commands()

    @_logged
    def restart(self) -> None:
        self._controller.restart()
