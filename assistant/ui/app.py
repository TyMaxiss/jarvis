"""Запуск интерфейса: окна, значок в трее, горячие клавиши, наблюдение за командами.

webview.start() занимает главный поток до закрытия всех окон; голос загружается в фоне
уже после того, как окна появились (Whisper загружается несколько секунд, а при первом
запуске модели ещё и скачиваются).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from assistant.actions.registry import ActionError
from assistant.commands.watcher import CommandWatcher
from assistant.ui.api import Api
from assistant.ui.controller import WINDOWS, UiController, VoiceHandle
from assistant.ui.tray import Tray, TrayActions

log = logging.getLogger(__name__)

UPDATE_DELAY = 20.0  # секунд после появления окон до проверки обновлений


class UiOut:
    """Вместо консоли для сообщений запуска голоса.

    Строки «! …» — уведомления в окнах, остальные — что сейчас делается (главная показывает
    это, пока голос загружается); ход скачивания с процентами приходит в progress.
    """

    def __init__(self, controller: UiController) -> None:
        self._controller = controller

    def write(self, text: str) -> int:
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("!"):
                log.warning("%s", line)
                self._controller.bus.emit("toast", {"kind": "error", "text": line.lstrip("! ")})
            elif line:
                log.info("%s", line)
                self._controller.voice_progress(line)
        return len(text)

    def progress(self, text: str, fraction: float | None) -> None:
        """Ход скачивания (assistant.progress.Progress): строка и доля сделанного."""
        self._controller.voice_progress(text, fraction)

    def flush(self) -> None:
        """Буфера нет."""


class UiApp:
    """Окна, трей и всё вокруг них.

    Args:
        controller: что умеет интерфейс.
        voice_starter: запускает голос (может долго грузить модели); None — голос выключен.
        voice_problem: почему голоса нет, если voice_starter не задан.
        debug: инструменты разработчика в окнах (F12).
        dev_url: адрес сервера разработки Vite вместо собранного ui/dist.
        start_hidden: сразу в трей, без главного окна (автозапуск или ui.start_minimized).
    """

    def __init__(
        self,
        controller: UiController,
        *,
        voice_starter: Callable[[Any], VoiceHandle] | None,
        voice_problem: str = "",
        debug: bool = False,
        dev_url: str | None = None,
        start_hidden: bool = False,
    ) -> None:
        self._controller = controller
        self._voice_starter = voice_starter
        self._voice_problem = voice_problem
        self._debug = debug
        self._dev_url = dev_url
        self._start_hidden = start_hidden
        self._tray: Tray | None = None
        self._chat_key: Any = None
        self._command_keys: Any = None
        self._watcher: CommandWatcher | None = None

    def run(self) -> None:
        """Показывает окна и ждёт, пока их закроют (выход из трея или перезапуск)."""
        import webview

        from assistant.ui.windows import Windows, ui_urls, window_icon

        controller = self._controller
        config = controller.config
        # Окно тянется только за саму шапку, а не за её кнопки.
        webview.settings["DRAG_REGION_DIRECT_TARGET_ONLY"] = True
        apis = {name: Api(controller, name) for name in WINDOWS}
        urls = ui_urls(config.ui.theme, config.ui.accent, self._dev_url)
        windows = Windows(
            webview, urls, apis, config.ui.theme, self._start_hidden, overlay_style=config.ui.overlay_style
        )
        controller.host = windows
        windows.on_overlay_stale = controller.overlay.hide_now
        controller.voice_problem = self._voice_problem
        controller.voice_loading = self._voice_starter is not None
        controller.on_chat_key = self._bind_chat_key
        controller.on_commands = self._bind_command_keys
        controller.on_state = self._update_tray
        controller.on_alarm = self._tray_notify
        controller.on_update = self._tray_notify

        self._start_tray()
        self._bind_chat_key(config.ui.quick_chat_key)
        self._bind_command_keys()
        self._watcher = CommandWatcher(controller.commands.directory, controller.reload_commands)
        controller.on_commands_written = self._watcher.acknowledge
        self._watcher.start()
        threading.Thread(target=controller.pump, name="ui-pump", daemon=True).start()
        try:
            webview.start(
                self._started,
                (windows,),
                gui="edgechromium",
                debug=self._debug,
                icon=window_icon(config.ui.accent),
            )
        finally:
            self._shutdown()

    def _started(self, windows: Any) -> None:
        """Окна созданы (отдельный поток): подписки, таймеры, затем загрузка голоса."""
        windows.watch_session_end()
        self._controller.start_alarms()
        # Обновления — чуть позже: сначала окна и голос, а GitHub подождёт.
        updates = threading.Timer(UPDATE_DELAY, self._controller.check_updates_on_start)
        updates.daemon = True
        updates.start()
        if self._voice_starter is not None:
            # Поток-демон: при первом запуске голос может минутами скачивать модели,
            # а выход из программы и «Перезапустить» ждать этого не должны.
            threading.Thread(
                target=self._start_voice, args=(self._voice_starter,), name="voice-start", daemon=True
            ).start()

    def _start_voice(self, starter: Callable[[Any], VoiceHandle]) -> None:
        """Загружает голос и сообщает окнам, получилось ли."""
        controller = self._controller
        try:
            voice = starter(UiOut(controller))
        except Exception as exc:
            log.warning("Голос недоступен: %s", exc, exc_info=not isinstance(exc, RuntimeError))
            controller.voice_failed(f"Голос недоступен: {exc}")
            return
        controller.attach_voice(voice)

    # --- Трей ------------------------------------------------------------------------

    def _start_tray(self) -> None:
        controller = self._controller

        def later(action: Callable[[], Any]) -> Callable[[], None]:
            # Пункты меню выполняются в потоке трея — долгие действия уводим в свой поток.
            return lambda: threading.Thread(target=action, daemon=True).start()

        actions = TrayActions(
            open_main=controller.show_main,
            open_chat=controller.open_chat,
            toggle_pause=controller.toggle_pause,
            is_paused=lambda: controller.voice is not None and controller.voice.loop.paused,
            can_pause=lambda: controller.voice is not None,
            reload_commands=later(controller.reload_commands),
            open_config=lambda: controller.open_path("config"),
            open_log=lambda: controller.open_path("log"),
            quit=controller.quit,
        )
        try:
            self._tray = Tray(self._tray_title(), controller.config.ui.accent, actions)
            self._tray.start()
        except Exception:
            log.exception("Не удалось показать значок в трее")
            self._tray = None

    def _tray_title(self) -> str:
        controller = self._controller
        name = controller.config.general.name.capitalize()
        voice = controller.voice
        if voice is None:
            return f"{name}: голос выключен"
        return f"{name}: на паузе" if voice.loop.paused else f"{name}: слушаю"

    def _update_tray(self, state: dict[str, Any]) -> None:
        if self._tray is not None:
            self._tray.update(self._tray_title())

    # --- Горячие клавиши -------------------------------------------------------------

    def _bind_chat_key(self, combo: str) -> None:
        """Клавиша быстрого чата — через RegisterHotKey: так окно чата может забрать фокус."""
        from assistant.ui.win32 import SystemHotkey

        if self._chat_key is not None:
            self._chat_key.close()
            self._chat_key = None
        if not combo:
            return
        hotkey = SystemHotkey(combo, self._controller.toggle_chat)
        try:
            hotkey.start()
        except ActionError as exc:
            log.warning("Быстрый чат: %s", exc)
            self._notify(f"Быстрый чат: {exc}. Выберите другое сочетание в настройках.")
            return
        self._chat_key = hotkey

    def _bind_command_keys(self) -> None:
        """Горячие клавиши команд из YAML (поле hotkey); перерегистрируются после перезагрузки."""
        from assistant.hotkeys import HotkeyError, HotkeyManager

        controller = self._controller
        if self._command_keys is not None:
            self._command_keys.close()
        self._command_keys = HotkeyManager(controller.hotkeys_paused)
        for command in controller.commands.commands:
            if not command.hotkey:
                continue
            if command.slots:
                log.warning("У команды %s есть слоты — по горячей клавише её не запустить", command.id)
                continue
            try:
                self._command_keys.add_press(command.hotkey, self._command_runner(command))
            except (HotkeyError, ImportError, ValueError) as exc:
                log.warning("Команда %s: %s", command.id, exc)

    def _command_runner(self, command: Any) -> Callable[[], None]:
        controller = self._controller

        def run() -> None:
            threading.Thread(target=controller.run_command, args=(command, "hotkey"), daemon=True).start()

        return run

    def _notify(self, text: str) -> None:
        """Сообщение, когда окон ещё может не быть: в трей и в окна."""
        self._controller.bus.emit("toast", {"kind": "error", "text": text})
        self._tray_notify(text)

    def _tray_notify(self, text: str) -> None:
        if self._tray is not None:
            self._tray.notify(text)

    # --- Выход -----------------------------------------------------------------------

    def _shutdown(self) -> None:
        if self._watcher is not None:
            self._watcher.stop()
        if self._chat_key is not None:
            self._chat_key.close()
        if self._command_keys is not None:
            self._command_keys.close()
        if self._tray is not None:
            self._tray.stop()
        self._controller.stop()
        self._controller.bus.close()
