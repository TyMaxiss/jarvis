"""Всё, что умеет интерфейс: состояние, настройки, нейросеть, голос, команды, окна.

UiController вызывают окна (через Api), трей и горячие клавиши. Окна он трогает через
WindowHost, поэтому pywebview ему не нужен и его можно проверить тестами.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import webbrowser
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import ValidationError

from assistant import __version__
from assistant.actions.registry import ActionError
from assistant.commands.loader import Command, CommandStore
from assistant.config import AppConfig, LlmConfig, LlmProfile
from assistant.llm.client import LlmError, resolve_api_key
from assistant.llm.providers import (
    CUSTOM_KEY_ENV,
    CUSTOM_PROFILE,
    build_profile,
    check_connection,
    list_models,
    provider_info,
    with_profile,
)
from assistant.ui.activity import Activity
from assistant.ui.confirm import UiConfirmer
from assistant.ui.events import EventBus
from assistant.ui.overlay import OverlayController
from assistant.ui.settings import (
    Choices,
    SettingsError,
    SettingsStore,
    check_key,
    needs_restart,
    settings_view,
    ui_view,
)

if TYPE_CHECKING:
    from assistant.actions.executor import ActionExecutor
    from assistant.llm.agent import Agent
    from assistant.pipeline import Pipeline
    from assistant.tts.base import SpeechQueue

log = logging.getLogger(__name__)

WINDOWS = ("main", "overlay", "chat")
MAIN_MIN = (880, 600)  # наименьший размер главного окна
PREVIEW_TEXT = "Привет! Я {name}. Так звучит мой голос."


class WindowHost(Protocol):
    """Окна, которыми управляет контроллер (pywebview; в тестах — подделка)."""

    def show(self, name: str) -> None: ...

    def hide(self, name: str) -> None: ...

    def visible(self, name: str) -> bool: ...

    def minimize(self) -> None: ...

    def toggle_maximize(self) -> None: ...

    def resize_main(self, width: int, height: int) -> None: ...

    def sender(self, name: str) -> Callable[[str], None]: ...

    def quit(self) -> None: ...


class VoiceHandle(Protocol):
    """Запущенный голос (VoiceRuntime из main.py)."""

    loop: Any  # VoiceLoop
    capture: Any  # MicrophoneCapture: level
    confirmer: Any  # VoiceConfirmer или None
    wake_word: bool


def _first_error(exc: ValidationError) -> str:
    """Понятный текст первой ошибки pydantic."""
    message = str(exc.errors()[0].get("msg", exc))
    return message.removeprefix("Value error, ")


def open_in_shell(path: Path) -> None:
    """Открывает файл или папку программой по умолчанию."""
    if sys.platform == "win32":
        os.startfile(path)  # type: ignore[attr-defined]
    else:
        log.info("Открыть %s можно только в Windows", path)


class UiController:
    """Всё, что умеет интерфейс: состояние, настройки, нейросеть, голос, окна.

    Args:
        config: текущий конфиг (обновляется при сохранении настроек).
        store: чтение и запись config.yaml и .env.
        pipeline: конвейер фраз.
        executor: исполнитель действий (dry-run меняется на лету).
        commands: команды из YAML.
        make_agent: создаёт агента нейросети, когда её включают в настройках.
        speech: озвучка или None.
        volume: громкость системы (VolumeController).
        bus: события окнам.
        activity: текущая и последние команды.
        confirmer: подтверждение кнопками и голосом.
        log_file: файл лога.
        journal_file: журнал действий.
        force_dry_run: dry-run включён аргументом --dry-run и из настроек не выключается.
        hotkeys_paused: пока установлено, глобальные сочетания не срабатывают.
    """

    def __init__(
        self,
        *,
        config: AppConfig,
        store: SettingsStore,
        pipeline: Pipeline,
        executor: ActionExecutor,
        commands: CommandStore,
        make_agent: Callable[[LlmConfig], Agent],
        speech: SpeechQueue | None,
        volume: Any,
        bus: EventBus,
        activity: Activity,
        confirmer: UiConfirmer,
        log_file: Path,
        journal_file: Path,
        force_dry_run: bool = False,
        hotkeys_paused: threading.Event | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.pipeline = pipeline
        self.executor = executor
        self.commands = commands
        self._make_agent = make_agent
        self.speech = speech
        self._volume = volume
        self.bus = bus
        self.activity = activity
        self.confirmer = confirmer
        self._log_file = log_file
        self._journal_file = journal_file
        self._force_dry_run = force_dry_run
        self.hotkeys_paused = hotkeys_paused or threading.Event()
        self.host: WindowHost | None = None
        self.voice: VoiceHandle | None = None
        self.voice_loading = False  # голос ещё загружается (Whisper — несколько секунд)
        self.voice_problem = ""
        self._voice_state = "off"
        self._paused = False
        self._meter = False
        self._resize_seq = 0
        self._resize_lock = threading.Lock()
        self._stopped = threading.Event()
        self.restart_requested = False
        self.restart_pending = False
        self.startup_issues: list[str] = []  # команды, пропущенные при запуске
        # Что ещё сделать при изменениях (их подставляет UiApp).
        self.on_state: Callable[[dict[str, Any]], None] | None = None
        self.on_chat_key: Callable[[str], None] | None = None
        self.on_commands: Callable[[], None] | None = None
        self.overlay = OverlayController(
            show=lambda: self._host_call("show", "overlay"),
            hide=lambda: self._host_call("hide", "overlay"),
            enabled=lambda: self.config.ui.overlay,
        )

    # --- Состояние -------------------------------------------------------------------

    def window_ready(self, name: str) -> None:
        """Страница окна загрузилась: с этого момента ей шлются события."""
        if self.host is None or name not in WINDOWS:
            return
        first = not self.bus.attached(name)
        self.bus.attach(name, self.host.sender(name))
        if first and name == "main" and self.startup_issues:
            self.bus.emit(
                "commands", {"count": len(self.commands.commands), "issues": self.startup_issues}, to=["main"]
            )

    def state(self) -> dict[str, Any]:
        """Состояние для окон (тип AppState в ui/src/lib/types.ts)."""
        config, voice = self.config, self.voice
        return {
            "name": config.general.name.capitalize(),
            "version": __version__,
            "voice": {
                "enabled": voice is not None,
                "state": self._voice_state if voice is not None else "off",
                "paused": voice is not None and voice.loop.paused,
                "wake_word": voice is not None and voice.wake_word,
                "push_to_talk_key": config.voice.push_to_talk_key if config.voice.push_to_talk else "",
                "stop_key": config.voice.stop_key,
                "loading": self.voice_loading,
                "problem": self.voice_problem,
            },
            "llm": self._llm_summary(),
            "dry_run": self.executor.dry_run,
            "volume": self._volume_state(),
            "recent": self.activity.recent,
            "scenarios": self.scenarios(),
            "ui": ui_view(config),
        }

    def push_state(self) -> None:
        """Рассылает окнам состояние целиком."""
        state = self.state()
        self.bus.emit("state", state)
        if self.on_state is not None:
            self.on_state(state)

    def _llm_summary(self) -> dict[str, Any]:
        llm = self.config.llm
        profile = llm.active
        problem = ""
        try:
            resolve_api_key(llm.profile, profile)
        except LlmError:
            problem = "Нужен ключ API — вставьте его в настройках"
        return {
            "enabled": llm.enabled and self.pipeline.agent is not None,
            "profile": llm.profile,
            "title": profile.title or llm.profile,
            "model": profile.model,
            "ready": not problem,
            "problem": problem,
        }

    def _volume_state(self) -> dict[str, Any]:
        try:
            return {"percent": self._volume.get(), "muted": self._volume.is_muted()}
        except Exception:  # не Windows или нет устройства вывода
            return {"percent": None, "muted": None}

    def scenarios(self) -> list[dict[str, Any]]:
        """Команды из нескольких шагов без слотов — их можно запустить кнопкой на главной."""
        items = []
        for command in self.commands.commands:
            if len(command.steps) < 2 or command.slots or not command.phrases or "{" in command.phrases[0]:
                continue
            phrase = command.phrases[0]
            items.append(
                {
                    "id": command.id,
                    "title": phrase[:1].upper() + phrase[1:],
                    "phrase": phrase,
                    "steps": len(command.steps),
                }
            )
        return items

    # --- Голос -----------------------------------------------------------------------

    def attach_voice(self, voice: VoiceHandle) -> None:
        """Голос запущен (он загружается в фоне после появления окон)."""
        self.voice = voice
        self.voice_loading = False
        self.voice_problem = ""
        self._voice_state = "idle"
        self.confirmer.voice = voice.confirmer
        self.push_state()

    def voice_failed(self, problem: str) -> None:
        """Голос запустить не удалось: окна покажут почему."""
        self.voice_loading = False
        self.voice_problem = problem
        self.push_state()

    def on_voice_state(self, state: str) -> None:
        """Голосовой цикл сменил состояние (VoiceStatus)."""
        self._voice_state = state
        self.bus.emit("voice", {"state": state, "detail": ""})
        self.overlay.voice_state(state)
        paused = self.voice is not None and self.voice.loop.paused
        if paused != self._paused:
            self._paused = paused
            self.push_state()

    def voice_pause(self, paused: bool) -> dict[str, Any]:
        if self.voice is not None:
            self.voice.loop.set_paused(paused)
        return self.state()

    def toggle_pause(self) -> None:
        """Пауза из трея."""
        if self.voice is not None:
            self.voice.loop.set_paused(not self.voice.loop.paused)

    def voice_listen(self) -> None:
        if self.voice is not None:
            self.voice.loop.push_to_talk()

    def level_meter(self, on: bool) -> None:
        self._meter = on

    def pump_levels(self, interval: float = 0.07) -> None:
        """Шлёт уровень микрофона индикатору в настройках и волне в оверлее (в своём потоке)."""
        while not self._stopped.wait(interval):
            voice = self.voice
            if voice is None:
                continue
            targets = [
                name for name, on in (("main", self._meter), ("overlay", self.overlay.listening)) if on
            ]
            if targets:
                self.bus.emit("level", {"value": round(float(voice.capture.level), 3)}, to=targets)

    # --- Команды ---------------------------------------------------------------------

    def send_text(self, text: str) -> None:
        """Команда текстом из быстрого чата: выполняется в фоне, ход — событиями."""
        phrase = " ".join(text.split())[:1000]
        if phrase:
            self._background(self._run_text, phrase, name="ui-text")

    def _run_text(self, phrase: str) -> None:
        self.activity.begin(phrase, "text")
        try:
            result = self.pipeline.handle_text(phrase, "text", self.confirmer.for_source("text"))
        except Exception:
            log.exception("Ошибка при выполнении «%s»", phrase)
            self.activity.failed(phrase, "text", "Ошибка — подробности в логе")
            return
        self.activity.finish(result, "text")

    def run_scenario(self, command_id: str) -> None:
        """Кнопка сценария на главной."""
        command = next((c for c in self.commands.commands if c.id == command_id), None)
        if command is not None:
            self._background(self.run_command, command, "text", name="ui-scenario")

    def run_command(self, command: Command, source: str) -> None:
        """Выполняет команду без фразы (сценарий или горячая клавиша команды)."""
        phrase = command.phrases[0] if command.phrases else command.id
        self.activity.begin(phrase, source)
        try:
            result = self.pipeline.run_command(command, source, self.confirmer.for_source(source))
        except Exception:
            log.exception("Ошибка в команде %s", command.id)
            self.activity.failed(phrase, source, "Ошибка — подробности в логе")
            return
        self.activity.finish(result, source)

    def confirm_answer(self, question_id: str, yes: bool) -> None:
        self.confirmer.answer(question_id, yes)

    def reload_commands(self) -> dict[str, Any]:
        """Перечитывает команды (кнопка, трей или изменение файлов)."""
        result = self.commands.reload()
        self.pipeline.reload_commands(self.commands.commands)
        issues = [str(issue) for issue in result.issues]
        if self.on_commands is not None:
            self.on_commands()
        self.bus.emit("commands", {"count": len(result.commands), "issues": issues})
        self.push_state()
        return {"count": len(result.commands), "issues": issues}

    # --- Громкость -------------------------------------------------------------------

    def set_volume(self, percent: Any) -> dict[str, Any]:
        try:
            self._volume.set(round(float(percent)))
        except (ActionError, TypeError, ValueError) as exc:
            log.info("Громкость не изменена: %s", exc)
        return self.state()

    def toggle_mute(self) -> dict[str, Any]:
        try:
            self._volume.set_muted(not self._volume.is_muted())
        except ActionError as exc:
            log.info("Звук не переключён: %s", exc)
        state = self.state()
        self.bus.emit("state", state)
        return state

    # --- Настройки -------------------------------------------------------------------

    def settings(self) -> dict[str, Any]:
        """Настройки для страницы «Настройки»."""
        return {
            **settings_view(self.config, self.store, self._choices()),
            "restart_pending": self.restart_pending,
        }

    def _choices(self) -> Choices:
        choices = Choices()
        try:
            from assistant.audio.capture import list_input_devices

            choices.microphones = list_input_devices()
        except Exception:
            log.debug("Список микрофонов недоступен", exc_info=True)
        try:
            from assistant.tts.sapi_tts import voice_names

            choices.sapi_voices = [(name, name) for name in voice_names()]
        except Exception:
            log.debug("Список голосов Windows недоступен", exc_info=True)
        return choices

    def save_settings(self, patch: Any) -> dict[str, Any]:
        """Сохраняет изменения со страницы настроек."""
        if not isinstance(patch, Mapping) or not all(isinstance(v, Mapping) for v in patch.values()):
            return self._save_result(False, ["Неверный формат настроек"])
        try:
            config = self.store.update(patch)
        except SettingsError as exc:
            return self._save_result(False, exc.errors)
        restart = needs_restart(patch)
        self.restart_pending = self.restart_pending or restart
        self._apply(config)
        return self._save_result(True, [], restart)

    def _save_result(self, ok: bool, errors: list[str], restart: bool = False) -> dict[str, Any]:
        return {"ok": ok, "errors": errors, "restart": restart, "settings": self.settings()}

    def _apply(self, config: AppConfig) -> None:
        """Применяет то, что меняется без перезапуска."""
        previous, self.config = self.config, config
        self.executor.dry_run = self._force_dry_run or config.general.dry_run
        if self.speech is not None:
            self.speech.volume = config.tts.volume
        self._apply_llm(config.llm)
        if previous.ui.quick_chat_key != config.ui.quick_chat_key and self.on_chat_key is not None:
            self.on_chat_key(config.ui.quick_chat_key)
        if not config.ui.overlay:
            self.overlay.hide_now()
        self.push_state()

    def _apply_llm(self, llm: LlmConfig) -> None:
        agent = self.pipeline.agent
        if not llm.enabled:
            if agent is not None:
                self.pipeline.set_agent(None)
        elif agent is None:
            self.pipeline.set_agent(self._make_agent(llm))
        elif agent.settings != llm:
            agent.reconfigure(llm)

    # --- Нейросеть -------------------------------------------------------------------

    def _llm_request(self, request: Mapping[str, Any]) -> tuple[str, LlmProfile, LlmConfig, str]:
        """Профиль из настроек: имя, профиль, раздел llm с ним и вставленный ключ.

        Raises:
            SettingsError: нет профиля, модели, адреса или ключ не похож на ключ.
        """
        name = str(request.get("profile") or "")
        llm = self.config.llm
        if name not in llm.profiles and name != CUSTOM_PROFILE:
            raise SettingsError([f"Нет профиля нейросети «{name}»"])
        model = str(request.get("model") or "").strip()
        if not model:
            raise SettingsError(["Укажите модель"])
        key = str(request.get("key") or "").strip()
        if key:
            key = check_key(key)
        # Свой сервис без ключа работает без него; с ключом — берёт его из CUSTOM_LLM_API_KEY.
        custom_key = (
            CUSTOM_KEY_ENV if name == CUSTOM_PROFILE and (key or self.store.key(CUSTOM_KEY_ENV)) else ""
        )
        base_url = str(request.get("base_url") or "").strip()
        try:
            profile = build_profile(llm, name, model, base_url, key_env=custom_key)
        except ValidationError as exc:
            raise SettingsError([_first_error(exc)]) from exc
        return name, profile, with_profile(llm, name, profile), key

    def llm_test(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Проверка связи с профилем, ещё не сохранённым (ключ — из поля, если вставлен)."""
        try:
            _, _, settings, key = self._llm_request(request)
        except SettingsError as exc:
            return {"ok": False, "text": "", "ms": 0, "error": "; ".join(exc.errors)}
        result = check_connection(settings, api_key=key or None, timeout=self.config.llm.request_timeout)
        return {"ok": result.ok, "text": result.text, "ms": round(result.ms), "error": result.error}

    def llm_models(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Модели, доступные по ключу."""
        try:
            _, _, settings, key = self._llm_request({**request, "model": request.get("model") or "-"})
            models = list_models(settings, api_key=key or None)
        except SettingsError as exc:
            return {"ok": False, "models": [], "error": "; ".join(exc.errors)}
        except LlmError as exc:
            return {"ok": False, "models": [], "error": str(exc)}
        return {"ok": True, "models": models, "error": ""}

    def llm_save(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Сохраняет ключ в .env, профиль в config.yaml и сразу переключает агента."""
        try:
            name, profile, _, key = self._llm_request(request)
            if key:
                self.store.save_key(profile.api_key_env or CUSTOM_KEY_ENV, key)
            fields: dict[str, Any] = {"model": profile.model}
            if provider_info(name, profile).base_url_editable:
                fields["base_url"] = profile.base_url
            if name == CUSTOM_PROFILE:
                fields.update(title=profile.title, api_key_env=profile.api_key_env)
            config = self.store.update_llm(name, fields)
        except SettingsError as exc:
            return self._save_result(False, exc.errors)
        log.info("Нейросеть: профиль %s, модель %s", name, profile.model)
        agent = self.pipeline.agent
        unchanged = agent is not None and agent.settings == config.llm
        self._apply(config)
        if key and unchanged and agent is not None:
            agent.reconfigure(config.llm)  # профиль тот же, а ключ новый — клиент нужен новый
        return self._save_result(True, [])

    def llm_forget_key(self, name: str) -> dict[str, Any]:
        """Удаляет сохранённый ключ профиля."""
        llm = self.config.llm
        env = llm.profiles[name].api_key_env if name in llm.profiles else ""
        if not env and name == CUSTOM_PROFILE:
            env = CUSTOM_KEY_ENV
        if not env:
            return self._save_result(False, ["У этого профиля нет ключа"])
        self.store.forget_key(env)
        agent = self.pipeline.agent
        if agent is not None and name == llm.profile:
            agent.reconfigure(llm)  # клиент с прежним ключом больше не нужен
        self.push_state()
        return self._save_result(True, [])

    def open_key_page(self, name: str) -> None:
        """Страница провайдера, где создают ключ (адрес — из списка, а не из окна)."""
        url = provider_info(name, self.config.llm.profiles.get(name)).key_url
        if url:
            webbrowser.open(url)

    # --- Озвучка ---------------------------------------------------------------------

    def preview_voice(self, tts: Mapping[str, Any]) -> None:
        """«Прослушать»: фраза выбранным голосом с выбранной скоростью и громкостью."""
        self._background(self._preview, dict(tts), name="voice-preview")

    def _preview(self, tts: dict[str, Any]) -> None:
        if self.speech is None:
            self.bus.emit("toast", {"kind": "error", "text": "Озвучка выключена или недоступна"})
            return
        try:
            rate = float(tts.get("rate") or 1.0)
            volume = float(tts.get("volume") or 1.0)
            if tts.get("engine") == "sapi":
                from assistant.tts.sapi_tts import SapiTTS

                engine: Any = SapiTTS(str(tts.get("sapi_voice") or ""), rate)
            else:
                from assistant.tts.piper_tts import PiperTTS

                path = self.config.resolve(str(tts.get("piper_voice") or ""))
                if path.suffix != ".onnx" or not path.exists():
                    raise ValueError(f"нет файла голоса {path.name}")
                engine = PiperTTS(path, rate)
            audio, samplerate = engine.synthesize(
                PREVIEW_TEXT.format(name=self.config.general.name.capitalize())
            )
        except Exception as exc:
            log.warning("Не удалось озвучить пример: %s", exc)
            self.bus.emit("toast", {"kind": "error", "text": f"Не удалось озвучить: {exc}"})
            return
        # Очередь сама умножит на свою громкость — пример звучит с выбранной.
        self.speech.play_sound(audio * (volume / max(self.speech.volume, 0.01)), samplerate)

    # --- Окна ------------------------------------------------------------------------

    def pause_hotkeys(self, paused: bool) -> None:
        if paused:
            self.hotkeys_paused.set()
        else:
            self.hotkeys_paused.clear()

    def window_action(self, window: str, action: str) -> None:
        if self.host is None or window not in WINDOWS:
            return
        if window == "main" and action == "minimize":
            self.host.minimize()
        elif window == "main" and action == "maximize":
            self.host.toggle_maximize()
        elif action in ("close", "hide"):
            self.host.hide(window)

    def window_resize(self, width: Any, height: Any, seq: Any) -> None:
        """Размер главного окна от уголка изменения размера; устаревшие запросы пропускаются."""
        if self.host is None:
            return
        with self._resize_lock:
            if int(seq) <= self._resize_seq:
                return
            self._resize_seq = int(seq)
            self.host.resize_main(
                max(MAIN_MIN[0], round(float(width))), max(MAIN_MIN[1], round(float(height)))
            )

    def show_main(self) -> None:
        self._host_call("show", "main")

    def toggle_chat(self) -> None:
        """Горячая клавиша быстрого чата: показать (с фокусом на поле ввода) или спрятать."""
        if self.host is None or self.hotkeys_paused.is_set():
            return
        if self.host.visible("chat"):
            self.host.hide("chat")
        else:
            self.host.show("chat")
            self.bus.emit("chat_shown", {}, to=["chat"])

    def open_chat(self) -> None:
        if self.host is not None and not self.host.visible("chat"):
            self.toggle_chat()

    def open_path(self, kind: str) -> None:
        """Открывает config.yaml, лог, журнал или папку команд."""
        paths = {
            "config": self.store.config_path,
            "log": self._log_file,
            "journal": self._journal_file,
            "commands": self.commands.directory,
        }
        path = paths.get(kind)
        if path is None:
            return
        if not path.exists():
            self.bus.emit("toast", {"kind": "info", "text": f"Файла {path.name} пока нет"})
            return
        try:
            open_in_shell(path)
        except OSError as exc:
            self.bus.emit("toast", {"kind": "error", "text": f"Не удалось открыть {path.name}: {exc}"})

    def restart(self) -> None:
        """Перезапуск: окна закрываются, main.py запускает программу заново."""
        self.restart_requested = True
        self.quit()

    def quit(self) -> None:
        """Выход: окна закрываются, webview.start() в main.py возвращается."""
        self.stop()
        self._host_call("quit")

    def stop(self) -> None:
        """Останавливает фоновые потоки контроллера (окна уже закрыты)."""
        self._stopped.set()

    def _host_call(self, method: str, *args: Any) -> None:
        if self.host is not None:
            getattr(self.host, method)(*args)

    @staticmethod
    def _background(target: Callable[..., None], *args: Any, name: str) -> None:
        threading.Thread(target=target, args=args, name=name, daemon=True).start()
