"""Точка входа: сборка компонентов и запуск.

Это единственное место, где создаются и связываются объекты приложения.
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

from assistant.actions import apps, browser, files, media, power, shell, system, timers, volume, windows
from assistant.actions import input as input_actions
from assistant.actions.apps import AppIndex, looks_like_path
from assistant.actions.executor import ActionExecutor
from assistant.actions.registry import ActionRegistry
from assistant.audio.sounds import SOUND_RATE, alarm_chime
from assistant.commands.loader import CommandStore, LoadResult
from assistant.commands.watcher import CommandWatcher
from assistant.config import ENV_FILE, TIMERS_FILE, AppConfig, ConfigError, LlmConfig, load_config
from assistant.console import ConsoleApp, ConsoleSpeaker
from assistant.journal import ActionJournal
from assistant.llm.agent import Agent
from assistant.llm.client import LlmError, resolve_api_key
from assistant.logging_setup import SecretMasker, setup_logging
from assistant.nlu.matcher import Matcher
from assistant.nlu.normalize import Normalizer
from assistant.pipeline import Pipeline
from assistant.safety.confirm import NO_PHRASES, YES_PHRASES, Confirmer, ConsoleConfirmer, VoiceConfirmer
from assistant.safety.policy import SafetyPolicy
from assistant.timers import Alarm, Alarms
from assistant.tts.base import SilentSpeech, Speaker, SpeechQueue, TeeSpeaker, TTSEngine, TTSError

if TYPE_CHECKING:
    from assistant.audio.capture import MicrophoneCapture
    from assistant.hotkeys import HotkeyManager
    from assistant.voice import StatusSink, VoiceLoop

log = logging.getLogger("assistant")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разбирает аргументы командной строки."""
    parser = argparse.ArgumentParser(prog="python -m assistant", description="Голосовой ассистент")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="путь к config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="только логировать действия")
    parser.add_argument("--debug", action="store_true", help="подробный лог и вывод в консоль")
    parser.add_argument("--text", action="store_true", help="только ввод текстом, без микрофона и озвучки")
    parser.add_argument("--console", action="store_true", help="консольный режим вместо окон")
    parser.add_argument("--download-models", action="store_true", help="скачать модели Vosk, Piper, Whisper")
    parser.add_argument("--list-devices", action="store_true", help="показать микрофоны и выйти")
    return parser.parse_args(argv)


def build_registry(
    config: AppConfig, index: AppIndex, speaker: Speaker, alarms: Alarms | None = None
) -> ActionRegistry:
    """Создаёт реестр и регистрирует все действия (alarms — таймеры; None — только в памяти)."""
    registry = ActionRegistry()
    apps.register(registry, index, config.apps)
    windows.register(registry, config.apps.match_threshold)
    input_actions.register(registry)
    media.register(registry)
    volume.register(registry)
    browser.register(registry, config.browser)
    system.register(registry, speaker)
    power.register(registry)
    files.register(registry)
    shell.register(registry)
    timers.register(registry, alarms or Alarms(None))
    return registry


# --- Голос --------------------------------------------------------------------------


class VoiceSetupError(RuntimeError):
    """Голос запустить не удалось; ассистент продолжит работать с вводом текста."""


class VoiceRuntime:
    """Запущенные голосовые компоненты — чтобы управлять ими и остановить при выходе."""

    def __init__(
        self,
        capture: MicrophoneCapture,
        loop: VoiceLoop,
        hotkeys: HotkeyManager,
        confirmer: VoiceConfirmer | None = None,
        wake_word: bool = False,
    ) -> None:
        self.capture = capture
        self.loop = loop
        self.hotkeys = hotkeys
        self.confirmer = confirmer
        self.wake_word = wake_word  # активация по имени работает

    def close(self) -> None:
        self.loop.close()
        self.hotkeys.close()
        self.capture.stop()


def create_speech(config: AppConfig, out: TextIO) -> SpeechQueue | None:
    """Озвучка: Piper, при ошибке — голоса Windows (SAPI); None — озвучки нет."""
    from assistant.tts.piper_tts import PiperTTS
    from assistant.tts.player import SoundDevicePlayer
    from assistant.tts.sapi_tts import SapiTTS

    sapi = SapiTTS(config.tts.sapi_voice, config.tts.rate) if sys.platform == "win32" else None
    primary: TTSEngine | None = sapi
    if config.tts.engine == "piper":
        try:
            primary = PiperTTS(config.resolve(config.tts.piper_voice), config.tts.rate)
        except Exception as exc:
            if not isinstance(exc, TTSError):
                log.exception("Не удалось загрузить голос Piper")
            out.write(f"! {exc}\n")
            if sapi is not None:
                out.write("  Пока буду говорить голосом Windows (SAPI).\n")
    if primary is None:
        out.write("! Озвучка недоступна — ответы будут только в консоли.\n")
        return None
    fallback = sapi if primary is not sapi else None
    return SpeechQueue(primary, SoundDevicePlayer(), fallback, config.tts.volume)


def whisper_prompt(config: AppConfig, store: CommandStore, index: AppIndex) -> Callable[[], str]:
    """Функция, собирающая подсказку для Whisper из команд и найденных программ."""
    from assistant.stt.whisper_stt import build_prompt

    def build() -> str:
        names = [n for n in config.apps.aliases.values() if not looks_like_path(n)]
        names += sorted((e.name for e in index.entries(timeout=0)), key=len)
        phrases = (phrase for command in store.commands for phrase in command.phrases)
        return build_prompt(config.general.name, phrases, names, config.stt.prompt_app_names)

    return build


def start_voice(
    config: AppConfig,
    pipeline: Pipeline,
    normalizer: Normalizer,
    status: StatusSink,
    speech: SpeechQueue | None,
    store: CommandStore,
    index: AppIndex,
    out: Any,
    *,
    handle_text: Callable[[str, Confirmer | None], Any] | None = None,
    hotkeys_paused: threading.Event | None = None,
) -> VoiceRuntime:
    """Запускает микрофон, распознавание и голосовой цикл.

    Args:
        status: куда сообщать о состоянии голоса (консоль или окна).
        out: куда писать ход запуска (консоль или окна).
        handle_text: обработка фразы; по умолчанию — конвейер с голосовым подтверждением.
        hotkeys_paused: пока установлено, горячие клавиши не срабатывают.

    Raises:
        VoiceSetupError: микрофона нет, нечем активировать или не загрузилась модель.
    """
    from assistant.audio.capture import CaptureError, MicrophoneCapture
    from assistant.audio.sounds import SOUND_RATE, listen_chime
    from assistant.audio.vad import SegmenterSettings, SileroVAD, SpeechSegmenter
    from assistant.audio.wakeword import VoskModelError, load_vosk_model, phrase_spotter, wake_spotter
    from assistant.hotkeys import HotkeyError, HotkeyManager
    from assistant.stt.whisper_stt import WhisperSTT
    from assistant.voice import STOP_PHRASES, STOP_SOUNDALIKES, VoiceLoop, VoiceSettings

    voice = config.voice
    if not (voice.wake_word or voice.push_to_talk):
        raise VoiceSetupError("в config.yaml выключены и слово активации, и push-to-talk")
    capture = MicrophoneCapture(voice.input_device)
    try:
        capture.start()
    except CaptureError as exc:
        raise VoiceSetupError(str(exc)) from exc

    hotkeys = HotkeyManager(hotkeys_paused)
    try:
        vosk_model = None
        try:
            vosk_model = load_vosk_model(config.resolve(voice.vosk_model))
        except VoskModelError as exc:
            out.write(f"! {exc}\n")
        spotter = None
        if voice.wake_word and vosk_model is not None:
            spotter = wake_spotter(vosk_model, config.general.name)
            if not spotter.usable:
                out.write(
                    f"! Слова «{config.general.name}» нет в словаре модели Vosk — активация по имени "
                    "выключена. Выберите другое имя (general.name) или используйте горячую клавишу.\n"
                )
                spotter = None
        if spotter is None and not voice.push_to_talk:
            raise VoiceSetupError("слово активации недоступно, а push-to-talk выключен")

        out.write("Загружаю распознавание речи (Whisper)… при первом запуске модель скачивается.\n")
        stt = WhisperSTT(
            config.stt, config.resolve(config.stt.download_dir), whisper_prompt(config, store, index)
        )
        try:
            stt.load()
        except Exception as exc:
            log.exception("Не удалось загрузить Whisper")
            raise VoiceSetupError(
                f"не удалось загрузить Whisper ({exc}). Нужен интернет для первой загрузки"
            ) from exc
        if stt.cuda_problem:
            out.write(f"! {stt.cuda_problem}\n")
        where = "видеокарте" if stt.device == "cuda" else "процессоре"
        out.write(f"Распознавание речи: Whisper {stt.model_name} на {where}.\n")

        confirmer = None
        if vosk_model is not None and speech is not None:
            model = vosk_model
            confirmer = VoiceConfirmer(
                speech,
                capture,
                lambda: phrase_spotter(model, YES_PHRASES + NO_PHRASES),
                voice.confirm_timeout,
                voice.confirm_confidence,
            )
        stop_spotter = None
        if vosk_model is not None:
            # «Джарвис, стоп» тоже считается — если имя есть в словаре Vosk.
            stop_phrases = [*STOP_PHRASES, *([config.general.name] if spotter is not None else [])]
            stop_spotter = phrase_spotter(vosk_model, stop_phrases, exclude_decoys=STOP_SOUNDALIKES)
        segmenter = SpeechSegmenter(
            SegmenterSettings(
                threshold=voice.vad_threshold,
                neg_threshold=max(0.05, voice.vad_threshold - 0.15),
                min_silence_ms=voice.min_silence_ms,
                max_seconds=voice.max_phrase_seconds,
            )
        )
        loop = VoiceLoop(
            source=capture,
            vad=SileroVAD(),
            segmenter=segmenter,
            spotter=spotter,
            stt=stt,
            handle_text=handle_text or (lambda text, conf: pipeline.handle_text(text, "voice", conf)),
            speech=speech or SilentSpeech(),
            normalizer=normalizer,
            wake_names=config.wake_words,
            settings=VoiceSettings(
                wake_word=spotter is not None,
                wake_confidence=voice.wake_confidence,
                stop_confidence=voice.stop_confidence,
                verify_wake_word=voice.verify_wake_word,
                followup_seconds=voice.followup_seconds,
                listen_timeout=voice.listen_timeout,
                push_to_talk_mode=voice.push_to_talk_mode,
                max_phrase_seconds=voice.max_phrase_seconds,
            ),
            status=status,
            confirmer=confirmer,
            chime=(listen_chime(), SOUND_RATE) if voice.sounds else None,
            stop_spotter=stop_spotter,
            on_stop=pipeline.cancel,
        )

        def stop_all() -> None:
            """Горячая клавиша остановки: прервать задачу нейросети, речь и прослушивание."""
            pipeline.cancel()
            loop.stop_speech()

        ways = []
        if spotter is not None:
            ways.append(f"скажите «{config.general.name.capitalize()}»")
        if sys.platform == "win32":
            try:
                if voice.push_to_talk and voice.push_to_talk_mode == "hold":
                    hotkeys.add_hold(voice.push_to_talk_key, loop.hold_start, loop.hold_end)
                    ways.append(f"удерживайте {voice.push_to_talk_key}")
                elif voice.push_to_talk:
                    hotkeys.add_press(voice.push_to_talk_key, loop.push_to_talk)
                    ways.append(f"нажмите {voice.push_to_talk_key}")
                hotkeys.add_press(voice.stop_key, stop_all)
            except HotkeyError as exc:
                out.write(f"! {exc}\n")
        ways.append("или введите :listen")
        loop.start()
    except Exception:
        hotkeys.close()
        capture.stop()
        raise
    stop_ways = f"{voice.stop_key} или «стоп»" if stop_spotter is not None else voice.stop_key
    out.write(f"Голос включён: {', '.join(ways)}. Прервать ответ или задачу — {stop_ways}.\n")
    return VoiceRuntime(capture, loop, hotkeys, confirmer, wake_word=spotter is not None)


# --- Запуск ---------------------------------------------------------------------------


def build_masker(config: AppConfig) -> SecretMasker:
    """Маскировка секретов в логах: переменные *_KEY, *_TOKEN… и ключи всех профилей нейросети.

    Ключ своего профиля может лежать в переменной с любым именем — он тоже маскируется.
    """
    masker = SecretMasker.from_environ()
    for profile in config.llm.profiles.values():
        if profile.api_key_env:
            masker.add(os.environ.get(profile.api_key_env, ""))
    return masker


def describe_llm(config: AppConfig) -> str:
    """Строки о нейросети для консоли при старте: профиль и чего не хватает."""
    llm = config.llm
    if not llm.enabled:
        return "Нейросеть выключена (llm.enabled: false): работают только команды.\n"
    lines = [f"Нейросеть: профиль {llm.profile}, модель {llm.active.model}."]
    try:
        resolve_api_key(llm.profile, llm.active)
    except LlmError as exc:
        lines.append(f"! {exc}. Пока ключа нет, на фразы вне команд ответ — «Нейросеть недоступна».")
    if llm.profile == "opencode_zen" and llm.active.model.endswith("-free"):
        lines.append(
            "! Бесплатные модели OpenCode Zen с сентября 2026 отвечают 403 FreeTierError всем программам,"
            " кроме самого OpenCode. Если так и будет — выберите llm.profile: deepseek или ollama."
        )
    return "\n".join(lines) + "\n"


class LogWriter:
    """Вместо консоли, когда её нет (запуск через pythonw): строки уходят в лог."""

    def write(self, text: str) -> int:
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("!"):
                log.warning("%s", line)
            elif line:
                log.info("%s", line)
        return len(text)

    def flush(self) -> None:
        """Буфера нет."""


@dataclass
class Core:
    """Всё, что обрабатывает фразы: одинаково для окон и консоли."""

    index: AppIndex
    registry: ActionRegistry
    executor: ActionExecutor
    normalizer: Normalizer
    matcher: Matcher
    store: CommandStore
    loaded: LoadResult
    pipeline: Pipeline
    journal_file: Path
    log_file: Path
    alarms: Alarms


def make_agent(llm: LlmConfig, executor: ActionExecutor, speaker: Speaker, name: str) -> Agent:
    """Агент нейросети с текущими настройками."""
    return Agent(llm, executor, speaker, name=name, window_provider=windows.active_window)


def build_core(
    config: AppConfig,
    speaker: Speaker,
    confirmer: Confirmer,
    masker: SecretMasker,
    dry_run: bool,
    debug: bool,
) -> Core:
    """Действия, команды, матчер, агент и конвейер."""
    log_dir = config.resolve(config.logging.dir)
    index = AppIndex(config.apps)
    index.start()
    alarms = Alarms(config.resolve(TIMERS_FILE))
    registry = build_registry(config, index, speaker, alarms)
    journal_file = log_dir / config.logging.journal
    executor = ActionExecutor(
        registry, SafetyPolicy(), confirmer, ActionJournal(journal_file, masker), dry_run=dry_run
    )
    normalizer = Normalizer(config.wake_words, config.matcher.filler_words)
    matcher = Matcher(normalizer, config.matcher, debug=debug)
    store = CommandStore(config.resolve(config.commands.dir), config.commands.user_file, registry)
    loaded = store.reload()
    matcher.set_commands(store.commands)
    agent = make_agent(config.llm, executor, speaker, config.general.name) if config.llm.enabled else None
    pipeline = Pipeline(matcher, executor, speaker, windows.active_window, agent=agent)
    return Core(index, registry, executor, normalizer, matcher, store, loaded, pipeline, journal_file,
                log_dir / config.logging.file, alarms)  # fmt: skip


def ui_problem() -> str:
    """Почему окна не запустить ('' — можно). Без WebView2 предлагает открыть страницу загрузки."""
    import webbrowser

    from assistant.ui import win32
    from assistant.ui.windows import UI_DIST

    if not os.environ.get("JARVIS_UI_URL") and not (UI_DIST / "index.html").exists():
        return "интерфейс не собран: нет ui/dist/index.html (см. README, раздел «Интерфейс»)"
    if win32.webview2_version() is None:
        question = (
            "Для окон ассистента нужен Microsoft Edge WebView2 Runtime — он бесплатный.\n\n"
            "Открыть страницу загрузки? Пока его нет, ассистент работает в консоли."
        )
        if win32.ask_yes_no("Джарвис", question):
            webbrowser.open(win32.WEBVIEW2_DOWNLOAD)
        return f"нет Microsoft Edge WebView2 Runtime — установите его: {win32.WEBVIEW2_DOWNLOAD}"
    try:
        import webview  # noqa: F401
    except ImportError as exc:
        return f"не установлен pywebview ({exc}) — выполните uv sync"
    return ""


def relaunch(argv: list[str] | None) -> None:
    """Запускает ассистента заново с теми же аргументами (кнопка «Перезапустить»)."""
    args = sys.argv[1:] if argv is None else argv
    log.info("Перезапуск")
    subprocess.Popen([sys.executable, "-m", "assistant", *args])


def main(argv: list[str] | None = None) -> int:
    """Запускает ассистента. Возвращает код выхода."""
    started = time.perf_counter()
    args = parse_args(argv)
    out: Any = sys.stdout if sys.stdout is not None else LogWriter()
    # pythonw: консоли нет, а библиотеки (прогресс загрузки моделей) пишут в stdout и stderr.
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))  # noqa: SIM115
    if hasattr(out, "reconfigure"):
        out.reconfigure(errors="replace")
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        if sys.stderr is not None:
            sys.stderr.write(f"{exc}\n")
        return 2

    if args.list_devices:
        from assistant.audio.capture import list_input_devices

        for index_, name in list_input_devices():
            out.write(f"{index_:3}  {name}\n")
        return 0

    debug = args.debug or config.general.debug
    dry_run = args.dry_run or config.general.dry_run
    masker = build_masker(config)
    log_dir = config.resolve(config.logging.dir)
    setup_logging(
        log_dir / config.logging.file,
        "DEBUG" if debug else config.logging.level,
        masker,
        console_level="DEBUG" if debug else "WARNING",
    )
    log.info("Запуск: Python %s, %s, dry-run=%s", platform.python_version(), platform.platform(), dry_run)

    if args.download_models:
        from assistant.models import download_all

        return 0 if download_all(config, out) else 1

    if config.ui.enabled and not args.console and sys.platform == "win32":
        problem = ui_problem()
        if not problem:
            return run_ui(config, args, argv, masker, dry_run, debug, started)
        out.write(f"! Окна недоступны: {problem}. Работаю в консоли.\n")
    return run_console(config, args, out, masker, dry_run, debug, started)


def run_console(
    config: AppConfig, args: argparse.Namespace, out: Any, masker: SecretMasker, dry_run: bool, debug: bool,
    started: float,
) -> int:  # fmt: skip
    """Консольный режим: фразы вводятся текстом, голос — как в окнах."""
    voice_wanted = config.voice.enabled and not args.text
    speech = create_speech(config, out) if voice_wanted else None
    console_speaker = ConsoleSpeaker(config.general.name)
    speaker: Speaker = TeeSpeaker(console_speaker, speech) if speech else console_speaker
    core = build_core(config, speaker, ConsoleConfirmer(), masker, dry_run, debug)
    console = ConsoleApp(
        core.pipeline, console_speaker, core.matcher, core.store, core.index, windows.active_window
    )

    for issue in core.loaded.issues:
        out.write(f"! Команда пропущена: {issue}\n")
    out.write(describe_llm(config))
    if dry_run:
        out.write("Режим dry-run: действия только записываются в журнал.\n")
    if sys.platform != "win32":
        out.write("Внимание: это не Windows — действия с системой работать не будут.\n")

    runtime = None
    if voice_wanted:
        try:
            runtime = start_voice(
                config, core.pipeline, core.normalizer, console, speech, core.store, core.index, out
            )
            console.on_listen = runtime.loop.push_to_talk
        except VoiceSetupError as exc:
            out.write(f"! Голос недоступен: {exc}. Работаю с вводом текста.\n")
        except Exception as exc:
            log.exception("Ошибка запуска голоса")
            out.write(
                f"! Голос недоступен из-за ошибки: {exc}. Подробности в логе. Работаю с вводом текста.\n"
            )
    log.info(
        "Готов к работе за %.1f с: действий %d, команд %d, голос %s",
        time.perf_counter() - started, len(core.registry), len(core.store.commands),
        "вкл" if runtime else "выкл",
    )  # fmt: skip

    def reload_commands() -> None:
        result = core.store.reload()
        core.pipeline.reload_commands(core.store.commands)
        out.write(f"Команды перечитаны: {len(result.commands)}\n")
        for issue in result.issues:
            out.write(f"! Команда пропущена: {issue}\n")

    def alarm(alarm: Alarm, missed: bool) -> None:
        if speech is not None:
            speech.play_sound(alarm_chime(), SOUND_RATE)
        speaker.say(alarm.message(missed))

    core.alarms.on_fire = alarm
    core.alarms.start()
    watcher = CommandWatcher(core.store.directory, reload_commands)
    watcher.start()
    try:
        console.run()
    finally:
        watcher.stop()
        core.alarms.stop()
        if runtime is not None:
            runtime.close()
        if speech is not None:
            speech.close()
    log.info("Выход")
    return 0


def run_ui(
    config: AppConfig, args: argparse.Namespace, argv: list[str] | None, masker: SecretMasker, dry_run: bool,
    debug: bool, started: float,
) -> int:  # fmt: skip
    """Режим с окнами: главное окно, оверлей, быстрый чат и значок в трее."""
    from assistant.actions.volume import VolumeController
    from assistant.ui.activity import Activity, UiSpeaker, VoiceStatus
    from assistant.ui.app import UiApp
    from assistant.ui.confirm import UiConfirmer
    from assistant.ui.controller import UiController
    from assistant.ui.events import EventBus
    from assistant.ui.settings import SettingsStore

    bus = EventBus()
    activity = Activity(bus)
    log_writer = LogWriter()
    voice_wanted = config.voice.enabled and not args.text
    speech = create_speech(config, log_writer) if voice_wanted else None
    ui_speaker = UiSpeaker(activity)
    speaker: Speaker = TeeSpeaker(ui_speaker, speech) if speech else ui_speaker
    confirmer = UiConfirmer(bus, timeout=30)
    core = build_core(config, speaker, confirmer.for_source("text"), masker, dry_run, debug)
    core.pipeline.on_agent_step = lambda step: activity.step(
        step.number, None, step.action, step.text, step.status in ("done", "dry_run")
    )
    core.pipeline.on_command_step = lambda number, total, step: activity.step(
        number, total, step.action, step.text, step.ok
    )
    hotkeys_paused = threading.Event()
    store = SettingsStore(config.resolve(args.config.name), config.resolve(ENV_FILE), on_key=masker.add)
    controller = UiController(
        config=config,
        store=store,
        pipeline=core.pipeline,
        executor=core.executor,
        commands=core.store,
        make_agent=lambda llm: make_agent(llm, core.executor, speaker, config.general.name),
        speech=speech,
        volume=VolumeController(),
        bus=bus,
        activity=activity,
        confirmer=confirmer,
        log_file=core.log_file,
        journal_file=core.journal_file,
        force_dry_run=args.dry_run,
        hotkeys_paused=hotkeys_paused,
        alarms=core.alarms,
    )
    controller.startup_issues = [str(issue) for issue in core.loaded.issues]
    confirmer.on_change = controller.overlay.confirming
    status = VoiceStatus(activity, controller.on_voice_state)

    def handle_voice(text: str, _: Confirmer | None) -> Any:
        # Голосовая команда подтверждается и голосом, и кнопками в оверлее.
        return core.pipeline.handle_text(text, "voice", confirmer.for_source("voice"))

    def voice_starter(progress: Any) -> VoiceRuntime:
        return start_voice(
            config, core.pipeline, core.normalizer, status, speech, core.store, core.index, progress,
            handle_text=handle_voice, hotkeys_paused=hotkeys_paused,
        )  # fmt: skip

    problem = ""
    if not config.voice.enabled:
        problem = "Голос выключен в настройках"
    elif args.text:
        problem = "Запущено с --text: голос выключен"
    app = UiApp(
        controller,
        voice_starter=voice_starter if voice_wanted else None,
        voice_problem=problem,
        debug=debug,
        dev_url=os.environ.get("JARVIS_UI_URL"),
    )
    log.info(
        "Окна готовы к запуску за %.1f с: действий %d, команд %d",
        time.perf_counter() - started, len(core.registry), len(core.store.commands),
    )  # fmt: skip
    try:
        app.run()
    finally:
        if controller.voice is not None:
            controller.voice.close()
        if speech is not None:
            speech.close()
    log.info("Выход")
    if controller.restart_requested:
        relaunch(argv)
    return 0
