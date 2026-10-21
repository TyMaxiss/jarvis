"""Точка входа: сборка компонентов и запуск.

Это единственное место, где создаются и связываются объекты приложения.
"""

from __future__ import annotations

import argparse
import logging
import platform
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from assistant.actions import apps, browser, media, system, volume, windows
from assistant.actions import input as input_actions
from assistant.actions.apps import AppIndex, looks_like_path
from assistant.actions.executor import ActionExecutor
from assistant.actions.registry import ActionRegistry
from assistant.commands.loader import CommandStore
from assistant.config import AppConfig, ConfigError, load_config
from assistant.console import ConsoleApp, ConsoleSpeaker
from assistant.journal import ActionJournal
from assistant.logging_setup import SecretMasker, setup_logging
from assistant.nlu.matcher import Matcher
from assistant.nlu.normalize import Normalizer
from assistant.pipeline import Pipeline
from assistant.safety.confirm import NO_PHRASES, YES_PHRASES, ConsoleConfirmer, VoiceConfirmer
from assistant.safety.policy import SafetyPolicy
from assistant.tts.base import SilentSpeech, Speaker, SpeechQueue, TeeSpeaker, TTSEngine, TTSError

if TYPE_CHECKING:
    from assistant.audio.capture import MicrophoneCapture
    from assistant.hotkeys import HotkeyManager
    from assistant.voice import VoiceLoop

log = logging.getLogger("assistant")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разбирает аргументы командной строки."""
    parser = argparse.ArgumentParser(prog="python -m assistant", description="Голосовой ассистент")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="путь к config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="только логировать действия")
    parser.add_argument("--debug", action="store_true", help="подробный лог и вывод в консоль")
    parser.add_argument("--text", action="store_true", help="только ввод текстом, без микрофона и озвучки")
    parser.add_argument("--download-models", action="store_true", help="скачать модели Vosk, Piper, Whisper")
    parser.add_argument("--list-devices", action="store_true", help="показать микрофоны и выйти")
    return parser.parse_args(argv)


def build_registry(config: AppConfig, index: AppIndex, speaker: Speaker) -> ActionRegistry:
    """Создаёт реестр и регистрирует все действия."""
    registry = ActionRegistry()
    apps.register(registry, index, config.apps)
    windows.register(registry, config.apps.match_threshold)
    input_actions.register(registry)
    media.register(registry)
    volume.register(registry)
    browser.register(registry, config.browser)
    system.register(registry, speaker)
    return registry


# --- Голос --------------------------------------------------------------------------


class VoiceSetupError(RuntimeError):
    """Голос запустить не удалось; ассистент продолжит работать с вводом текста."""


class VoiceRuntime:
    """Запущенные голосовые компоненты — чтобы остановить их при выходе."""

    def __init__(self, capture: MicrophoneCapture, loop: VoiceLoop, hotkeys: HotkeyManager) -> None:
        self.capture = capture
        self.loop = loop
        self.hotkeys = hotkeys

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
    console: ConsoleApp,
    speech: SpeechQueue | None,
    store: CommandStore,
    index: AppIndex,
    out: TextIO,
) -> VoiceRuntime:
    """Запускает микрофон, распознавание и голосовой цикл.

    Raises:
        VoiceSetupError: микрофона нет, нечем активировать или не загрузилась модель.
    """
    from assistant.audio.capture import CaptureError, MicrophoneCapture
    from assistant.audio.sounds import SOUND_RATE, listen_chime
    from assistant.audio.vad import SegmenterSettings, SileroVAD, SpeechSegmenter
    from assistant.audio.wakeword import VoskModelError, load_vosk_model, phrase_spotter, wake_spotter
    from assistant.hotkeys import HotkeyError, HotkeyManager
    from assistant.stt.whisper_stt import WhisperSTT
    from assistant.voice import VoiceLoop, VoiceSettings

    voice = config.voice
    if not (voice.wake_word or voice.push_to_talk):
        raise VoiceSetupError("в config.yaml выключены и слово активации, и push-to-talk")
    capture = MicrophoneCapture(voice.input_device)
    try:
        capture.start()
    except CaptureError as exc:
        raise VoiceSetupError(str(exc)) from exc

    hotkeys = HotkeyManager()
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
            handle_text=lambda text, conf: pipeline.handle_text(text, "voice", conf),
            speech=speech or SilentSpeech(),
            normalizer=normalizer,
            wake_names=config.wake_words,
            settings=VoiceSettings(
                wake_word=spotter is not None,
                wake_confidence=voice.wake_confidence,
                verify_wake_word=voice.verify_wake_word,
                followup_seconds=voice.followup_seconds,
                listen_timeout=voice.listen_timeout,
                push_to_talk_mode=voice.push_to_talk_mode,
                max_phrase_seconds=voice.max_phrase_seconds,
            ),
            status=console,
            confirmer=confirmer,
            chime=(listen_chime(), SOUND_RATE) if voice.sounds else None,
        )

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
                hotkeys.add_press(voice.stop_key, loop.stop_speech)
            except HotkeyError as exc:
                out.write(f"! {exc}\n")
        ways.append("или введите :listen")
        loop.start()
    except Exception:
        hotkeys.close()
        capture.stop()
        raise
    out.write(f"Голос включён: {', '.join(ways)}. Прервать речь — {voice.stop_key}.\n")
    return VoiceRuntime(capture, loop, hotkeys)


# --- Запуск ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Запускает ассистента. Возвращает код выхода."""
    started = time.perf_counter()
    args = parse_args(argv)
    out = sys.stdout
    if hasattr(out, "reconfigure"):
        out.reconfigure(errors="replace")  # type: ignore[union-attr]
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    if args.list_devices:
        from assistant.audio.capture import list_input_devices

        for index_, name in list_input_devices():
            out.write(f"{index_:3}  {name}\n")
        return 0

    debug = args.debug or config.general.debug
    dry_run = args.dry_run or config.general.dry_run
    masker = SecretMasker.from_environ()
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

    voice_wanted = config.voice.enabled and not args.text
    speech = create_speech(config, out) if voice_wanted else None
    console_speaker = ConsoleSpeaker(config.general.name)
    speaker: Speaker = TeeSpeaker(console_speaker, speech) if speech else console_speaker

    index = AppIndex(config.apps)
    index.start()
    registry = build_registry(config, index, speaker)
    journal = ActionJournal(log_dir / config.logging.journal, masker)
    executor = ActionExecutor(registry, SafetyPolicy(), ConsoleConfirmer(), journal, dry_run=dry_run)
    normalizer = Normalizer(config.wake_words, config.matcher.filler_words)
    matcher = Matcher(normalizer, config.matcher, debug=debug)
    store = CommandStore(config.resolve(config.commands.dir), config.commands.user_file, registry)
    loaded = store.reload()
    matcher.set_commands(store.commands)
    pipeline = Pipeline(matcher, executor, speaker, windows.active_window)
    console = ConsoleApp(pipeline, console_speaker, matcher, store, index, windows.active_window)

    for issue in loaded.issues:
        out.write(f"! Команда пропущена: {issue}\n")
    if dry_run:
        out.write("Режим dry-run: действия только записываются в журнал.\n")
    if sys.platform != "win32":
        out.write("Внимание: это не Windows — действия с системой работать не будут.\n")

    runtime = None
    if voice_wanted:
        try:
            runtime = start_voice(config, pipeline, normalizer, console, speech, store, index, out)
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
        time.perf_counter() - started, len(registry), len(store.commands), "вкл" if runtime else "выкл",
    )  # fmt: skip

    try:
        console.run()
    finally:
        if runtime is not None:
            runtime.close()
        if speech is not None:
            speech.close()
    log.info("Выход")
    return 0
