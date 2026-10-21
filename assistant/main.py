"""Точка входа: сборка компонентов и запуск.

Это единственное место, где создаются и связываются объекты приложения.
"""

from __future__ import annotations

import argparse
import logging
import platform
import sys
import time
from pathlib import Path

from assistant.actions import apps, browser, media, system, volume, windows
from assistant.actions import input as input_actions
from assistant.actions.apps import AppIndex
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
from assistant.safety.confirm import ConsoleConfirmer
from assistant.safety.policy import SafetyPolicy
from assistant.tts.base import Speaker

log = logging.getLogger("assistant")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разбирает аргументы командной строки."""
    parser = argparse.ArgumentParser(prog="python -m assistant", description="Голосовой ассистент")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="путь к config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="только логировать действия")
    parser.add_argument("--debug", action="store_true", help="подробный лог и вывод в консоль")
    parser.add_argument(
        "--text", action="store_true", help="консольный режим: фразы вводятся текстом (сейчас единственный)"
    )
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


def main(argv: list[str] | None = None) -> int:
    """Запускает ассистента. Возвращает код выхода."""
    started = time.perf_counter()
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # type: ignore[union-attr]
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

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

    speaker = ConsoleSpeaker(config.general.name)
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

    for issue in loaded.issues:
        sys.stdout.write(f"! Команда пропущена: {issue}\n")
    if dry_run:
        sys.stdout.write("Режим dry-run: действия только записываются в журнал.\n")
    if sys.platform != "win32":
        sys.stdout.write("Внимание: это не Windows — действия с системой работать не будут.\n")
    log.info("Готов к работе за %.1f с: действий %d, команд %d", time.perf_counter() - started,
             len(registry), len(store.commands))  # fmt: skip

    ConsoleApp(pipeline, speaker, matcher, store, index, windows.active_window).run()
    log.info("Выход")
    return 0
