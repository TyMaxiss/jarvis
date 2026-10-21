"""Загрузка и валидация конфигурации (config.yaml) и секретов (.env)."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

log = logging.getLogger(__name__)

CONFIG_FILE = "config.yaml"
EXAMPLE_FILE = "config.example.yaml"
ENV_FILE = ".env"


class ConfigError(Exception):
    """Конфиг не найден, не читается или содержит ошибки."""


class _Section(BaseModel):
    """База для разделов конфига: неизвестные ключи считаются ошибкой (опечатки)."""

    model_config = ConfigDict(extra="forbid")


class GeneralConfig(_Section):
    """Общие настройки ассистента."""

    name: str = Field("джарвис", description="Имя ассистента, оно же wake word")
    name_variants: list[str] = Field(
        default_factory=lambda: ["джарвиз", "жарвис", "jarvis"],
        description="Как ещё распознаватель может записать имя; удаляется из фразы",
    )
    dry_run: bool = Field(False, description="Только логировать действия, не выполняя их")
    debug: bool = Field(False, description="Подробный лог, в том числе топ-3 кандидата матчера")


class MatcherConfig(_Section):
    """Настройки локального поиска команд."""

    threshold: float = Field(85, ge=0, le=100, description="Минимальная оценка совпадения")
    ambiguity_gap: float = Field(
        5, ge=0, le=100, description="Если два лучших кандидата ближе — фраза уходит в LLM"
    )
    filler_words: list[str] = Field(
        default_factory=lambda: [
            "пожалуйста",
            "слушай",
            "ну",
            "давай",
            "ка",
            "мне",
            "будь добр",
            "будь добра",
            "эй",
            "окей",
        ]  # fmt: skip
    )
    split_words: list[str] = Field(
        default_factory=lambda: ["а затем", "а потом", "и затем", "и потом", "затем", "потом", "и"],
        description="Союзы, по которым делятся составные фразы",
    )


class CommandsConfig(_Section):
    """Где лежат YAML-паки команд."""

    dir: Path = Path("commands")
    user_file: str = Field("my.yaml", description="Пользовательский пак: грузится последним")


class AppsConfig(_Section):
    """Поиск и запуск приложений."""

    aliases: dict[str, str] = Field(
        default_factory=dict,
        description="Разговорное название → название приложения, путь или URI",
    )
    match_threshold: float = Field(80, ge=0, le=100)
    launch_wait: float = Field(
        3.0, ge=0, le=15, description="Сколько секунд ждать появления окна после запуска"
    )
    exclude: list[str] = Field(
        default_factory=lambda: [
            "uninstall",
            "удалить",
            "удаление",
            "деинсталл",
            "readme",
            "help",
            "справка",
            "website",
            "веб-сайт",
            "documentation",
            "документация",
            "license",
            "лицензия",
            "release notes",
        ]  # fmt: skip
    )


class BrowserConfig(_Section):
    """Адреса для поиска в интернете и на YouTube."""

    search_url: str = "https://www.google.com/search?q={query}"
    youtube_search_url: str = "https://www.youtube.com/results?search_query={query}"


class LoggingConfig(_Section):
    """Логи и журнал действий."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    dir: Path = Path("logs")
    file: str = "assistant.log"
    journal: str = "actions.jsonl"


class AppConfig(_Section):
    """Вся конфигурация ассистента."""

    general: GeneralConfig = Field(default_factory=GeneralConfig)
    matcher: MatcherConfig = Field(default_factory=MatcherConfig)
    commands: CommandsConfig = Field(default_factory=CommandsConfig)
    apps: AppsConfig = Field(default_factory=AppsConfig)
    browser: BrowserConfig = Field(default_factory=BrowserConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    _base_dir: Path = PrivateAttr(default_factory=Path.cwd)

    @property
    def base_dir(self) -> Path:
        """Папка, относительно которой разрешаются пути из конфига."""
        return self._base_dir

    def resolve(self, path: Path | str) -> Path:
        """Превращает путь из конфига в абсолютный (относительно папки конфига)."""
        path = Path(path)
        return path if path.is_absolute() else self._base_dir / path

    @property
    def wake_words(self) -> list[str]:
        """Имя ассистента вместе с вариантами написания."""
        return [self.general.name, *self.general.name_variants]


def _format_validation_error(path: Path, error: ValidationError) -> str:
    """Делает понятный текст из ошибки pydantic."""
    lines = [f"Ошибки в {path.name}:"]
    for item in error.errors():
        where = ".".join(str(part) for part in item["loc"]) or "(корень)"
        if item["type"] == "extra_forbidden":
            lines.append(f"  {where}: неизвестный параметр (опечатка?)")
        else:
            lines.append(f"  {where}: {item['msg']}")
    return "\n".join(lines)


def load_config(path: Path) -> AppConfig:
    """Загружает config.yaml и переменные из .env рядом с ним.

    Если config.yaml нет, он создаётся копированием config.example.yaml.

    Raises:
        ConfigError: файл не читается или содержит ошибки.
    """
    path = path.resolve()
    base_dir = path.parent
    env_path = base_dir / ENV_FILE
    if env_path.exists():
        load_dotenv(env_path, override=False)

    if not path.exists():
        example = base_dir / EXAMPLE_FILE
        if example.exists():
            shutil.copyfile(example, path)
            log.info("Создан %s из %s", path.name, example.name)
        else:
            log.warning("%s не найден, используются настройки по умолчанию", path.name)
            config = AppConfig()
            config._base_dir = base_dir
            return config

    try:
        data = YAML(typ="safe", pure=True).load(path.read_text(encoding="utf-8")) or {}
    except (OSError, YAMLError) as exc:
        raise ConfigError(f"Не удалось прочитать {path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path.name}: на верхнем уровне должен быть словарь разделов")

    try:
        config = AppConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(path, exc)) from exc
    config._base_dir = base_dir
    return config
