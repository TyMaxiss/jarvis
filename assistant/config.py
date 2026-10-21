"""Загрузка и валидация конфигурации (config.yaml) и секретов (.env)."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Literal

from dotenv import dotenv_values
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    field_validator,
    model_validator,
)
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


class VoiceConfig(_Section):
    """Голосовой ввод: микрофон, активация, определение конца фразы."""

    enabled: bool = Field(True, description="false — только ввод текстом в консоли")
    wake_word: bool = Field(True, description="Активация по имени ассистента (Vosk)")
    push_to_talk: bool = Field(True, description="Активация горячей клавишей")
    push_to_talk_key: str = "ctrl+alt+space"
    push_to_talk_mode: Literal["press", "hold"] = Field(
        "press", description="press — нажать и говорить (конец по паузе), hold — говорить, удерживая"
    )
    stop_key: str = Field("ctrl+alt+q", description="Прервать речь ассистента и задачу нейросети")
    followup_seconds: float = Field(5, ge=0, le=60, description="Сколько слушать продолжение без имени")
    listen_timeout: float = Field(6, ge=1, le=60, description="Сколько ждать начала речи после активации")
    min_silence_ms: int = Field(700, ge=200, le=3000, description="Пауза, после которой фраза закончена")
    max_phrase_seconds: float = Field(15, ge=2, le=60)
    vad_threshold: float = Field(0.5, gt=0, lt=1)
    input_device: int | str | None = Field(None, description="Номер или часть названия микрофона")
    sounds: bool = Field(True, description="Короткий сигнал «слушаю»")
    wake_confidence: float = Field(0.8, ge=0, le=1, description="Уверенность Vosk в имени ассистента")
    verify_wake_word: bool = Field(True, description="Имя должно быть и в тексте от Whisper")
    confirm_timeout: float = Field(5, ge=1, le=30, description="Сколько ждать «да» или «нет»")
    confirm_confidence: float = Field(0.7, ge=0, le=1, description="Уверенность Vosk в «да/нет»")
    stop_confidence: float = Field(
        0.8, ge=0, le=1, description="Уверенность Vosk в «стоп» / «отмена» во время задачи и речи"
    )
    vosk_model: Path = Path("models/vosk-model-small-ru-0.22")


class SttConfig(_Section):
    """Распознавание речи (faster-whisper)."""

    model: str = Field("auto", description="auto: large-v3-turbo на видеокарте, small на процессоре")
    device: Literal["auto", "cuda", "cpu"] = "auto"
    compute_type: str = Field("auto", description="auto: float16 на видеокарте, int8 на процессоре")
    beam_size: int = Field(5, ge=1, le=10, description="5 — точнее, 1 — на несколько процентов быстрее")
    cpu_threads: int = Field(
        0, ge=0, le=64, description="Потоков на процессоре; 0 — по числу ядер, но не меньше 4"
    )
    download_dir: Path = Path("models/whisper")
    prompt_app_names: int = Field(30, ge=0, le=200, description="Сколько названий программ подсказывать")


class TtsConfig(_Section):
    """Озвучка ответов."""

    engine: Literal["piper", "sapi"] = "piper"
    piper_voice: Path = Path("models/piper/ru_RU-dmitri-medium.onnx")
    sapi_voice: str = Field("", description="Часть названия голоса SAPI; пусто — первый русский")
    rate: float = Field(1.0, ge=0.5, le=2.0, description="Скорость речи")
    volume: float = Field(1.0, ge=0.1, le=2.0)


class LlmProfile(_Section):
    """Провайдер нейросети: OpenAI-совместимый API (DeepSeek, ChatGPT, Ollama…) или Claude."""

    title: str = Field("", description="Название в интерфейсе; пусто — имя профиля")
    provider: Literal["openai", "anthropic"] = Field(
        "openai", description="openai — любой OpenAI-совместимый API; anthropic — Claude через SDK anthropic"
    )
    base_url: str = Field("", description="Адрес API; для anthropic можно не указывать")
    model: str
    api_key_env: str = Field("", description="Переменная в .env с ключом API; пусто — ключ не нужен")
    vision: bool = Field(False, description="Модель принимает картинки (снимок экрана — этап 5)")
    temperature: float | None = Field(
        None, ge=0, le=2, description="Не задана — не передаётся: часть моделей её не принимает"
    )
    max_tokens: int | None = Field(None, ge=1, description="Предел длины ответа; для Claude — 16000")
    effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None = Field(
        None,
        description=(
            "Сколько модели думать перед ответом: ChatGPT — reasoning_effort (none — не думать), "
            "Claude — effort (low — быстрее всего)"
        ),
    )
    fallbacks: bool = Field(
        True, description="Claude: если модель откажется отвечать, повторить на рекомендованной запасной"
    )
    extra_body: dict[str, Any] = Field(
        default_factory=dict, description="Дополнительные поля запроса, например отключение рассуждений"
    )

    @model_validator(mode="after")
    def _base_url_for_openai(self) -> LlmProfile:
        if self.provider == "openai" and not self.base_url:
            raise ValueError("для провайдера openai нужен base_url — адрес OpenAI-совместимого API")
        if self.provider == "anthropic" and self.effort in ("none", "minimal"):
            raise ValueError(f"у Claude нет effort {self.effort}: используйте значения от low до max")
        return self


# Профили из ТЗ. Пользовательские настройки профиля дополняют эти, а не заменяют целиком.
DEFAULT_LLM_PROFILES: dict[str, dict[str, Any]] = {
    "deepseek": {
        "title": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-flash",
        "api_key_env": "DEEPSEEK_API_KEY",
        "vision": True,
        "temperature": 0.3,
        # Для команд нужен быстрый режим без рассуждений (api-docs.deepseek.com, Thinking Mode).
        "extra_body": {"thinking": {"type": "disabled"}},
    },
    "openai": {
        "title": "ChatGPT",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-6-luna",
        "api_key_env": "OPENAI_API_KEY",
        "vision": True,
        # С GPT-5.4 Chat Completions вызывает инструменты только без рассуждений:
        # reasoning_effort "none" (developers.openai.com, Migrate to Responses).
        # temperature не задана: при других effort модели её не принимают.
        "effort": "none",
    },
    "anthropic": {
        "title": "Claude",
        "provider": "anthropic",
        "model": "claude-opus-5-5",
        "api_key_env": "ANTHROPIC_API_KEY",
        "vision": True,
        "max_tokens": 16000,
        # Рассуждения у Claude Opus 5.5 не выключаются; low — самый быстрый ответ.
        "effort": "low",
    },
    "ollama": {
        "title": "Ollama (без интернета)",
        "base_url": "http://localhost:11434/v1",
        "model": "qwen3:8b",
        "temperature": 0.3,
        # Без «размышлений»: быстрее отвечает. Для моделей вроде qwen3 Ollama переводит
        # reasoning_effort "none" в think=false (docs/api/openai-compatibility в репозитории ollama).
        "extra_body": {"reasoning_effort": "none"},
    },
    "opencode_zen": {
        "title": "OpenCode Zen",
        "base_url": "https://opencode.ai/zen/v1",
        "model": "deepseek-v4-flash-free",
        "api_key_env": "OPENCODE_API_KEY",
        "temperature": 0.3,
    },
}


class LlmConfig(_Section):
    """Нейросеть: профиль провайдера и лимиты агента."""

    enabled: bool = Field(True, description="false — фразы, которых нет в командах, не обрабатываются")
    profile: str = Field("deepseek", description="Какой профиль из profiles использовать")
    profiles: dict[str, LlmProfile] = Field(
        default_factory=lambda: {name: LlmProfile(**data) for name, data in DEFAULT_LLM_PROFILES.items()}
    )
    max_steps: int = Field(15, ge=1, le=50, description="Сколько раз агент может обратиться к нейросети")
    timeout: float = Field(120, ge=10, le=600, description="Общий лимит на одну задачу, секунды")
    request_timeout: float = Field(30, ge=5, le=300, description="Лимит на один запрос, секунды")
    retries: int = Field(2, ge=0, le=5, description="Повторы при ошибках сети, таймаутах и 429")
    history: int = Field(10, ge=0, le=50, description="Сколько последних обменов помнить")
    reset_after: float = Field(300, ge=0, description="Через сколько секунд тишины забыть разговор")

    @field_validator("profiles", mode="before")
    @classmethod
    def _merge_default_profiles(cls, value: Any) -> Any:
        """Встроенные профили + настройки пользователя поверх них."""
        if value is not None and not isinstance(value, dict):
            return value
        merged = {name: dict(data) for name, data in DEFAULT_LLM_PROFILES.items()}
        for name, data in (value or {}).items():
            merged[name] = {**merged.get(name, {}), **(data or {})}
        return merged

    @model_validator(mode="after")
    def _profile_exists(self) -> LlmConfig:
        if self.profile not in self.profiles:
            raise ValueError(f"профиль «{self.profile}» не найден; есть: {', '.join(self.profiles)}")
        return self

    @property
    def active(self) -> LlmProfile:
        """Выбранный профиль."""
        return self.profiles[self.profile]


class UiConfig(_Section):
    """Интерфейс: окна, тема, оверлей, быстрый чат."""

    enabled: bool = Field(True, description="false — консольный режим без окон")
    theme: Literal["dark", "light"] = "dark"
    accent: str = Field("#7c6cff", pattern=r"^#[0-9a-fA-F]{6}$", description="Акцентный цвет, #rrggbb")
    overlay: bool = Field(True, description="Плашка внизу экрана: что услышал и что делает")
    quick_chat_key: str = Field("ctrl+alt+enter", description="Окно для команды текстом")
    start_minimized: bool = Field(False, description="При запуске сразу свернуться в трей")


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
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    stt: SttConfig = Field(default_factory=SttConfig)
    tts: TtsConfig = Field(default_factory=TtsConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    ui: UiConfig = Field(default_factory=UiConfig)
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


def format_validation_error(path: Path, error: ValidationError) -> str:
    """Делает понятный текст из ошибки pydantic."""
    lines = [f"Ошибки в {path.name}:"]
    for item in error.errors():
        where = ".".join(str(part) for part in item["loc"]) or "(корень)"
        if item["type"] == "extra_forbidden":
            lines.append(f"  {where}: неизвестный параметр (опечатка?)")
        else:
            lines.append(f"  {where}: {item['msg']}")
    return "\n".join(lines)


def load_env(env_path: Path) -> None:
    """Переменные из .env в окружение процесса.

    Значения из .env важнее одноимённых переменных окружения Windows: ключ, сохранённый
    в настройках, должен работать, даже если в системе задан другой (например, старый).
    """
    values = {name: value for name, value in dotenv_values(env_path).items() if value is not None}
    for name, value in values.items():
        if os.environ.get(name, value) != value:
            log.info("%s берётся из %s, а не из переменных окружения", name, env_path.name)
    os.environ.update(values)


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
        load_env(env_path)

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
        raise ConfigError(format_validation_error(path, exc)) from exc
    config._base_dir = base_dir
    return config
