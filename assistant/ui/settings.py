"""Настройки из интерфейса: config.yaml (с сохранением комментариев) и ключи API в .env.

Интерфейс может менять только перечисленные здесь параметры (UI_FIELDS) и профиль
нейросети через update_llm; ключи пишутся в .env только в переменные известных профилей.
"""

from __future__ import annotations

import io
import logging
import os
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from ruamel.yaml.error import YAMLError

from assistant.config import DEFAULT_LLM_PROFILES, AppConfig, LlmProfile, format_validation_error
from assistant.llm.keys import clean_key, format_problem
from assistant.llm.providers import CUSTOM_KEY_ENV, CUSTOM_PROFILE, CUSTOM_TITLE, provider_info
from assistant.tts.edge_tts import EDGE_VOICES
from assistant.tts.silero_tts import SILERO_SPEAKERS

log = logging.getLogger(__name__)

# Что интерфейс может менять через save_settings.
UI_FIELDS: dict[str, frozenset[str]] = {
    "general": frozenset({"name", "dry_run"}),
    "voice": frozenset(
        {
            "enabled",
            "wake_word",
            "push_to_talk",
            "push_to_talk_key",
            "push_to_talk_mode",
            "stop_key",
            "input_device",
            "wake_confidence",
            "followup_seconds",
            "min_silence_ms",
            "sounds",
        }
    ),
    "stt": frozenset({"model", "device"}),
    "tts": frozenset(
        {"engine", "piper_voice", "silero_speaker", "edge_voice", "sapi_voice", "rate", "volume"}
    ),
    "ui": frozenset({"theme", "accent", "overlay", "quick_chat_key"}),
    "llm": frozenset({"enabled"}),
}
# Что применяется сразу; остальное — после перезапуска (голос, распознавание, голос озвучки, имя).
LIVE_FIELDS = frozenset(
    {("general", "dry_run"), ("llm", "enabled"), ("tts", "volume"), *(("ui", key) for key in UI_FIELDS["ui"])}
)

# Ключ API — одна строка видимых символов ASCII без пробелов.
_KEY_RE = re.compile(r"^[\x21-\x7e]{8,512}$")
_ENV_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


class SettingsError(Exception):
    """Изменения не прошли проверку; errors — понятные строки для интерфейса."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


def needs_restart(patch: Mapping[str, Mapping[str, Any]]) -> bool:
    """Нужен ли перезапуск, чтобы изменения вступили в силу."""
    return any((section, key) not in LIVE_FIELDS for section, values in patch.items() for key in values)


def check_key(value: str) -> str:
    """Ключ API без пробелов по краям и невидимых символов, прилипших при копировании.

    Raises:
        SettingsError: ключ записан с ошибкой: сокращённая запись, пробелы, не латиница.
    """
    key = clean_key(value)
    problem = format_problem(key)
    if problem:
        raise SettingsError([f"Ключ не подходит: {problem}"])
    if not _KEY_RE.match(key):
        raise SettingsError(["Ключ не подходит: он длиннее 512 символов"])
    return key


def key_hint(key: str) -> str:
    """Подсказка, какой ключ сохранён: «••••3f2a» (весь ключ интерфейс не получает)."""
    return f"••••{key[-4:]}" if len(key) >= 8 else ("••••" if key else "")


def _plain(node: Any) -> Any:
    """Узлы ruamel → обычные dict, list и скаляры (для проверки pydantic)."""
    if isinstance(node, Mapping):
        return {str(key): _plain(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_plain(item) for item in node]
    if isinstance(node, bool):
        return bool(node)
    if isinstance(node, int):
        return int(node)
    if isinstance(node, float):
        return float(node)
    if isinstance(node, str):
        return str(node)
    return node


def _section(data: CommentedMap, *path: str) -> CommentedMap:
    """Вложенный раздел; если его нет (или он пустой), создаётся."""
    node = data
    for key in path:
        child = node.get(key)
        if not isinstance(child, dict):
            child = CommentedMap()
            node[key] = child
        node = child
    return node


class SettingsStore:
    """Чтение и запись config.yaml и .env.

    Args:
        config_path: путь к config.yaml.
        env_path: путь к .env.
        environ: переменные окружения (ключи профилей); по умолчанию os.environ.
        on_key: вызывается с каждым сохранённым ключом (маскировка в логах).
    """

    def __init__(
        self,
        config_path: Path,
        env_path: Path,
        environ: dict[str, str] | None = None,
        on_key: Callable[[str], None] | None = None,
    ) -> None:
        self.config_path = config_path
        self.env_path = env_path
        self._environ = os.environ if environ is None else environ
        self._on_key = on_key
        self._lock = threading.Lock()

    # --- config.yaml -----------------------------------------------------------------

    def update(self, patch: Mapping[str, Mapping[str, Any]]) -> AppConfig:
        """Проверяет и сохраняет изменения из интерфейса; возвращает новый конфиг.

        Raises:
            SettingsError: неизвестный параметр или неверное значение.
        """
        errors = [
            f"{section}.{key}: этот параметр из интерфейса не меняется"
            for section, values in patch.items()
            for key in values
            if key not in UI_FIELDS.get(section, ())
        ]
        if errors:
            raise SettingsError(errors)

        def change(data: CommentedMap) -> None:
            for section, values in patch.items():
                node = _section(data, section)
                for key, value in values.items():
                    node[key] = self._yaml_value(section, key, value)

        return self._write(change)

    def update_llm(self, name: str, profile: Mapping[str, Any]) -> AppConfig:
        """Выбирает профиль нейросети и сохраняет его поля (модель, адрес…).

        Raises:
            SettingsError: профиль не прошёл проверку.
        """

        def change(data: CommentedMap) -> None:
            llm = _section(data, "llm")
            llm["profile"] = name
            node = _section(llm, "profiles", name)
            for key, value in profile.items():
                node[key] = value

        return self._write(change)

    def load(self) -> AppConfig:
        """Текущий config.yaml.

        Raises:
            SettingsError: файл не читается или содержит ошибки.
        """
        return self._write(None)

    def _write(self, change: Callable[[CommentedMap], None] | None) -> AppConfig:
        """Читает config.yaml, применяет изменение, проверяет и (если было изменение) сохраняет."""
        yaml = YAML()
        yaml.preserve_quotes = True
        with self._lock:
            try:
                text = self.config_path.read_text(encoding="utf-8") if self.config_path.exists() else ""
                data = yaml.load(text) if text.strip() else CommentedMap()
            except (OSError, YAMLError) as exc:
                raise SettingsError([f"Не удалось прочитать {self.config_path.name}: {exc}"]) from exc
            if not isinstance(data, CommentedMap):
                raise SettingsError([f"{self.config_path.name}: на верхнем уровне должны быть разделы"])
            if change is not None:
                change(data)
            try:
                config = AppConfig.model_validate(_plain(data))
            except ValidationError as exc:
                lines = format_validation_error(self.config_path, exc).splitlines()[1:]
                raise SettingsError([line.strip() for line in lines]) from exc
            if change is not None:
                buffer = io.StringIO()
                yaml.dump(data, buffer)
                temporary = self.config_path.with_name(self.config_path.name + ".tmp")
                temporary.write_text(buffer.getvalue(), encoding="utf-8")
                os.replace(temporary, self.config_path)
                log.info("Настройки сохранены в %s", self.config_path.name)
        config._base_dir = self.config_path.parent
        return config

    @staticmethod
    def _yaml_value(section: str, key: str, value: Any) -> Any:
        """Значение из интерфейса → значение для config.yaml."""
        if section == "voice" and key == "input_device":
            return value or None  # «» — микрофон по умолчанию
        if section == "general" and key == "name" and isinstance(value, str):
            return value.strip().lower()
        return value

    # --- Ключи в .env ----------------------------------------------------------------

    def save_key(self, env_name: str, value: str) -> None:
        """Записывает ключ в .env и в окружение процесса.

        Raises:
            SettingsError: ключ не похож на ключ или имя переменной неверное.
        """
        from dotenv import set_key

        key = check_key(value)
        if not _ENV_NAME_RE.match(env_name):
            raise SettingsError([f"Неверное имя переменной для ключа: {env_name}"])
        with self._lock:
            set_key(self.env_path, env_name, key, quote_mode="always")
        self._environ[env_name] = key
        if self._on_key is not None:
            self._on_key(key)
        log.info("Ключ %s сохранён в %s", env_name, self.env_path.name)

    def forget_key(self, env_name: str) -> None:
        """Удаляет ключ из .env и из окружения процесса."""
        from dotenv import unset_key

        with self._lock:
            if self.env_path.exists():
                unset_key(self.env_path, env_name, quote_mode="always")
        self._environ.pop(env_name, None)
        log.info("Ключ %s удалён из %s", env_name, self.env_path.name)

    def key(self, env_name: str) -> str:
        """Значение ключа из окружения ('' — не задан)."""
        return self._environ.get(env_name, "").strip() if env_name else ""

    def key_from_system(self, env_name: str) -> bool:
        """Ключ задан переменной окружения Windows, а не сохранён в .env."""
        from dotenv import dotenv_values

        if not self.key(env_name):
            return False
        with self._lock:
            saved = dotenv_values(self.env_path) if self.env_path.exists() else {}
        return not saved.get(env_name)


# --- Представление для интерфейса ------------------------------------------------------


@dataclass
class Choices:
    """Варианты для выпадающих списков в настройках."""

    microphones: list[tuple[int, str]] = field(default_factory=list)  # номер и название устройства
    piper_voices: list[tuple[str, str]] = field(default_factory=list)
    sapi_voices: list[tuple[str, str]] = field(default_factory=list)
    stt_models: list[tuple[str, str]] = field(
        default_factory=lambda: [
            ("auto", "Автоматически"),
            ("small", "small — быстро, для процессора"),
            ("medium", "medium — точнее, медленнее"),
            ("large-v3-turbo", "large-v3-turbo — для видеокарты"),
            ("large-v3", "large-v3 — точнее всех, нужна видеокарта"),
        ]
    )


_PIPER_NAMES = {"denis": "Денис", "dmitri": "Дмитрий", "irina": "Ирина", "ruslan": "Руслан"}


def piper_voices(config: AppConfig) -> list[tuple[str, str]]:
    """Голоса Piper рядом с выбранным (*.onnx в той же папке)."""
    current = config.resolve(config.tts.piper_voice)
    folder = current.parent
    files = sorted(folder.glob("*.onnx")) if folder.is_dir() else []
    if current not in files:
        files.insert(0, current)
    voices = []
    for path in files:
        try:
            value = path.relative_to(config.base_dir).as_posix()
        except ValueError:
            value = str(path)
        parts = path.stem.split("-")  # ru_RU-dmitri-medium
        name = parts[1] if len(parts) >= 2 else path.stem
        title = _PIPER_NAMES.get(name, name.capitalize())
        voices.append((value, title if path.exists() else f"{title} (файла нет)"))
    return voices


def _choices(items: list[tuple[str, str]], current: str) -> list[dict[str, str]]:
    """Список для интерфейса; текущее значение есть в нём, даже если его не нашли."""
    result = [{"value": value, "title": title} for value, title in items]
    if current and current not in {value for value, _ in items}:
        result.insert(0, {"value": current, "title": current})
    return result


def profile_view(name: str, profile: LlmProfile, store: SettingsStore) -> dict[str, Any]:
    """Профиль нейросети для настроек: всё, кроме самого ключа."""
    info = provider_info(name, profile)
    custom = name == CUSTOM_PROFILE
    key_env = profile.api_key_env or (CUSTOM_KEY_ENV if custom else "")
    key = store.key(key_env)
    return {
        "name": name,
        "title": profile.title or name,
        "provider": profile.provider,
        "model": profile.model,
        "models": list(info.models),
        "base_url": profile.base_url,
        "base_url_editable": info.base_url_editable,
        "configured": True,
        "key_env": key_env,
        "key_set": bool(key),
        "key_hint": key_hint(key),
        "key_from_system": store.key_from_system(key_env),
        "key_optional": custom,
        "key_url": info.key_url,
        "key_example": info.key_example,
        "note": info.note,
    }


def custom_view(store: SettingsStore) -> dict[str, Any]:
    """Свой сервис, которого ещё нет в конфиге."""
    key = store.key(CUSTOM_KEY_ENV)
    info = provider_info(CUSTOM_PROFILE)
    return {
        "name": CUSTOM_PROFILE,
        "title": CUSTOM_TITLE,
        "provider": "openai",
        "model": "",
        "models": [],
        "base_url": "",
        "base_url_editable": True,
        "configured": False,
        "key_env": CUSTOM_KEY_ENV,
        "key_set": bool(key),
        "key_hint": key_hint(key),
        "key_from_system": store.key_from_system(CUSTOM_KEY_ENV),
        "key_optional": True,
        "key_url": "",
        "key_example": "",
        "note": info.note,
    }


def llm_profiles_view(config: AppConfig, store: SettingsStore) -> list[dict[str, Any]]:
    """Профили в порядке: встроенные, добавленные в config.yaml, свой сервис."""
    profiles = config.llm.profiles
    order = [name for name in DEFAULT_LLM_PROFILES if name in profiles]
    order += [name for name in profiles if name not in DEFAULT_LLM_PROFILES and name != CUSTOM_PROFILE]
    views = [profile_view(name, profiles[name], store) for name in order]
    views.append(
        profile_view(CUSTOM_PROFILE, profiles[CUSTOM_PROFILE], store)
        if CUSTOM_PROFILE in profiles
        else custom_view(store)
    )
    return views


def settings_view(config: AppConfig, store: SettingsStore, choices: Choices) -> dict[str, Any]:
    """Настройки в формате интерфейса (тип Settings в ui/src/lib/types.ts)."""
    general, voice, stt, tts = config.general, config.voice, config.stt, config.tts
    device = voice.input_device
    if isinstance(device, int):  # в config.yaml номер устройства — в интерфейсе его название
        device = next((name for index, name in choices.microphones if index == device), str(device))
    microphone = device or ""
    piper = config.resolve(tts.piper_voice)
    try:
        piper_value = piper.relative_to(config.base_dir).as_posix()
    except ValueError:
        piper_value = str(piper)
    return {
        "general": {"name": general.name, "dry_run": general.dry_run},
        "voice": {
            "enabled": voice.enabled,
            "wake_word": voice.wake_word,
            "push_to_talk": voice.push_to_talk,
            "push_to_talk_key": voice.push_to_talk_key,
            "push_to_talk_mode": voice.push_to_talk_mode,
            "stop_key": voice.stop_key,
            "input_device": microphone,
            "wake_confidence": voice.wake_confidence,
            "followup_seconds": voice.followup_seconds,
            "min_silence_ms": voice.min_silence_ms,
            "sounds": voice.sounds,
        },
        "stt": {"model": stt.model, "device": stt.device},
        "tts": {
            "engine": tts.engine,
            "piper_voice": piper_value,
            "silero_speaker": tts.silero_speaker,
            "edge_voice": tts.edge_voice,
            "sapi_voice": tts.sapi_voice,
            "rate": tts.rate,
            "volume": tts.volume,
        },
        "llm": {
            "enabled": config.llm.enabled,
            "profile": config.llm.profile,
            "profiles": llm_profiles_view(config, store),
        },
        "ui": ui_view(config),
        "choices": {
            "microphones": [
                {"value": "", "title": "Микрофон по умолчанию"},
                *_choices([(name, name) for _, name in choices.microphones], microphone),
            ],
            "piper_voices": _choices(choices.piper_voices or piper_voices(config), piper_value),
            "sapi_voices": [
                {"value": "", "title": "Первый русский голос"},
                *_choices(choices.sapi_voices, tts.sapi_voice),
            ],
            "silero_speakers": _choices(list(SILERO_SPEAKERS), tts.silero_speaker),
            "edge_voices": _choices(list(EDGE_VOICES), tts.edge_voice),
            "silero_ready": silero_ready(config),
            "stt_models": _choices(choices.stt_models, stt.model),
        },
    }


def silero_ready(config: AppConfig) -> bool:
    """Голос Silero уже скачан: PyTorch и модель есть — выбор сработает без загрузки."""
    from assistant.models import silero_runtime

    return silero_runtime()[0] and config.resolve(config.tts.silero_model).is_file()


def ui_view(config: AppConfig) -> dict[str, Any]:
    """Раздел ui для интерфейса."""
    return config.ui.model_dump(include={"theme", "accent", "overlay", "quick_chat_key"})
