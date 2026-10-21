"""Загрузка YAML-паков команд, их проверка по реестру действий и перезагрузка.

Формат команды:

    - id: volume_set
      phrases: ["громкость на {percent}", "поставь громкость {percent}"]
      slots: {percent: int}
      when: {active_window_contains: "YouTube"}   # необязательно
      confirm: false                              # необязательно
      reply: "Громкость {percent}"                # необязательно; {result} — итог последнего шага
      hotkey: "ctrl+alt+1"                        # необязательно, запуск по горячей клавише
      steps:
        - volume_set: {percent: "{percent}"}

Ошибочная команда пропускается с предупреждением в логе, остальные загружаются.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    field_validator,
)
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from assistant.actions.registry import ActionRegistry, ArgsError, UnknownActionError
from assistant.actions.winapi import WindowInfo

log = logging.getLogger(__name__)

# text — как str, но с пунктуацией исходной фразы (для диктовки: «напечатай Привет, мир!»).
SlotType = Literal["int", "float", "str", "text"]
PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")
# Значения-образцы для проверки шагов со слотами при загрузке.
_SAMPLE_VALUES: dict[str, Any] = {"int": 1, "float": 1.0, "str": "текст", "text": "текст"}


def _as_list(value: Any) -> Any:
    """Строку превращает в список из одной строки (для условий when)."""
    return [value] if isinstance(value, str) else value


StrList = Annotated[list[str] | None, BeforeValidator(_as_list)]


class WhenCondition(BaseModel):
    """Условие по активному окну. Все указанные поля должны выполняться;
    внутри поля достаточно совпадения с любым из значений списка."""

    model_config = ConfigDict(extra="forbid")

    active_window_contains: StrList = None
    active_process: StrList = None

    def matches(self, window: WindowInfo) -> bool:
        """Проверяет условие для окна."""
        if self.active_window_contains is not None:
            title = window.title.lower()
            if not any(part.lower() in title for part in self.active_window_contains):
                return False
        if self.active_process is not None:
            names = {window.process.lower(), window.process_stem.lower()} - {""}
            if not any(proc.lower() in names for proc in self.active_process):
                return False
        return True


class Step(BaseModel):
    """Шаг команды: имя действия и аргументы (могут содержать {слоты})."""

    action: str
    args: dict[str, Any] = Field(default_factory=dict)


class Command(BaseModel):
    """Команда из YAML."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    phrases: list[str] = Field(default_factory=list)
    slots: dict[str, SlotType] = Field(default_factory=dict)
    steps: list[Step] = Field(min_length=1)
    when: WhenCondition | None = None
    confirm: bool = False
    reply: str | None = None
    hotkey: str | None = None

    _source: str = PrivateAttr(default="")
    _line: int = PrivateAttr(default=0)

    @field_validator("steps", mode="before")
    @classmethod
    def _parse_steps(cls, value: Any) -> Any:
        """Превращает `- hotkey: {keys: "win+d"}` в Step(action="hotkey", args={...})."""
        if not isinstance(value, list):
            return value
        steps = []
        for item in value:
            if isinstance(item, Mapping) and len(item) == 1 and "action" not in item:
                [(name, args)] = item.items()
                steps.append({"action": name, "args": {} if args is None else args})
            else:
                steps.append(item)
        return steps

    @property
    def source(self) -> str:
        """Файл, из которого загружена команда."""
        return self._source

    @property
    def location(self) -> str:
        """«файл:строка» для сообщений об ошибках."""
        return f"{self._source}:{self._line}" if self._line else self._source

    def placeholders(self) -> set[str]:
        """Слоты, упомянутые во фразах."""
        return {name for phrase in self.phrases for name in PLACEHOLDER_RE.findall(phrase)}

    def slot_type(self, name: str) -> SlotType:
        """Тип слота; необъявленные слоты считаются строками."""
        return self.slots.get(name, "str")


def substitute(value: Any, slots: Mapping[str, Any]) -> Any:
    """Подставляет значения слотов в аргументы шага.

    Если строка целиком состоит из одного слота ("{percent}"), подставляется значение
    с его типом (число), иначе — текстовая подстановка ("-{delta}" → "-10").
    """
    if isinstance(value, str):
        whole = PLACEHOLDER_RE.fullmatch(value)
        if whole and whole.group(1) in slots:
            return slots[whole.group(1)]
        return PLACEHOLDER_RE.sub(
            lambda m: str(slots[m.group(1)]) if m.group(1) in slots else m.group(0), value
        )
    if isinstance(value, Mapping):
        return {key: substitute(item, slots) for key, item in value.items()}
    if isinstance(value, list):
        return [substitute(item, slots) for item in value]
    return value


def render_reply(template: str, slots: Mapping[str, Any], result: str) -> str:
    """Текст ответа: подставляет слоты и {result} (итог последнего шага)."""
    return str(substitute(template, {**slots, "result": result}))


@dataclass(frozen=True)
class LoadIssue:
    """Проблема при загрузке команд."""

    location: str
    message: str

    def __str__(self) -> str:
        return f"{self.location}: {self.message}"


@dataclass
class LoadResult:
    """Итог загрузки: команды и найденные проблемы."""

    commands: list[Command] = field(default_factory=list)
    issues: list[LoadIssue] = field(default_factory=list)


def _to_plain(node: Any) -> Any:
    """Превращает узлы ruamel (CommentedMap, ScalarString…) в обычные dict/list/str."""
    if isinstance(node, Mapping):
        return {str(key): _to_plain(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_to_plain(item) for item in node]
    if isinstance(node, str):
        return str(node)
    if isinstance(node, bool):
        return bool(node)
    if isinstance(node, int):
        return int(node)
    if isinstance(node, float):
        return float(node)
    return node


def _check_command(command: Command, registry: ActionRegistry) -> list[str]:
    """Проверяет шаги команды по реестру. Возвращает список ошибок."""
    errors = []
    used = command.placeholders()
    unknown_slots = set(command.slots) - used
    if unknown_slots:
        errors.append(f"слоты {sorted(unknown_slots)} объявлены, но не встречаются во фразах")
    if not command.phrases and not command.hotkey:
        errors.append("нет ни фраз, ни горячей клавиши — команду нечем запустить")
    samples = {name: _SAMPLE_VALUES[command.slot_type(name)] for name in used}
    for number, step in enumerate(command.steps, start=1):
        if step.action not in registry:
            errors.append(f"шаг {number}: неизвестное действие {step.action}")
            continue
        missing = {m for m in PLACEHOLDER_RE.findall(repr(step.args))} - used
        if missing:
            errors.append(f"шаг {number}: слоты {sorted(missing)} не встречаются во фразах")
            continue
        try:
            registry.get(step.action).parse_args(substitute(step.args, samples))
        except (ArgsError, UnknownActionError) as exc:
            errors.append(f"шаг {number}: {exc}")
    return errors


def load_file(path: Path, registry: ActionRegistry) -> LoadResult:
    """Загружает один YAML-пак."""
    result = LoadResult()
    try:
        data = YAML(typ="rt").load(path.read_text(encoding="utf-8"))
    except (OSError, YAMLError) as exc:
        result.issues.append(LoadIssue(path.name, f"не удалось прочитать файл: {exc}"))
        return result
    if data is None:
        return result
    if not isinstance(data, list):
        result.issues.append(LoadIssue(path.name, "файл должен содержать список команд"))
        return result

    for index, item in enumerate(data):
        line = data.lc.item(index)[0] + 1 if hasattr(data, "lc") else 0
        location = f"{path.name}:{line}"
        try:
            command = Command.model_validate(_to_plain(item))
        except ValidationError as exc:
            details = "; ".join(
                f"{'.'.join(str(p) for p in err['loc']) or 'команда'}: {err['msg']}" for err in exc.errors()
            )
            result.issues.append(LoadIssue(location, details))
            continue
        command._source, command._line = path.name, line
        errors = _check_command(command, registry)
        if errors:
            result.issues.append(LoadIssue(f"{location} ({command.id})", "; ".join(errors)))
            continue
        result.commands.append(command)
    return result


def pack_files(directory: Path, user_file: str) -> list[Path]:
    """Файлы паков по алфавиту; пользовательский пак — последним."""
    files = sorted(directory.glob("*.yaml"), key=lambda p: p.name.lower())
    user = [p for p in files if p.name == user_file]
    return [p for p in files if p.name != user_file] + user


def load_commands(directory: Path, user_file: str, registry: ActionRegistry) -> LoadResult:
    """Загружает все паки. Команда с повторным id перекрывает загруженную раньше."""
    result = LoadResult()
    by_id: dict[str, Command] = {}
    if not directory.is_dir():
        result.issues.append(LoadIssue(str(directory), "папка с командами не найдена"))
        return result
    for path in pack_files(directory, user_file):
        loaded = load_file(path, registry)
        result.issues.extend(loaded.issues)
        for command in loaded.commands:
            previous = by_id.get(command.id)
            if previous is not None:
                log.info("Команда %s из %s перекрывает %s", command.id, command.location, previous.location)
            by_id[command.id] = command
    result.commands = list(by_id.values())
    return result


class CommandStore:
    """Текущий набор команд с потокобезопасной перезагрузкой.

    Args:
        directory: папка с паками.
        user_file: имя пользовательского пака.
        registry: реестр для проверки шагов.
    """

    def __init__(self, directory: Path, user_file: str, registry: ActionRegistry) -> None:
        self.directory = directory
        self.user_file = user_file
        self._registry = registry
        self._commands: list[Command] = []
        self._lock = threading.Lock()

    @property
    def commands(self) -> list[Command]:
        """Снимок текущих команд."""
        with self._lock:
            return list(self._commands)

    def reload(self) -> LoadResult:
        """Перечитывает паки. Проблемы пишутся в лог и возвращаются вызывающему."""
        result = load_commands(self.directory, self.user_file, self._registry)
        for issue in result.issues:
            log.warning("Команда пропущена: %s", issue)
        with self._lock:
            self._commands = result.commands
        log.info("Загружено команд: %d (проблем: %d)", len(result.commands), len(result.issues))
        return result
