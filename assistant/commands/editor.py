"""Редактор команд для раздела «Команды»: дерево паков, команда для формы, сохранение в YAML.

Свои команды живут в пользовательском паке (commands/my.yaml). Остальные паки редактор
не меняет: изменённая команда из них сохраняется копией в my.yaml с тем же id и заменяет
исходную — так устроен загрузчик. При сохранении в файле меняется только текст этой
команды: ruamel.yaml (round-trip) обновляет её узлы вместе с комментариями, а остальной
файл — другие команды, комментарии, переносы строк — остаётся байт в байт.
"""

from __future__ import annotations

import io
import logging
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.error import YAMLError
from ruamel.yaml.scalarstring import DoubleQuotedScalarString

from assistant.actions.registry import ActionRegistry, Risk
from assistant.actions.winapi import EMPTY_WINDOW, WindowInfo
from assistant.commands.loader import (
    PLACEHOLDER_RE,
    Command,
    CommandStore,
    check_command,
    pack_files,
    parse_steps,
    to_plain,
)
from assistant.nlu.matcher import Matcher

log = logging.getLogger(__name__)

ID_RE = re.compile(r"^[\w-]+$")
# Порядок полей команды в YAML — как в README и встроенных паках.
KEY_ORDER = ("id", "phrases", "slots", "when", "confirm", "reply", "hotkey", "steps")
WHEN_KEYS = ("active_window_contains", "active_process")
# Группы действий в списке «+ Действие» — по модулю, где действие объявлено.
ACTION_GROUPS = {
    "apps": "Программы",
    "windows": "Окна",
    "input": "Клавиатура и мышь",
    "volume": "Звук",
    "media": "Музыка и видео",
    "browser": "Браузер и поиск",
    "files": "Файлы и папки",
    "system": "Разное",
    "power": "Питание",
    "shell": "PowerShell",
}
_OTHER_GROUP = "Другое"
_YAML_WIDTH = 4096  # длинные строки не переносятся: перенос внутри кавычек портит текст


class EditorError(Exception):
    """Команду не сохранить или не выполнить; errors — понятные строки для окна."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


# --- Чтение и запись паков ---------------------------------------------------------------


def _yaml() -> YAML:
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.width = _YAML_WIDTH
    return yaml


@dataclass
class Pack:
    """Пак, прочитанный для правки: текст файла и список команд (None — только комментарии)."""

    path: Path
    text: str
    root: CommentedSeq | None

    @property
    def nodes(self) -> list[Any]:
        return list(self.root) if self.root is not None else []

    def find(self, command_id: str) -> int | None:
        """Номер команды с этим id."""
        return next((i for i, node in enumerate(self.nodes) if _node_id(node) == command_id), None)

    def span(self, index: int) -> tuple[int, int]:
        """Строки команды в файле: [начало, начало следующей) — с её хвостовыми комментариями."""
        assert self.root is not None
        start = self.root.lc.item(index)[0]
        end = self.root.lc.item(index + 1)[0] if index + 1 < len(self.root) else len(self.lines)
        return start, end

    @property
    def lines(self) -> list[str]:
        return self.text.splitlines(keepends=True)

    @property
    def newline(self) -> str:
        return "\r\n" if "\r\n" in self.text else "\n"


def read_pack(path: Path) -> Pack:
    """Читает пак для правки.

    Raises:
        EditorError: файл не читается или в нём не список команд.
    """
    try:
        text = ""
        if path.exists():
            with path.open(encoding="utf-8-sig", newline="") as file:  # переводы строк как в файле
                text = file.read()
    except OSError as exc:
        raise EditorError([f"Не удалось прочитать {path.name}: {exc}"]) from exc
    try:
        # ruamel оставляет \r в комментариях; число строк от замены \r\n не меняется.
        root = _yaml().load(text.replace("\r\n", "\n")) if text.strip() else None
    except YAMLError as exc:
        raise EditorError([f"{path.name} не читается, исправьте его вручную: {exc}"]) from exc
    if root is not None and not isinstance(root, CommentedSeq):
        raise EditorError([f"{path.name}: на верхнем уровне должен быть список команд"])
    if root is not None and root.fa.flow_style():
        raise EditorError([f"{path.name}: список команд записан в одну строку, редактор его не меняет"])
    return Pack(path, text, root)


def write_text(path: Path, text: str) -> None:
    """Записывает файл целиком: сначала во временный, потом подменой — без полузаписанного файла."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        file.write(text)
    os.replace(temporary, path)


def dump_item(node: CommentedMap, newline: str = "\n") -> str:
    """Текст одной команды в стиле паков: «- id: …», шаги с отступом 4."""
    out = io.StringIO()
    _yaml().dump(CommentedSeq([node]), out)
    # Отступ списка ruamel ставит и корню: сдвигаем всё на 2 пробела обратно.
    lines = out.getvalue().splitlines(keepends=True)
    text = "".join(line[2:] if line.startswith("  ") else line for line in lines)
    return text.replace("\n", newline) if newline != "\n" else text


def _with_tail(body: str, original: str, newline: str) -> str:
    """Новый текст команды с теми же пустыми строками в конце, что были в файле."""
    content = original.rstrip("\r\n")
    return body.rstrip("\r\n") + (original[len(content) :] or newline)


def keep_original_lines(new: str, original: str) -> str:
    """Строки, не изменившиеся по сути, берутся из файла как были.

    ruamel выводит комментарии и пробелы перед ними по-своему; так у неизменённых строк
    остаются прежние отступы комментариев и выравнивание.
    """
    known: dict[str, str] = {}
    for line in original.splitlines(keepends=True):
        known.setdefault(" ".join(line.split()), line)
    return "".join(known.get(" ".join(line.split()), line) for line in new.splitlines(keepends=True))


def _node_id(node: Any) -> str:
    return str(node.get("id") or "") if isinstance(node, Mapping) else ""


# --- Данные команды ----------------------------------------------------------------------


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value] if isinstance(value, list) else []


def form_data(plain: Mapping[str, Any]) -> dict[str, Any]:
    """Команда из YAML → данные формы (тип CommandForm в ui/src/lib/types.ts)."""
    when = plain.get("when") if isinstance(plain.get("when"), Mapping) else {}
    slots = plain.get("slots") if isinstance(plain.get("slots"), Mapping) else {}
    steps = parse_steps(plain.get("steps"))
    return {
        "id": str(plain.get("id") or ""),
        "phrases": _as_list(plain.get("phrases")),
        "slots": {str(name): str(kind) for name, kind in slots.items()},
        "when": {key: _as_list(when.get(key)) for key in WHEN_KEYS},
        "confirm": bool(plain.get("confirm")),
        "reply": str(plain.get("reply") or ""),
        "hotkey": str(plain.get("hotkey") or ""),
        "steps": [
            {"action": str(step.get("action") or ""), "args": dict(step.get("args") or {})}
            for step in (steps if isinstance(steps, list) else [])
            if isinstance(step, Mapping)
        ],
    }


def plain_command(form: Mapping[str, Any], registry: ActionRegistry | None = None) -> dict[str, Any]:
    """Данные формы → команда для YAML: пустые поля опускаются, числа из полей — числами."""
    data: dict[str, Any] = {"id": str(form.get("id") or "").strip()}
    phrases = [" ".join(str(p).split()) for p in form.get("phrases") or [] if str(p).strip()]
    if phrases:
        data["phrases"] = phrases
    used = {name for phrase in phrases for name in PLACEHOLDER_RE.findall(phrase)}
    slots = form.get("slots") if isinstance(form.get("slots"), Mapping) else {}
    # str — тип слота по умолчанию, его не пишем; слоты, которых нет во фразах, — тоже.
    declared = {str(k): str(v) for k, v in slots.items() if str(k) in used and str(v) != "str"}
    if declared:
        data["slots"] = declared
    when_form = form.get("when") if isinstance(form.get("when"), Mapping) else {}
    when = {
        key: values
        for key in WHEN_KEYS
        if (values := [v.strip() for v in _as_list(when_form.get(key)) if v.strip()])
    }
    if when:
        data["when"] = {key: values[0] if len(values) == 1 else values for key, values in when.items()}
    if form.get("confirm"):
        data["confirm"] = True
    for key in ("reply", "hotkey"):
        value = str(form.get(key) or "").strip()
        if value:
            data[key] = value
    steps = []
    for step in form.get("steps") or []:
        if not isinstance(step, Mapping):
            continue
        action = str(step.get("action") or "").strip()
        raw = step.get("args") if isinstance(step.get("args"), Mapping) else {}
        args = {str(k): v for k, v in raw.items() if v is not None and v != ""}
        steps.append({"action": action, "args": _typed_args(registry, action, args)})
    data["steps"] = steps
    return data


def _typed_args(registry: ActionRegistry | None, action: str, args: dict[str, Any]) -> dict[str, Any]:
    """«30» в числовом параметре → 30 (в YAML без кавычек); слоты «{percent}» остаются строкой."""
    if registry is None or action not in registry:
        return args
    properties = registry.get(action).parameters_schema().get("properties", {})
    typed = {}
    for name, value in args.items():
        kind = properties.get(name, {}).get("type")
        if isinstance(value, str) and kind in ("integer", "number"):
            try:
                number = float(value.replace(",", "."))
                value = int(number) if kind == "integer" and number.is_integer() else number
            except ValueError:
                pass
        typed[name] = value
    return typed


def _steps_plain(value: Any) -> list[dict[str, Any]]:
    steps = parse_steps(value)
    return [
        {"action": str(s.get("action")), "args": dict(s.get("args") or {})}
        for s in (steps if isinstance(steps, list) else [])
        if isinstance(s, Mapping)
    ]


def _when_plain(value: Any) -> dict[str, list[str]]:
    when = value if isinstance(value, Mapping) else {}
    return {key: _as_list(when.get(key)) for key in WHEN_KEYS if _as_list(when.get(key))}


# --- Узлы ruamel -------------------------------------------------------------------------


def _quoted(value: Any) -> Any:
    """Значение для YAML: строки в двойных кавычках, списки и словари — в одну строку."""
    if isinstance(value, str):
        return DoubleQuotedScalarString(value)
    if isinstance(value, Mapping):
        return _flow_map(value)
    if isinstance(value, list):
        seq = CommentedSeq([_quoted(item) for item in value])
        seq.fa.set_flow_style()
        return seq
    return value


def _flow_map(values: Mapping[str, Any], quote: bool = True) -> CommentedMap:
    node = CommentedMap()
    for key, value in values.items():
        node[key] = _quoted(value) if quote else value
    node.fa.set_flow_style()
    return node


def _step_node(step: Mapping[str, Any]) -> CommentedMap:
    node = CommentedMap()
    node[step["action"]] = _flow_map(step["args"])
    return node


def _field_node(key: str, value: Any) -> Any:
    if key == "id":
        return value
    if key == "slots":
        return _flow_map(value, quote=False)  # {percent: int}
    if key == "confirm":
        return True
    return _quoted(value)


def _same(key: str, old: Any, new: Any) -> bool:
    if key == "steps":
        return _steps_plain(old) == new
    if key == "when":
        return _when_plain(old) == _when_plain(new)
    if key == "phrases":
        return _as_list(old) == new
    return old == new


def update_node(node: CommentedMap, data: Mapping[str, Any]) -> None:
    """Меняет в узле команды только изменившиеся поля: у остальных комментарии на месте."""
    old = to_plain(node)
    for key in KEY_ORDER:
        if key not in data:
            if key in node:
                del node[key]
            continue
        value = data[key]
        if key in node and _same(key, old.get(key), value):
            continue
        if key == "steps":
            new: Any = _steps_node(node.get("steps"), value)
        else:
            new = _field_node(key, value)
        if key in node:
            node[key] = new
        else:
            node.insert(_position(node, key), key, new)


def _steps_node(current: Any, steps: list[dict[str, Any]]) -> CommentedSeq:
    """Шаги: если число шагов то же, неизменённые узлы (с комментариями) остаются."""
    if isinstance(current, CommentedSeq) and len(current) == len(steps):
        old = _steps_plain(to_plain(current))
        for index, step in enumerate(steps):
            if old[index] != step:
                current[index] = _step_node(step)
        return current
    return CommentedSeq([_step_node(step) for step in steps])


def _position(node: CommentedMap, key: str) -> int:
    """Куда вставить новое поле, чтобы порядок был как в KEY_ORDER."""
    rank = KEY_ORDER.index(key)
    for position, existing in enumerate(node):
        if existing in KEY_ORDER and KEY_ORDER.index(existing) > rank:
            return position
    return len(node)


# --- Редактор ----------------------------------------------------------------------------


def window_for(command: Command) -> WindowInfo:
    """Окно, в котором условие when команды выполнено (для проверки фразы в редакторе)."""
    if command.when is None:
        return EMPTY_WINDOW
    titles = command.when.active_window_contains or [""]
    processes = command.when.active_process or [""]
    return WindowInfo(0, titles[0], "", 0, processes[0])


def action_catalog(registry: ActionRegistry) -> list[dict[str, Any]]:
    """Действия для «+ Действие»: по группам, с JSON-схемой параметров для формы шага."""
    groups = [*ACTION_GROUPS.values(), _OTHER_GROUP]
    items = [
        {
            "name": action.name,
            "description": action.description,
            "risk": action.risk.value,
            "group": ACTION_GROUPS.get(action.func.__module__.rsplit(".", 1)[-1], _OTHER_GROUP),
            "schema": action.parameters_schema(),
        }
        for action in registry
        if action.risk is not Risk.FORBIDDEN
    ]
    return sorted(items, key=lambda item: groups.index(item["group"]))


def _pydantic_errors(exc: ValidationError) -> list[str]:
    errors = []
    for item in exc.errors():
        where = [str(part) for part in item["loc"]]
        if where[:1] == ["steps"] and item["type"] == "too_short":
            errors.append("Добавьте хотя бы одно действие")
        elif where[:1] == ["slots"] and item["type"] == "literal_error":
            errors.append(f"Слот {where[1]}: тип int, float, str или text")
        elif where[:1] == ["steps"] and len(where) > 1:
            errors.append(f"Шаг {int(where[1]) + 1}: {item['msg']}")
        else:
            errors.append(f"{'.'.join(where) or 'команда'}: {item['msg']}")
    return errors


class CommandEditor:
    """Что умеет раздел «Команды».

    Args:
        store: загруженные команды (папка и пользовательский пак).
        registry: действия — для проверки шагов и форм.
        matcher: текущий матчер; для проверки фразы берётся его копия с черновиком команды.
    """

    def __init__(self, store: CommandStore, registry: ActionRegistry, matcher: Callable[[], Matcher]) -> None:
        self.store = store
        self.registry = registry
        self._matcher = matcher

    @property
    def user_file(self) -> str:
        return self.store.user_file

    # --- Чтение ----------------------------------------------------------------------

    def packs(self) -> list[Path]:
        """Паки: сначала свой (даже если его ещё нет), потом остальные по алфавиту."""
        files = [p for p in pack_files(self.store.directory, self.user_file) if p.name != self.user_file]
        return [self.store.directory / self.user_file, *files]

    def tree(self) -> list[dict[str, Any]]:
        """Дерево паков для левой колонки."""
        active = {command.id: command.source for command in self.store.commands}
        tree = []
        for path in self.packs():
            entry: dict[str, Any] = {
                "file": path.name,
                "user": path.name == self.user_file,
                "error": "",
                "commands": [],
            }
            try:
                pack = read_pack(path)
            except EditorError as exc:
                entry["error"] = exc.errors[0]
                tree.append(entry)
                continue
            for index, node in enumerate(pack.nodes):
                plain = to_plain(node)
                if not isinstance(plain, dict):
                    continue
                command_id = _node_id(plain)
                phrases = _as_list(plain.get("phrases"))
                owner = active.get(command_id, "")
                entry["commands"].append(
                    {
                        "id": command_id,
                        "title": phrases[0] if phrases else str(plain.get("hotkey") or command_id),
                        "phrases": len(phrases),
                        "steps": len(_steps_plain(plain.get("steps"))),
                        "hotkey": str(plain.get("hotkey") or ""),
                        "line": pack.root.lc.item(index)[0] + 1 if pack.root is not None else 0,
                        "active": owner == path.name,
                        "replaced_by": owner if owner and owner != path.name else "",
                        "problem": "; ".join(self.problems(plain)),
                    }
                )
            tree.append(entry)
        return tree

    def command(self, file: str, command_id: str) -> dict[str, Any]:
        """Команда для формы: данные, откуда она и что с ней не так.

        Raises:
            EditorError: нет такого пака или команды.
        """
        pack = read_pack(self._path(file))
        index = pack.find(command_id)
        if index is None:
            raise EditorError([f"В {file} нет команды «{command_id}»"])
        plain = to_plain(pack.nodes[index])
        replaced = [
            path.name for path in self.packs() if path.name != file and _has_command(path, command_id)
        ]
        return {
            "file": file,
            "user": file == self.user_file,
            "command": form_data(plain),
            "problems": self.problems(plain),
            "also_in": replaced,
        }

    def problems(self, plain: Mapping[str, Any]) -> list[str]:
        """Что не так с командой ('' — всё в порядке)."""
        try:
            self.validate(form_data(plain))
        except EditorError as exc:
            return exc.errors
        return []

    # --- Проверка и запись ----------------------------------------------------------

    def validate(self, form: Mapping[str, Any]) -> Command:
        """Команда из формы, проверенная как при загрузке.

        Raises:
            EditorError: понятные строки, что исправить.
        """
        data = plain_command(form, self.registry)
        errors = []
        if not data["id"]:
            errors.append("Укажите id команды")
        elif not ID_RE.match(data["id"]):
            errors.append("id: только буквы, цифры, «_» и «-»")
        try:
            command = Command.model_validate(data)
        except ValidationError as exc:
            raise EditorError(errors + _pydantic_errors(exc)) from exc
        errors += [error[:1].upper() + error[1:] for error in check_command(command, self.registry)]
        if errors:
            raise EditorError(errors)
        return command

    def save(self, file: str, original_id: str, form: Mapping[str, Any]) -> str:
        """Сохраняет команду в свой пак; возвращает её id.

        file и original_id — откуда команда открыта. Своя команда меняется на месте,
        команда другого пака сохраняется копией в свой пак.

        Raises:
            EditorError: команда с ошибками или свой пак не читается.
        """
        self.validate(form)
        data = plain_command(form, self.registry)
        pack = read_pack(self.store.directory / self.user_file)
        index = pack.find(original_id) if file == self.user_file and original_id else None
        clash = pack.find(data["id"])
        if clash is not None and clash != index:
            raise EditorError([f"В {self.user_file} уже есть команда «{data['id']}» — выберите другой id"])
        if index is None:
            node = CommentedMap()
            update_node(node, data)
            body = dump_item(node, pack.newline)
            text = pack.text.rstrip("\r\n")
            text = f"{text}{pack.newline * 2}{body}" if text else body
        else:
            node = pack.nodes[index]
            update_node(node, data)
            start, end = pack.span(index)
            lines = pack.lines
            original = "".join(lines[start:end])
            body = keep_original_lines(dump_item(node, pack.newline), original)
            text = "".join(lines[:start]) + _with_tail(body, original, pack.newline) + "".join(lines[end:])
        write_text(pack.path, text)
        log.info("Команда %s сохранена в %s", data["id"], self.user_file)
        return data["id"]

    def delete(self, file: str, command_id: str) -> None:
        """Удаляет свою команду (комментарии после неё остаются).

        Raises:
            EditorError: команда не из своего пака или её нет.
        """
        if file != self.user_file:
            raise EditorError([f"Удалять можно только свои команды ({self.user_file})"])
        pack = read_pack(self.store.directory / self.user_file)
        index = pack.find(command_id)
        if index is None:
            raise EditorError([f"В {file} нет команды «{command_id}»"])
        start, end = pack.span(index)
        lines = pack.lines
        last = max(i for i in range(start, end) if lines[i].strip() and not lines[i].lstrip().startswith("#"))
        rest = lines[last + 1 :]
        # Пустая строка, отделявшая команду от следующей, уходит вместе с командой.
        if rest and not rest[0].strip() and start > 0 and not lines[start - 1].strip():
            rest = rest[1:]
        write_text(pack.path, "".join(lines[:start] + rest))
        log.info("Команда %s удалена из %s", command_id, file)

    # --- Проверка фразы и запуск ------------------------------------------------------

    def test_phrase(self, text: str, form: Mapping[str, Any] | None, original_id: str = "") -> dict[str, Any]:
        """Какая команда сработает на фразу — с учётом несохранённого черновика.

        Черновик заменяет свою прежнюю версию; окно — такое, где его условие when выполнено.
        """
        draft: Command | None = None
        errors: list[str] = []
        if form is not None:
            try:
                draft = self.validate(form)
            except EditorError as exc:
                errors = exc.errors
        commands = self.store.commands
        if draft is not None:
            replaced = {draft.id, original_id}
            commands = [c for c in commands if c.id not in replaced] + [draft]
        window = window_for(draft) if draft is not None else EMPTY_WINDOW
        result = self._matcher().with_commands(commands).match(text, window)
        return {
            "status": result.status,
            "reason": result.reason,
            "normalized": result.normalized,
            "matches": [
                {
                    "id": m.command.id,
                    "score": round(m.score),
                    "phrase": m.phrase,
                    "text": m.text,
                    "slots": m.slots,
                    "draft": m.command is draft,
                }
                for m in result.matches
            ],
            "candidates": [
                {"id": c.command.id, "score": round(c.score), "phrase": c.phrase}
                for c in result.candidates[:5]
            ],
            "errors": errors,
        }

    def prepare_run(self, form: Mapping[str, Any], phrase: str = "") -> tuple[Command, dict[str, Any], str]:
        """Черновик команды для «Выполнить»: команда, значения слотов и фраза для журнала.

        Raises:
            EditorError: в команде ошибки, фраза к ней не подходит или у слотов нет значений.
        """
        command = self.validate(form)
        if phrase.strip():
            result = self._matcher().with_commands([command]).match(phrase, window_for(command))
            match = next((m for m in result.matches if m.command is command), None)
            if match is None:
                raise EditorError([f"Фраза «{phrase.strip()}» не подходит к этой команде"])
            return command, match.slots, phrase.strip()
        if command.placeholders():
            raise EditorError(
                ["У команды есть слоты: впишите фразу в «Проверить фразу» — значения возьмутся из неё"]
            )
        return command, {}, command.phrases[0] if command.phrases else command.id

    def _path(self, file: str) -> Path:
        """Путь к паку по имени файла из окна (только паки из папки команд)."""
        for path in self.packs():
            if path.name == file:
                return path
        raise EditorError([f"Нет пака {file}"])


def _has_command(path: Path, command_id: str) -> bool:
    try:
        return read_pack(path).find(command_id) is not None
    except EditorError:
        return False
