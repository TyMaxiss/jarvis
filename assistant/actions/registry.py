"""Реестр действий — единственный источник истины и для YAML-команд, и для LLM.

Действие — это обычная функция с аннотациями типов. Схема параметров (pydantic) и
JSON-схема инструмента для LLM строятся из её сигнатуры автоматически:

    @registry.action("volume_set", "Установить громкость в процентах")
    def volume_set(percent: Annotated[int, Field(ge=0, le=100)]) -> str:
        ...
        return "Громкость 50%"
"""

from __future__ import annotations

import inspect
import typing
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError, create_model

F = TypeVar("F", bound=Callable[..., str])


class Risk(StrEnum):
    """Уровень риска действия."""

    SAFE = "safe"
    CONFIRM = "confirm"
    FORBIDDEN = "forbidden"


class ActionError(Exception):
    """Действие не выполнено. Текст понятен пользователю и пригоден для озвучки."""


class UnknownActionError(KeyError):
    """Действия с таким именем нет в реестре."""

    def __str__(self) -> str:
        return f"Неизвестное действие: {self.args[0]}"


class ArgsError(ValueError):
    """Аргументы действия не прошли проверку схемы."""


# Русские тексты для частых ошибок pydantic; {ключи} берутся из ctx ошибки.
_ERROR_TEXTS = {
    "missing": "не указан",
    "extra_forbidden": "лишний параметр",
    "less_than_equal": "должно быть не больше {le}",
    "less_than": "должно быть меньше {lt}",
    "greater_than_equal": "должно быть не меньше {ge}",
    "greater_than": "должно быть больше {gt}",
    "int_parsing": "нужно целое число",
    "int_from_float": "нужно целое число",
    "float_parsing": "нужно число",
    "bool_parsing": "нужно true или false",
    "string_too_short": "пустое значение",
    "string_too_long": "слишком длинный текст",
    "literal_error": "допустимые значения: {expected}",
    "string_type": "нужна строка",
}


def _format_args_error(action: str, error: ValidationError) -> str:
    """Превращает ошибку pydantic в короткий русский текст."""
    parts = []
    for item in error.errors():
        where = ".".join(str(p) for p in item["loc"]) or "аргументы"
        template = _ERROR_TEXTS.get(item["type"])
        try:
            text = template.format(**item.get("ctx", {})) if template else item["msg"]
        except (KeyError, IndexError):
            text = item["msg"]
        parts.append(f"{where} — {text}")
    return f"{action}: " + "; ".join(parts)


# Ключи схемы, в которых лежат данные, а не схемы: их содержимое не меняется.
_SCHEMA_DATA_KEYS = frozenset({"default", "enum", "const", "examples"})


def _clean_schema(node: Any) -> Any:
    """Упрощает JSON-схему pydantic для LLM: убирает title и «| None» из anyOf.

    title — ключевое слово, которое pydantic пишет в каждый узел схемы. Параметр с именем
    title (ключ внутри properties) остаётся: иначе у focus_window пропадал бы параметр.
    """
    if isinstance(node, list):
        return [_clean_schema(item) for item in node]
    if not isinstance(node, dict):
        return node
    any_of = node.get("anyOf")
    if isinstance(any_of, list):
        non_null = [item for item in any_of if item != {"type": "null"}]
        if len(non_null) == 1:
            merged = {k: v for k, v in node.items() if k != "anyOf"}
            merged.update(non_null[0])
            node = merged
    cleaned: dict[str, Any] = {}
    for key, value in node.items():
        if key == "title":
            continue
        if key in ("properties", "$defs") and isinstance(value, dict):
            cleaned[key] = {name: _clean_schema(schema) for name, schema in value.items()}
        elif key in _SCHEMA_DATA_KEYS:
            cleaned[key] = value
        else:
            cleaned[key] = _clean_schema(value)
    return cleaned


def build_params_model(func: Callable[..., Any], name: str) -> type[BaseModel]:
    """Строит pydantic-модель параметров по сигнатуре функции.

    Значения по умолчанию становятся необязательными параметрами, описания берутся из
    Annotated[..., Field(description=...)].
    """
    hints = typing.get_type_hints(func, include_extras=True)
    fields: dict[str, Any] = {}
    for param in inspect.signature(func).parameters.values():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise TypeError(f"{name}: *args и **kwargs в действиях не поддерживаются")
        annotation = hints.get(param.name, Any)
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[param.name] = (annotation, default)
    # Числа в строковых параметрах допустимы: `say: {text: 1}` в YAML даёт "1".
    config = ConfigDict(extra="forbid", coerce_numbers_to_str=True)
    return create_model(f"{name}_args", __config__=config, **fields)


@dataclass(frozen=True)
class Action:
    """Описание действия в реестре."""

    name: str
    description: str
    params: type[BaseModel]
    risk: Risk
    func: Callable[..., str]

    def parse_args(self, raw: Mapping[str, Any] | None) -> BaseModel:
        """Проверяет и приводит аргументы к типам схемы («50» → 50).

        Raises:
            ArgsError: аргументы не подходят под схему.
        """
        try:
            return self.params.model_validate(dict(raw or {}))
        except ValidationError as exc:
            raise ArgsError(_format_args_error(self.name, exc)) from exc

    def run(self, args: BaseModel) -> str:
        """Выполняет действие с уже проверенными аргументами."""
        kwargs = {field: getattr(args, field) for field in type(args).model_fields}
        return str(self.func(**kwargs))

    def tool_schema(self) -> dict[str, Any]:
        """Описание действия в формате tools для OpenAI-совместимого API."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": _clean_schema(self.params.model_json_schema()),
            },
        }


class ActionRegistry:
    """Набор доступных действий."""

    def __init__(self) -> None:
        self._actions: dict[str, Action] = {}

    def action(self, name: str, description: str, risk: Risk = Risk.SAFE) -> Callable[[F], F]:
        """Декоратор: регистрирует функцию как действие."""

        def decorator(func: F) -> F:
            params = build_params_model(func, name)
            self.add(Action(name=name, description=description, params=params, risk=risk, func=func))
            return func

        return decorator

    def add(self, action: Action) -> None:
        """Добавляет действие; повторное имя — ошибка программиста."""
        if action.name in self._actions:
            raise ValueError(f"Действие {action.name} уже зарегистрировано")
        self._actions[action.name] = action

    def get(self, name: str) -> Action:
        """Возвращает действие по имени.

        Raises:
            UnknownActionError: такого действия нет.
        """
        try:
            return self._actions[name]
        except KeyError:
            raise UnknownActionError(name) from None

    def __contains__(self, name: object) -> bool:
        return name in self._actions

    def __iter__(self) -> Iterator[Action]:
        return iter(self._actions.values())

    def __len__(self) -> int:
        return len(self._actions)

    def names(self) -> list[str]:
        """Имена всех действий."""
        return list(self._actions)

    def tools(self, exclude: Iterable[str] = ()) -> list[dict[str, Any]]:
        """JSON-схемы всех действий для LLM (запрещённые не показываются)."""
        skip = set(exclude)
        return [
            action.tool_schema()
            for action in self._actions.values()
            if action.risk is not Risk.FORBIDDEN and action.name not in skip
        ]
