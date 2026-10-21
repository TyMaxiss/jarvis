"""Политика безопасности: можно ли выполнить действие с данными аргументами.

Вся логика допуска собрана здесь. Базовый уровень риска задаётся при регистрации
действия, а правила ниже могут повысить его в зависимости от аргументов.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel

from assistant.actions.registry import Action, Risk


class Verdict(StrEnum):
    """Решение политики."""

    ALLOW = "allow"
    CONFIRM = "confirm"
    FORBID = "forbid"


@dataclass(frozen=True)
class Decision:
    """Решение политики с пояснением для пользователя и журнала."""

    verdict: Verdict
    reason: str = ""


def _close_app_rule(args: BaseModel) -> Decision | None:
    """Принудительное завершение процесса может потерять несохранённые данные."""
    if getattr(args, "force", False):
        return Decision(Verdict.CONFIRM, "принудительное завершение программы")
    return None


# Правила, зависящие от аргументов: имя действия → функция, возвращающая решение или None.
_ARG_RULES: dict[str, Callable[[BaseModel], Decision | None]] = {
    "close_app": _close_app_rule,
}

_SEVERITY = {Verdict.ALLOW: 0, Verdict.CONFIRM: 1, Verdict.FORBID: 2}


class SafetyPolicy:
    """Оценивает действия перед выполнением.

    Args:
        forbidden_actions: имена действий, запрещённых настройками.
    """

    def __init__(self, forbidden_actions: Iterable[str] = ()) -> None:
        self._forbidden = frozenset(forbidden_actions)

    def evaluate(self, action: Action, args: BaseModel) -> Decision:
        """Возвращает самое строгое из решений: по уровню риска и по правилам аргументов."""
        if action.name in self._forbidden or action.risk is Risk.FORBIDDEN:
            return Decision(Verdict.FORBID, "действие запрещено")
        decision = Decision(Verdict.CONFIRM if action.risk is Risk.CONFIRM else Verdict.ALLOW)
        rule = _ARG_RULES.get(action.name)
        extra = rule(args) if rule else None
        if extra and _SEVERITY[extra.verdict] > _SEVERITY[decision.verdict]:
            return extra
        return decision
