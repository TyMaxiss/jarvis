"""Политика безопасности: можно ли выполнить действие с данными аргументами.

Вся логика допуска собрана здесь. Базовый уровень риска задаётся при регистрации
действия, а правила ниже могут повысить его в зависимости от аргументов.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel

from assistant.actions.files import resolve_user_path
from assistant.actions.registry import Action, ActionError, Risk


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


# Команды PowerShell и cmd, которые не выполняются даже с подтверждением. Список не может
# быть полным, поэтому любая другая команда run_shell всё равно требует подтверждения.
# [^|;&\n]* — в пределах одной команды конвейера.
SHELL_DENYLIST: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), what)
    for pattern, what in (
        (r"\bformat(\.com)?\s+[a-z]:", "форматирование диска"),
        (r"\b(format-volume|clear-disk|initialize-disk|remove-partition)\b", "разметка диска"),
        (r"\bdiskpart\b", "разметка дисков"),
        (r"\b(del|erase|rd|rmdir)\b[^|;&\n]*\s/s\b", "рекурсивное удаление"),
        (r"\b(remove-item|ri|rm|rmdir|rd|del|erase)\b[^|;&\n]*\s-r(ecurse)?\b", "рекурсивное удаление"),
        (r"\bdirectory\]::delete\b", "удаление папки"),
        (r"\breg(\.exe)?\s+delete\b", "удаление из реестра"),
        (r"\bremove-item(property)?\b[^|;&\n]*\bhk(lm|cu|cr|u|cc)\b", "удаление из реестра"),
        (r"\bbcdedit\b", "правка загрузчика"),
        (r"\bvssadmin\b", "теневые копии"),
        (r"\bwbadmin\b[^|;&\n]*\bdelete\b", "удаление резервных копий"),
        (r"\bcipher\b[^|;&\n]*\s/w\b", "затирание диска"),
        (r"\bsdelete\b", "безвозвратное удаление"),
        (r"\b(powershell|pwsh)(\.exe)?\b[^|;&\n]*\s-(e|ec|enc|encodedcommand)\b", "скрытая команда"),
    )
)


def _shell_rule(args: BaseModel) -> Decision | None:
    """Опасная команда запрещена; любая другая и так требует подтверждения (риск confirm)."""
    command = str(getattr(args, "command", ""))
    for pattern, what in SHELL_DENYLIST:
        if pattern.search(command):
            return Decision(Verdict.FORBID, f"опасная команда: {what}")
    return None


def _write_file_rule(args: BaseModel) -> Decision | None:
    """Перезапись существующего файла — только с подтверждения."""
    try:
        exists = resolve_user_path(str(getattr(args, "path", ""))).exists()
    except ActionError:
        return None  # путь пустой: действие само сообщит об ошибке
    return Decision(Verdict.CONFIRM, "файл уже существует и будет перезаписан") if exists else None


# Правила, зависящие от аргументов: имя действия → функция, возвращающая решение или None.
_ARG_RULES: dict[str, Callable[[BaseModel], Decision | None]] = {
    "close_app": _close_app_rule,
    "run_shell": _shell_rule,
    "write_file": _write_file_rule,
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
