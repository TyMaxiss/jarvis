"""Выполнение действий: проверка аргументов, политика, подтверждение, журнал, dry-run."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from assistant.actions.registry import Action, ActionError, ActionRegistry
from assistant.journal import ActionJournal
from assistant.safety.confirm import Confirmer
from assistant.safety.policy import SafetyPolicy, Verdict

log = logging.getLogger(__name__)


class StepStatus(StrEnum):
    """Итог выполнения шага."""

    DONE = "done"
    FAILED = "failed"
    FORBIDDEN = "forbidden"
    CANCELLED = "cancelled"
    DRY_RUN = "dry_run"


@dataclass(frozen=True)
class PlannedStep:
    """Действие с уже проверенными аргументами."""

    action: Action
    args: BaseModel

    def describe(self) -> str:
        """Короткая запись вида volume_set(percent=50) для логов и вопросов."""
        values = ", ".join(f"{k}={v!r}" for k, v in self.args_dict().items())
        return f"{self.action.name}({values})"

    def args_dict(self) -> dict[str, Any]:
        """Аргументы в виде словаря (для журнала)."""
        return self.args.model_dump(mode="json")


@dataclass(frozen=True)
class StepResult:
    """Результат одного шага."""

    action: str
    args: dict[str, Any]
    status: StepStatus
    text: str
    duration_ms: float = 0.0

    @property
    def ok(self) -> bool:
        """Шаг выполнен (или имитирован в dry-run)."""
        return self.status in (StepStatus.DONE, StepStatus.DRY_RUN)


@dataclass
class RunResult:
    """Результат выполнения последовательности шагов."""

    steps: list[StepResult] = field(default_factory=list)
    status: StepStatus = StepStatus.DONE

    @property
    def ok(self) -> bool:
        """Все шаги выполнены."""
        return self.status in (StepStatus.DONE, StepStatus.DRY_RUN)

    @property
    def last_text(self) -> str:
        """Текст результата последнего шага (для reply «{result}»)."""
        return self.steps[-1].text if self.steps else ""


class ActionExecutor:
    """Выполняет действия из реестра с учётом политики безопасности.

    Args:
        registry: реестр действий.
        policy: политика безопасности.
        confirmer: способ спросить подтверждение у пользователя.
        journal: журнал действий.
        dry_run: только логировать действия, не выполняя их.
    """

    def __init__(
        self,
        registry: ActionRegistry,
        policy: SafetyPolicy,
        confirmer: Confirmer,
        journal: ActionJournal,
        *,
        dry_run: bool = False,
    ) -> None:
        self.registry = registry
        self.policy = policy
        self.confirmer = confirmer
        self.journal = journal
        self.dry_run = dry_run

    def plan(self, steps: Iterable[tuple[str, Mapping[str, Any] | None]]) -> list[PlannedStep]:
        """Проверяет имена действий и аргументы до начала выполнения.

        Raises:
            UnknownActionError: неизвестное действие.
            ArgsError: аргументы не подходят под схему.
        """
        planned = []
        for name, raw_args in steps:
            action = self.registry.get(name)
            planned.append(PlannedStep(action, action.parse_args(raw_args)))
        return planned

    def run(
        self,
        planned: list[PlannedStep],
        *,
        source: str,
        command_id: str | None = None,
        phrase: str | None = None,
        require_confirm: bool = False,
        question: str | None = None,
    ) -> RunResult:
        """Выполняет шаги по порядку; при первой ошибке останавливается.

        Перед выполнением проверяется политика для всех шагов: если хоть один запрещён,
        не выполняется ничего; если хоть один требует подтверждения, вопрос задаётся
        один раз на всю последовательность.

        Args:
            planned: шаги из plan().
            source: откуда пришла команда: local, llm, hotkey…
            command_id: id YAML-команды (для журнала).
            phrase: исходная фраза пользователя (для журнала).
            require_confirm: команда сама требует подтверждения (confirm: true в YAML).
            question: текст вопроса; по умолчанию собирается из описаний шагов.
        """
        context = {"source": source, "command_id": command_id, "phrase": phrase}
        decisions = [self.policy.evaluate(step.action, step.args) for step in planned]

        forbidden = [(s, d) for s, d in zip(planned, decisions, strict=True) if d.verdict is Verdict.FORBID]
        if forbidden:
            step, decision = forbidden[0]
            text = f"Действие {step.action.name} запрещено: {decision.reason}"
            log.warning("%s (%s)", text, step.describe())
            self._journal(step, StepStatus.FORBIDDEN, text, 0.0, context)
            return RunResult([StepResult(step.action.name, step.args_dict(), StepStatus.FORBIDDEN, text)],
                             StepStatus.FORBIDDEN)  # fmt: skip

        needs_confirm = require_confirm or any(d.verdict is Verdict.CONFIRM for d in decisions)
        if needs_confirm and not self.dry_run:
            reasons = {d.reason for d in decisions if d.reason}
            text = question or "Выполнить: " + "; ".join(s.action.description for s in planned)
            if reasons:
                text += f" ({', '.join(sorted(reasons))})"
            if not self.confirmer.confirm(text):
                log.info("Пользователь не подтвердил: %s", text)
                for step in planned:
                    self._journal(step, StepStatus.CANCELLED, "не подтверждено", 0.0, context)
                return RunResult([], StepStatus.CANCELLED)

        result = RunResult()
        for step in planned:
            step_result = self._run_step(step, context)
            result.steps.append(step_result)
            if not step_result.ok:
                result.status = step_result.status
                break
        else:
            result.status = StepStatus.DRY_RUN if self.dry_run else StepStatus.DONE
        return result

    def _run_step(self, step: PlannedStep, context: dict[str, Any]) -> StepResult:
        """Выполняет один шаг (или имитирует его в dry-run) и пишет журнал."""
        if self.dry_run:
            text = f"[dry-run] {step.describe()}"
            log.info(text)
            self._journal(step, StepStatus.DRY_RUN, text, 0.0, context)
            return StepResult(step.action.name, step.args_dict(), StepStatus.DRY_RUN, text)

        started = time.perf_counter()
        try:
            text = step.action.run(step.args)
            status = StepStatus.DONE
        except ActionError as exc:
            text, status = str(exc), StepStatus.FAILED
            log.warning("Действие %s не выполнено: %s", step.describe(), text)
        except Exception:
            log.exception("Ошибка в действии %s", step.describe())
            text, status = f"Не получилось выполнить {step.action.name}", StepStatus.FAILED
        duration = (time.perf_counter() - started) * 1000
        log.info("%s → %s (%.0f мс)", step.describe(), text, duration)
        self._journal(step, status, text, duration, context)
        return StepResult(step.action.name, step.args_dict(), status, text, duration)

    def _journal(
        self,
        step: PlannedStep,
        status: StepStatus,
        text: str,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        self.journal.write(
            source=context["source"],
            action=step.action.name,
            args=step.args_dict(),
            status=status.value,
            result=text,
            duration_ms=duration_ms,
            command_id=context["command_id"],
            phrase=context["phrase"],
        )
