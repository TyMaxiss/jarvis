"""Тесты исполнителя действий: политика, подтверждение, dry-run, журнал."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import pytest
from pydantic import Field

from assistant.actions.executor import ActionExecutor, StepStatus
from assistant.actions.registry import ActionError, ActionRegistry, Risk
from assistant.journal import ActionJournal
from assistant.logging_setup import SecretMasker
from assistant.safety.confirm import StaticConfirmer
from assistant.safety.policy import SafetyPolicy, Verdict


class Calls:
    def __init__(self) -> None:
        self.log: list[str] = []


@pytest.fixture
def calls() -> Calls:
    return Calls()


@pytest.fixture
def reg(calls: Calls) -> ActionRegistry:
    reg = ActionRegistry()

    @reg.action("say", "Сказать")
    def say(text: str) -> str:
        calls.log.append(f"say:{text}")
        return f"сказал {text}"

    @reg.action("shutdown", "Выключить компьютер", risk=Risk.CONFIRM)
    def shutdown() -> str:
        calls.log.append("shutdown")
        return "выключаю"

    @reg.action("format_disk", "Отформатировать диск", risk=Risk.FORBIDDEN)
    def format_disk() -> str:
        calls.log.append("format")
        return "?"

    @reg.action("broken", "Всегда падает")
    def broken() -> str:
        raise ActionError("Не нашёл приложение")

    @reg.action("crash", "Внутренняя ошибка")
    def crash() -> str:
        raise RuntimeError("bug")

    @reg.action("close_app", "Закрыть программу")
    def close_app(name: str, force: Annotated[bool, Field()] = False) -> str:
        calls.log.append(f"close:{name}:{force}")
        return "закрыл"

    return reg


def make_executor(
    reg: ActionRegistry, tmp_path: Path, answer: bool = True, dry_run: bool = False
) -> tuple[ActionExecutor, StaticConfirmer, Path]:
    confirmer = StaticConfirmer(answer)
    journal_path = tmp_path / "actions.jsonl"
    journal = ActionJournal(journal_path, SecretMasker(["sk-secret-value-1234567890"]))
    return ActionExecutor(reg, SafetyPolicy(), confirmer, journal, dry_run=dry_run), confirmer, journal_path


def journal_entries(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_safe_steps_run_in_order(reg: ActionRegistry, calls: Calls, tmp_path: Path) -> None:
    executor, confirmer, journal = make_executor(reg, tmp_path)
    planned = executor.plan([("say", {"text": "раз"}), ("say", {"text": "два"})])
    result = executor.run(planned, source="local", command_id="test", phrase="скажи")
    assert result.status is StepStatus.DONE
    assert calls.log == ["say:раз", "say:два"]
    assert confirmer.questions == []
    entries = journal_entries(journal)
    assert [e["args"] for e in entries] == [{"text": "раз"}, {"text": "два"}]
    assert entries[0]["source"] == "local" and entries[0]["command"] == "test"
    assert entries[0]["phrase"] == "скажи" and entries[0]["status"] == "done"
    assert set(entries[0]) >= {"time", "action", "result", "duration_ms"}


def test_confirm_risk_asks_once(reg: ActionRegistry, calls: Calls, tmp_path: Path) -> None:
    executor, confirmer, _ = make_executor(reg, tmp_path, answer=True)
    planned = executor.plan([("say", {"text": "пока"}), ("shutdown", {})])
    result = executor.run(planned, source="local")
    assert result.ok
    assert len(confirmer.questions) == 1
    assert calls.log == ["say:пока", "shutdown"]


def test_declined_confirmation_runs_nothing(reg: ActionRegistry, calls: Calls, tmp_path: Path) -> None:
    executor, _, journal = make_executor(reg, tmp_path, answer=False)
    result = executor.run(executor.plan([("say", {"text": "x"}), ("shutdown", {})]), source="local")
    assert result.status is StepStatus.CANCELLED
    assert calls.log == []
    assert {e["status"] for e in journal_entries(journal)} == {"cancelled"}


def test_command_level_confirm(reg: ActionRegistry, calls: Calls, tmp_path: Path) -> None:
    executor, confirmer, _ = make_executor(reg, tmp_path, answer=False)
    result = executor.run(executor.plan([("say", {"text": "x"})]), source="local", require_confirm=True,
                          question="Выполнить «тест»?")  # fmt: skip
    assert result.status is StepStatus.CANCELLED
    assert confirmer.questions == ["Выполнить «тест»?"]


def test_forbidden_runs_nothing(reg: ActionRegistry, calls: Calls, tmp_path: Path) -> None:
    executor, confirmer, journal = make_executor(reg, tmp_path)
    result = executor.run(executor.plan([("say", {"text": "x"}), ("format_disk", {})]), source="llm")
    assert result.status is StepStatus.FORBIDDEN
    assert calls.log == [] and confirmer.questions == []
    assert journal_entries(journal)[0]["status"] == "forbidden"


def test_argument_rule_requires_confirmation(reg: ActionRegistry, calls: Calls, tmp_path: Path) -> None:
    executor, confirmer, _ = make_executor(reg, tmp_path, answer=True)
    executor.run(executor.plan([("close_app", {"name": "x"})]), source="local")
    assert confirmer.questions == []
    executor.run(executor.plan([("close_app", {"name": "x", "force": True})]), source="local")
    assert len(confirmer.questions) == 1
    assert "принудительное" in confirmer.questions[0]


def test_failure_stops_sequence(reg: ActionRegistry, calls: Calls, tmp_path: Path) -> None:
    executor, _, _ = make_executor(reg, tmp_path)
    result = executor.run(executor.plan([("broken", {}), ("say", {"text": "после"})]), source="local")
    assert result.status is StepStatus.FAILED
    assert result.last_text == "Не нашёл приложение"
    assert calls.log == []


def test_unexpected_exception_is_contained(reg: ActionRegistry, tmp_path: Path) -> None:
    executor, _, _ = make_executor(reg, tmp_path)
    result = executor.run(executor.plan([("crash", {})]), source="local")
    assert result.status is StepStatus.FAILED
    assert "crash" in result.last_text


def test_dry_run_only_logs(reg: ActionRegistry, calls: Calls, tmp_path: Path) -> None:
    executor, confirmer, journal = make_executor(reg, tmp_path, dry_run=True)
    result = executor.run(executor.plan([("say", {"text": "x"}), ("shutdown", {})]), source="local")
    assert result.status is StepStatus.DRY_RUN and result.ok
    assert calls.log == [] and confirmer.questions == []
    assert [e["status"] for e in journal_entries(journal)] == ["dry_run", "dry_run"]


def test_secrets_never_reach_journal(reg: ActionRegistry, tmp_path: Path) -> None:
    executor, _, journal = make_executor(reg, tmp_path)
    executor.run(executor.plan([("say", {"text": "ключ sk-secret-value-1234567890"})]), source="llm")
    text = journal.read_text(encoding="utf-8")
    assert "sk-secret-value" not in text and "***" in text


def test_policy_verdicts(reg: ActionRegistry) -> None:
    policy = SafetyPolicy(forbidden_actions=["say"])
    say = reg.get("say")
    assert policy.evaluate(say, say.parse_args({"text": "x"})).verdict is Verdict.FORBID
    shutdown = reg.get("shutdown")
    assert SafetyPolicy().evaluate(shutdown, shutdown.parse_args({})).verdict is Verdict.CONFIRM
