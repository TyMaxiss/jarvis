"""Самопроверка сборки: итог по каждой части и код выхода."""

from __future__ import annotations

import io

from assistant.main import parse_args
from assistant.selfcheck import CHECKS, run_self_check


def test_self_check_reports_each_part() -> None:
    out = io.StringIO()

    def broken() -> str:
        raise OSError("нет libvosk.dll")

    assert run_self_check(out, {"Работает": lambda: "да", "Vosk": broken}) == 1
    assert out.getvalue().splitlines() == [
        "✓ Работает: да",
        "✗ Vosk: OSError: нет libvosk.dll",
        "Не работает: 1.",
    ]
    out = io.StringIO()
    assert run_self_check(out, {"Работает": lambda: "да"}) == 0
    assert out.getvalue().endswith("Всё на месте.\n")


def test_cross_platform_parts_load_here() -> None:
    # Нативные части, которые есть и не в Windows, загружаются и в тестах на Linux.
    out = io.StringIO()
    checks = {name: CHECKS[name] for name in ("Vosk (слово активации)", "Silero VAD", "Piper (espeak-ng)")}
    assert run_self_check(out, checks) == 0, out.getvalue()
    assert parse_args(["--self-check"]).self_check
