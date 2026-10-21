"""Тесты консоли: как показываются состояния голоса."""

from __future__ import annotations

import io
from typing import Any

from assistant.console import ConsoleApp
from tests.conftest import NO_WINDOW


def test_voice_states_are_shown() -> None:
    out = io.StringIO()
    unused: Any = None
    console = ConsoleApp(unused, unused, unused, unused, unused, lambda: NO_WINDOW, out=out)
    console.voice_status("listening")
    console.voice_status("recognized", "Сделай громче.")
    console.voice_status("processing")  # служебное состояние не печатается
    console.voice_status("idle")
    assert out.getvalue().splitlines() == [
        "  [голос] Слушаю…",
        "  [голос] Распознано: «Сделай громче.»",
        "  [голос] Жду обращения",
    ]
