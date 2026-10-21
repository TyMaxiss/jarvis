"""Обёртки WinAPI, которые проверяются и не в Windows."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from assistant.actions import winapi


@dataclass
class FakeProcess:
    pid: int
    process_name: str
    chain: list[FakeProcess]

    def name(self) -> str:
        return self.process_name

    def parents(self) -> list[FakeProcess]:
        return self.chain


def test_own_processes_stop_at_windows_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    # python ← uv ← powershell ← терминал ← Проводник: окна Проводника — не окна ассистента.
    chain = [
        FakeProcess(2, "uv.exe", []),
        FakeProcess(3, "powershell.exe", []),
        FakeProcess(4, "WindowsTerminal.exe", []),
        FakeProcess(5, "explorer.exe", []),
        FakeProcess(6, "parent-of-explorer.exe", []),
    ]
    monkeypatch.setattr(winapi.psutil, "Process", lambda: FakeProcess(1, "python.exe", chain))
    assert winapi.own_process_tree() == {1, 2, 3, 4}
    # Запуск ярлыком: родитель — сразу Проводник.
    monkeypatch.setattr(winapi.psutil, "Process", lambda: FakeProcess(1, "Jarvis.exe", chain[3:]))
    assert winapi.own_process_tree() == {1}
