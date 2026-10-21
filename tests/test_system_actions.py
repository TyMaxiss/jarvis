"""Тесты системных действий и правил безопасности для них (без настоящей Windows)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from assistant.actions import files, power, shell
from assistant.actions.registry import ActionError, ActionRegistry, Risk
from assistant.safety.policy import SafetyPolicy, Verdict


class FakePower(power.PowerControl):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def lock(self) -> None:
        self.calls.append("lock")

    def sleep(self) -> None:
        self.calls.append("sleep")

    def shutdown(self, restart: bool) -> None:
        self.calls.append("restart" if restart else "shutdown")


def verdict(registry: ActionRegistry, name: str, **args: Any) -> tuple[Verdict, str]:
    action = registry.get(name)
    decision = SafetyPolicy().evaluate(action, action.parse_args(args))
    return decision.verdict, decision.reason


@pytest.fixture
def registry(tmp_path: Path) -> ActionRegistry:
    reg = ActionRegistry()
    power.register(reg, FakePower())
    shell.register(reg, runner=lambda *a, **k: subprocess.CompletedProcess(a, 0, "ok", ""))
    folders = {
        "home": tmp_path,
        "desktop": tmp_path / "Desktop",
        "downloads": tmp_path / "Downloads",
    }
    files.register(reg, opener=lambda path: None, folders=folders.__getitem__)
    return reg


class TestPower:
    def test_risk_levels(self, registry: ActionRegistry) -> None:
        risks = {name: registry.get(name).risk for name in ("lock", "sleep", "shutdown", "restart")}
        assert risks["lock"] is Risk.SAFE
        assert {risks["sleep"], risks["shutdown"], risks["restart"]} == {Risk.CONFIRM}

    def test_actions_call_windows(self) -> None:
        fake = FakePower()
        reg = ActionRegistry()
        power.register(reg, fake)
        replies = [
            reg.get(name).run(reg.get(name).parse_args({})) for name in ("lock", "shutdown", "restart")
        ]
        assert fake.calls == ["lock", "shutdown", "restart"]
        assert replies[1] == "Выключаю компьютер через 5 секунд"


class TestShellPolicy:
    @pytest.mark.parametrize(
        "command",
        [
            "Get-ChildItem C:\\Users",
            "Get-Process | Format-Table Name, Id",  # Format-Table — не форматирование диска
            "Remove-Item C:\\temp\\a.txt",  # удаление одного файла — с подтверждением
            "del report.txt",
            "ipconfig /all",
        ],
    )
    def test_ordinary_commands_need_confirmation(self, registry: ActionRegistry, command: str) -> None:
        assert verdict(registry, "run_shell", command=command)[0] is Verdict.CONFIRM

    @pytest.mark.parametrize(
        "command",
        [
            "format C: /q",
            "Format-Volume -DriveLetter D",
            "diskpart /s script.txt",
            "del /s /q C:\\Users\\me\\*.*",
            "rd /s /q C:\\Projects",
            "cmd /c rmdir /S C:\\data",
            "Remove-Item -Path C:\\Projects -Recurse -Force",
            "Get-ChildItem C:\\tmp | Remove-Item -Recurse",
            "rm -r C:\\data",
            "[System.IO.Directory]::Delete('C:\\data', $true)",
            "reg delete HKCU\\Software\\App /f",
            "Remove-Item HKLM:\\Software\\App",
            "bcdedit /set {current} safeboot minimal",
            "vssadmin delete shadows /all",
            "wbadmin delete catalog",
            "cipher /w:C:",
            "powershell -enc ZQBjAGgAbwA=",
        ],
    )
    def test_dangerous_commands_are_forbidden(self, registry: ActionRegistry, command: str) -> None:
        result, reason = verdict(registry, "run_shell", command=command)
        assert result is Verdict.FORBID and reason.startswith("опасная команда")

    def test_denylist_checks_each_pipeline_part(self, registry: ActionRegistry) -> None:
        # «-Recurse» относится к Get-ChildItem, а удаляется один файл.
        command = "Remove-Item a.txt; Get-ChildItem -Recurse"
        assert verdict(registry, "run_shell", command=command)[0] is Verdict.CONFIRM


class TestShellAction:
    def run(self, result: Any) -> str:
        def runner(*args: Any, **kwargs: Any) -> Any:
            if isinstance(result, Exception):
                raise result
            return result

        reg = ActionRegistry()
        shell.register(reg, runner=runner)
        action = reg.get("run_shell")
        return action.run(action.parse_args({"command": "Get-Date"}))

    def test_output_is_returned_and_shortened(self) -> None:
        assert self.run(subprocess.CompletedProcess([], 0, "  привет\n", "")) == "привет"
        long = self.run(subprocess.CompletedProcess([], 0, "x" * 5000, ""))
        assert len(long) < 1600 and long.endswith("(ещё 3500 символов)")

    def test_error_and_timeout(self) -> None:
        with pytest.raises(ActionError, match="код 1.*не найден"):
            self.run(subprocess.CompletedProcess([], 1, "", "Путь не найден"))
        with pytest.raises(ActionError, match="не завершилась за 30"):
            self.run(subprocess.TimeoutExpired("powershell", 30))

    def test_output_is_utf8(self) -> None:
        args = shell.powershell_args("Get-Date")
        assert args[:4] == ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command"]
        assert args[4].startswith("[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; ")


def test_mouse_events() -> None:
    from assistant.actions import winapi

    double = winapi.mouse_button_events("left", double=True)
    assert [e.mi.dwFlags for e in double] == [0x0002, 0x0004, 0x0002, 0x0004]
    assert [e.mi.dwFlags for e in winapi.mouse_button_events("right")] == [0x0008, 0x0010]
    assert winapi.wheel_event(3).mi.mouseData == 360
    assert winapi.wheel_event(-1).mi.mouseData == 0xFFFFFF88  # −120 дополнением до 32 бит


class TestFiles:
    def test_paths(self, tmp_path: Path) -> None:
        folders = {"home": tmp_path, "desktop": tmp_path / "OneDrive" / "Рабочий стол"}.__getitem__
        desktop = tmp_path / "OneDrive" / "Рабочий стол"
        assert files.resolve_user_path("Рабочий стол/Проекты", folders) == desktop / "Проекты"
        assert files.resolve_user_path("рабочий  стол", folders) == desktop
        assert files.resolve_user_path("заметки\\план.txt", folders) == tmp_path / "заметки/план.txt"
        assert files.resolve_user_path(str(tmp_path / "abs"), folders) == tmp_path / "abs"
        with pytest.raises(ActionError):
            files.resolve_user_path("  ", folders)

    def test_create_and_write(self, registry: ActionRegistry, tmp_path: Path) -> None:
        create = registry.get("create_folder")
        assert create.run(create.parse_args({"path": "Рабочий стол/Проекты"})).startswith("Создал папку")
        assert (tmp_path / "Desktop" / "Проекты").is_dir()
        write = registry.get("write_file")
        write.run(write.parse_args({"path": "Рабочий стол/Проекты/план.txt", "text": "Пункт 1"}))
        assert (tmp_path / "Desktop/Проекты/план.txt").read_text(encoding="utf-8") == "Пункт 1"

    def test_overwrite_needs_confirmation(self, registry: ActionRegistry, tmp_path: Path) -> None:
        target = tmp_path / "plan.txt"
        assert verdict(registry, "write_file", path=str(target), text="новый")[0] is Verdict.ALLOW
        target.write_text("старый", encoding="utf-8")
        result, reason = verdict(registry, "write_file", path=str(target), text="новый")
        assert result is Verdict.CONFIRM and "перезаписан" in reason

    def test_open_missing_folder(self, registry: ActionRegistry) -> None:
        action = registry.get("open_folder")
        with pytest.raises(ActionError, match="нет"):
            action.run(action.parse_args({"path": "Нет такой папки"}))
