"""Автозапуск и единственный экземпляр: что и как запускать, переключатель в настройках.

Настоящие ярлык и мьютекс Windows проверяются в tests/test_windows.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from assistant.autostart import AUTOSTART_ARG, Autostart, launch_command, project_dir
from assistant.instance import SingleInstance
from assistant.main import parse_args
from tests.conftest import ROOT


def test_source_launch_uses_pythonw_from_project_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    python = tmp_path / "python.exe"
    (tmp_path / "pythonw.exe").touch()
    monkeypatch.setattr(sys, "executable", str(python))
    program, arguments, workdir = launch_command(tmp_path / "config.yaml")
    assert program == tmp_path / "pythonw.exe"  # без окна консоли
    assert arguments == f'-m assistant --config "{(tmp_path / "config.yaml").resolve()}" --autostart'
    assert workdir == ROOT == project_dir()  # отсюда python находит пакет assistant


def test_frozen_launch_uses_exe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exe = tmp_path / "Jarvis" / "Jarvis.exe"
    monkeypatch.setattr(sys, "executable", str(exe))
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    program, arguments, workdir = launch_command(tmp_path / "Jarvis" / "config.yaml")
    assert (program, workdir) == (exe.resolve(), exe.resolve().parent)
    assert arguments.endswith(AUTOSTART_ARG) and "-m assistant" not in arguments


def test_autostart_flag_is_parsed() -> None:
    assert parse_args([AUTOSTART_ARG]).autostart and not parse_args([]).autostart


def test_disable_removes_shortcut(tmp_path: Path) -> None:
    autostart = Autostart(tmp_path / "config.yaml", folder=tmp_path / "Startup")
    autostart.set(False)  # ярлыка нет — не ошибка
    autostart.path.parent.mkdir()
    autostart.path.write_bytes(b"lnk")
    assert autostart.enabled() and autostart.path.name == "Джарвис.lnk"
    autostart.set(False)
    assert not autostart.enabled()


@pytest.mark.skipif(sys.platform == "win32", reason="вне Windows")
def test_enable_outside_windows_is_explained(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="только в Windows"):
        Autostart(tmp_path / "config.yaml", folder=tmp_path).set(True)
    instance = SingleInstance("JarvisTest")
    assert instance.acquire() and not instance.notify()  # вне Windows проверки нет
    instance.release()
