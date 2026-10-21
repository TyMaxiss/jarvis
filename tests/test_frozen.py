"""Собранный exe: подсказки с правильной командой и PATH для запускаемых программ."""

from __future__ import annotations

import os

import pytest

from assistant import frozen


def test_run_hint_names_console_exe_in_build(monkeypatch: pytest.MonkeyPatch) -> None:
    assert frozen.run_hint("--download-models") == "uv run python -m assistant --download-models"
    monkeypatch.setattr(frozen, "FROZEN", True)
    assert frozen.run_hint("--download-models") == "JarvisConsole.exe --download-models"


def test_clean_path_drops_bundle_folders(tmp_path: os.PathLike[str]) -> None:
    bundle = os.path.join(str(tmp_path), "_internal")
    path = os.pathsep.join(
        [os.path.join(bundle, "vosk"), "/usr/bin", bundle, "", os.path.join(str(tmp_path), "other")]
    )
    assert frozen.clean_path(path, bundle) == os.pathsep.join(
        ["/usr/bin", os.path.join(str(tmp_path), "other")]
    )


def test_external_launch_does_nothing_outside_build() -> None:
    before = os.environ.get("PATH")
    with frozen.external_launch():
        assert os.environ.get("PATH") == before
    assert os.environ.get("PATH") == before


def test_run_program_is_like_subprocess_run() -> None:
    import subprocess
    import sys

    done = frozen.run_program(
        [sys.executable, "-c", "print('готово')"], capture_output=True, text=True, timeout=30
    )
    assert (done.returncode, done.stdout.strip()) == (0, "готово")
    with pytest.raises(subprocess.CalledProcessError):
        frozen.run_program([sys.executable, "-c", "raise SystemExit(3)"], check=True, timeout=30)
    with pytest.raises(subprocess.TimeoutExpired):
        frozen.run_program([sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.5)
