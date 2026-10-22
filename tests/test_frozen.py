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


def test_clean_path_drops_cuda_folder_but_not_lookalikes(tmp_path: os.PathLike[str]) -> None:
    bundle, cuda = os.path.join(str(tmp_path), "_internal"), os.path.join(str(tmp_path), "cuda")
    keep = [os.path.join(str(tmp_path), "_internal2"), os.path.join(str(tmp_path), "cuda.download")]
    path = os.pathsep.join([cuda, *keep, os.path.join(bundle, "vosk")])
    assert frozen.clean_path(path, bundle, cuda) == os.pathsep.join(keep)


def test_cuda_folder_is_next_to_program() -> None:
    assert frozen.cuda_dir() == frozen.app_dir() / "cuda"


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


def test_update_mark_is_taken_once(tmp_path: os.PathLike[str], monkeypatch: pytest.MonkeyPatch) -> None:
    # Отметку оставляет установщик, запуская программу после обновления (packaging/jarvis.iss).
    from pathlib import Path

    monkeypatch.setattr(frozen, "app_dir", lambda: Path(tmp_path))
    assert not frozen.take_update_mark()
    (Path(tmp_path) / frozen.UPDATED_MARK).write_text("2026-10-22", encoding="utf-8")
    assert frozen.take_update_mark()
    assert not (Path(tmp_path) / frozen.UPDATED_MARK).exists()
    assert not frozen.take_update_mark()  # следующий запуск — обычный


def test_installer_leaves_the_mark_it_is_named_after() -> None:
    # Имя отметки в установщике и в программе одно и то же.
    from tests.conftest import ROOT

    script = (ROOT / "packaging" / "jarvis.iss").read_text(encoding="utf-8-sig")
    assert "{app}\\" + frozen.UPDATED_MARK in script
