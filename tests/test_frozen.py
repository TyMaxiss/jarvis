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
