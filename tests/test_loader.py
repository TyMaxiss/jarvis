"""Тесты загрузки YAML-команд."""

from __future__ import annotations

from pathlib import Path

import pytest

from assistant.actions.registry import ActionRegistry
from assistant.commands.loader import (
    CommandStore,
    load_commands,
    load_file,
    pack_files,
    render_reply,
    substitute,
)
from tests.conftest import ROOT


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_builtin_packs_load_without_issues(registry: ActionRegistry) -> None:
    result = load_commands(ROOT / "commands", "my.yaml", registry)
    assert result.issues == []
    ids = {c.id for c in result.commands}
    assert {"minimize_all", "volume_set", "open_youtube", "youtube_pause"} <= ids


def test_user_pack_example_turns_on_by_removing_hashes(tmp_path: Path, registry: ActionRegistry) -> None:
    # В my.yaml сказано «уберите # чтобы включить» — после этого файл должен читаться.
    text = (ROOT / "commands" / "my.yaml").read_text(encoding="utf-8")
    uncommented = "\n".join(
        line[2:] if line.startswith(("# -", "#  ")) else line for line in text.splitlines()
    )
    result = load_file(write(tmp_path / "my.yaml", uncommented), registry)
    assert result.issues == []
    assert [c.id for c in result.commands] == ["work_mode"]


def test_spec_example_commands(tmp_path: Path, registry: ActionRegistry) -> None:
    pack = write(
        tmp_path / "spec.yaml",
        """
- id: minimize_all
  phrases: ["сверни все окна", "покажи рабочий стол"]
  steps:
    - hotkey: {keys: "win+d"}

- id: volume_set
  phrases: ["громкость на {percent}", "поставь громкость {percent}"]
  slots: {percent: int}
  steps:
    - volume_set: {percent: "{percent}"}

- id: youtube_pause
  phrases: ["пауза", "стоп"]
  when: {active_window_contains: "YouTube"}
  steps:
    - hotkey: {keys: "k"}

- id: work_mode
  phrases: ["рабочий режим"]
  steps:
    - open_app: {name: "Telegram"}
    - open_app: {name: "Visual Studio Code"}
    - volume_set: {percent: 30}
    - say: {text: "Все системы готовы"}
""",
    )
    result = load_file(pack, registry)
    assert result.issues == []
    by_id = {c.id: c for c in result.commands}
    assert by_id["volume_set"].slots == {"percent": "int"}
    assert by_id["youtube_pause"].when.active_window_contains == ["YouTube"]
    assert [s.action for s in by_id["work_mode"].steps] == ["open_app", "open_app", "volume_set", "say"]
    assert by_id["minimize_all"].location == "spec.yaml:2"


@pytest.mark.parametrize(
    ("yaml_text", "message"),
    [
        ("- id: x\n  phrases: [тест]\n  steps:\n    - fly_to_moon: {}\n", "неизвестное действие fly_to_moon"),
        ("- id: x\n  phrases: [тест]\n  steps:\n    - volume_set: {percent: 150}\n", "не больше 100"),
        ("- id: x\n  phrases: [тест]\n  steps:\n    - volume_set: {}\n", "percent — не указан"),
        (
            "- id: x\n  phrases: [тест]\n  steps:\n    - volume_set: {percent: 5, extra: 1}\n",
            "лишний параметр",
        ),
        (
            "- id: x\n  phrases: [тест]\n  steps:\n    - volume_set: {percent: '{p}'}\n",
            "не встречаются во фразах",
        ),
        (
            "- id: x\n  phrases: ['громкость {p}']\n  steps:\n    - volume_set: {percent: '{p}'}\n",
            "нужно целое число",
        ),  # слот без типа — строка, а громкости нужно число
        (
            "- id: x\n  phrases: [тест]\n  slots: {p: int}\n  steps:\n    - say: {text: a}\n",
            "не встречаются во фразах",
        ),
        ("- id: x\n  steps:\n    - say: {text: a}\n", "нечем запустить"),
        ("- id: x\n  phrases: [тест]\n  steps: []\n", "steps"),
        ("- id: x\n  phrases: [тест]\n  colour: red\n  steps:\n    - say: {text: a}\n", "colour"),
    ],
)
def test_invalid_commands_are_reported(
    tmp_path: Path, registry: ActionRegistry, yaml_text: str, message: str
) -> None:
    result = load_file(write(tmp_path / "bad.yaml", yaml_text), registry)
    assert result.commands == []
    assert len(result.issues) == 1
    assert message in str(result.issues[0])
    assert str(result.issues[0]).startswith("bad.yaml:1")


def test_bad_command_does_not_break_others(tmp_path: Path, registry: ActionRegistry) -> None:
    pack = write(
        tmp_path / "mixed.yaml",
        "- id: good\n  phrases: [хорошо]\n  steps:\n    - say: {text: ok}\n"
        "- id: bad\n  phrases: [плохо]\n  steps:\n    - nope: {}\n",
    )
    result = load_file(pack, registry)
    assert [c.id for c in result.commands] == ["good"]
    assert "mixed.yaml:5" in str(result.issues[0])


def test_broken_yaml_and_wrong_top_level(tmp_path: Path, registry: ActionRegistry) -> None:
    broken = load_file(write(tmp_path / "broken.yaml", "- id: x\n  phrases: [unclosed\n"), registry)
    assert "не удалось прочитать" in str(broken.issues[0])
    mapping = load_file(write(tmp_path / "map.yaml", "id: x\n"), registry)
    assert "список команд" in str(mapping.issues[0])
    empty = load_file(write(tmp_path / "empty.yaml", "# только комментарий\n"), registry)
    assert empty.commands == [] and empty.issues == []


def test_step_without_args(tmp_path: Path, registry: ActionRegistry) -> None:
    pack = write(tmp_path / "p.yaml", "- id: t\n  phrases: [время]\n  steps:\n    - get_time:\n")
    result = load_file(pack, registry)
    assert result.issues == []
    assert result.commands[0].steps[0].args == {}


def test_user_pack_loads_last_and_overrides(tmp_path: Path, registry: ActionRegistry) -> None:
    write(tmp_path / "b.yaml", "- id: hello\n  phrases: [привет]\n  steps:\n    - say: {text: встроенная}\n")
    write(tmp_path / "my.yaml", "- id: hello\n  phrases: [привет]\n  steps:\n    - say: {text: моя}\n")
    write(tmp_path / "z.yaml", "- id: other\n  phrases: [другое]\n  steps:\n    - say: {text: z}\n")
    assert [p.name for p in pack_files(tmp_path, "my.yaml")] == ["b.yaml", "z.yaml", "my.yaml"]
    result = load_commands(tmp_path, "my.yaml", registry)
    hello = next(c for c in result.commands if c.id == "hello")
    assert hello.steps[0].args == {"text": "моя"}
    assert hello.source == "my.yaml"


def test_store_reload_picks_up_changes(tmp_path: Path, registry: ActionRegistry) -> None:
    pack = write(tmp_path / "a.yaml", "- id: one\n  phrases: [раз]\n  steps:\n    - say: {text: 1}\n")
    store = CommandStore(tmp_path, "my.yaml", registry)
    store.reload()
    assert [c.id for c in store.commands] == ["one"]
    write(pack, "- id: two\n  phrases: [два]\n  steps:\n    - say: {text: 2}\n")
    store.reload()
    assert [c.id for c in store.commands] == ["two"]


def test_missing_directory(tmp_path: Path, registry: ActionRegistry) -> None:
    result = load_commands(tmp_path / "nope", "my.yaml", registry)
    assert result.commands == [] and "не найдена" in str(result.issues[0])


def test_substitute_keeps_types_for_whole_placeholders() -> None:
    slots = {"percent": 50, "name": "Telegram"}
    assert substitute("{percent}", slots) == 50
    assert substitute("-{percent}", slots) == "-50"
    nested = {"a": ["{name}", "x {name}"], "b": 3}
    assert substitute(nested, slots) == {"a": ["Telegram", "x Telegram"], "b": 3}
    assert substitute("{unknown}", slots) == "{unknown}"


def test_render_reply() -> None:
    assert render_reply("Громкость {percent}%", {"percent": 30}, "") == "Громкость 30%"
    assert render_reply("{result}", {}, "Сейчас 12:00") == "Сейчас 12:00"
