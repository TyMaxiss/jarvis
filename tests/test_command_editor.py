"""Редактор команд: дерево паков, форма, сохранение в YAML без порчи файла, проверка фразы."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from assistant.actions.registry import ActionRegistry
from assistant.commands.editor import CommandEditor, EditorError, action_catalog, dump_item, update_node
from assistant.commands.loader import CommandStore
from assistant.config import AppConfig
from assistant.nlu.matcher import Matcher
from assistant.nlu.normalize import Normalizer
from tests.conftest import ROOT

MY_YAML = """\
# Мои команды.
# Пример — ниже.

- id: work_mode
  phrases: ["рабочий режим"]  # главная фраза
  reply: "Все системы готовы"
  steps:
    - open_app: {name: "Telegram"}
    # потом редактор
    - open_app: {name: "Visual Studio Code"}

# Музыка
- id: loud
  phrases: ["погромче"]
  steps:
    - volume_change: {delta: 20}
"""


@pytest.fixture
def commands(tmp_path: Path) -> Path:
    directory = tmp_path / "commands"
    shutil.copytree(ROOT / "commands", directory)
    (directory / "my.yaml").write_text(MY_YAML, encoding="utf-8")
    return directory


@pytest.fixture
def editor(commands: Path, registry: ActionRegistry, config: AppConfig) -> CommandEditor:
    store = CommandStore(commands, "my.yaml", registry)
    store.reload()
    matcher = Matcher(Normalizer(config.wake_words, config.matcher.filler_words), config.matcher)
    matcher.set_commands(store.commands)
    return CommandEditor(store, registry, lambda: matcher)


def form(**fields: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "new_cmd",
        "phrases": [],
        "slots": {},
        "when": {"active_window_contains": [], "active_process": []},
        "confirm": False,
        "reply": "",
        "hotkey": "",
        "steps": [],
    }
    return {**base, **fields}


def test_tree_starts_with_own_pack_and_marks_replaced_commands(editor: CommandEditor, commands: Path) -> None:
    with (commands / "my.yaml").open("a", encoding="utf-8") as file:
        file.write('\n- id: volume_up\n  phrases: ["громче"]\n  steps:\n    - volume_change: {delta: 5}\n')
    editor.store.reload()
    tree = editor.tree()
    assert [pack["file"] for pack in tree] == [
        "my.yaml",
        "browser.yaml",
        "media.yaml",
        "screen.yaml",
        "timers.yaml",
        "windows.yaml",
    ]
    mine = {c["id"]: c for c in tree[0]["commands"]}
    assert mine["work_mode"] == {
        **mine["work_mode"],
        "title": "рабочий режим",
        "steps": 2,
        "active": True,
        "line": 4,
    }
    media = {c["id"]: c for c in tree[2]["commands"]}  # media.yaml
    assert media["volume_up"]["replaced_by"] == "my.yaml" and not media["volume_up"]["active"]
    assert all(not c["problem"] for pack in tree for c in pack["commands"])


def test_broken_command_is_listed_with_its_problem(editor: CommandEditor, commands: Path) -> None:
    with (commands / "my.yaml").open("a", encoding="utf-8") as file:
        file.write('\n- id: broken\n  phrases: ["сломано"]\n  steps:\n    - fly_away: {}\n')
    editor.store.reload()
    broken = next(c for c in editor.tree()[0]["commands"] if c["id"] == "broken")
    assert not broken["active"] and "неизвестное действие fly_away" in broken["problem"]
    assert editor.command("my.yaml", "broken")["problems"] == ["Шаг 1: неизвестное действие fly_away"]


def test_command_for_form(editor: CommandEditor) -> None:
    data = editor.command("media.yaml", "youtube_pause")
    assert not data["user"] and data["also_in"] == []
    assert data["command"] == form(
        id="youtube_pause",
        phrases=["пауза", "поставь на паузу", "продолжи", "сними с паузы"],
        when={"active_window_contains": ["YouTube"], "active_process": []},
        steps=[{"action": "hotkey", "args": {"keys": "k"}}],
    )
    with pytest.raises(EditorError):
        editor.command("../config.yaml", "x")  # только паки из папки команд


def test_new_command_is_appended_in_pack_style(editor: CommandEditor, commands: Path) -> None:
    steps = [
        {"action": "volume_set", "args": {"percent": "{percent}"}},
        {"action": "say", "args": {"text": "Готово"}},
    ]
    saved = editor.save("", "", form(id="night", phrases=["ночной режим {percent}"], slots={"percent": "int"},
                                     steps=steps))  # fmt: skip
    assert saved == "night"
    text = (commands / "my.yaml").read_text(encoding="utf-8")
    assert text == MY_YAML + (
        "\n"
        "- id: night\n"
        '  phrases: ["ночной режим {percent}"]\n'
        "  slots: {percent: int}\n"
        "  steps:\n"
        '    - volume_set: {percent: "{percent}"}\n'
        '    - say: {text: "Готово"}\n'
    )
    assert editor.store.reload().issues == []


def test_editing_own_command_changes_only_its_text(editor: CommandEditor, commands: Path) -> None:
    current = editor.command("my.yaml", "work_mode")["command"]
    current["reply"] = "Поехали"
    current["confirm"] = True
    editor.save("my.yaml", "work_mode", current)
    text = (commands / "my.yaml").read_text(encoding="utf-8")
    # Комментарии внутри команды и вокруг неё на месте; изменились две строки.
    assert text == MY_YAML.replace('  reply: "Все системы готовы"\n', '  confirm: true\n  reply: "Поехали"\n')


def test_editing_in_file_with_windows_line_endings(editor: CommandEditor, commands: Path) -> None:
    (commands / "my.yaml").write_bytes(MY_YAML.replace("\n", "\r\n").encode("utf-8"))
    current = editor.command("my.yaml", "work_mode")["command"]
    current["reply"] = "Поехали"
    editor.save("my.yaml", "work_mode", current)
    expected = MY_YAML.replace('  reply: "Все системы готовы"\n', '  reply: "Поехали"\n')
    assert (commands / "my.yaml").read_bytes().decode("utf-8") == expected.replace("\n", "\r\n")


def test_editing_steps_keeps_unchanged_steps(editor: CommandEditor, commands: Path) -> None:
    current = editor.command("my.yaml", "work_mode")["command"]
    current["steps"][1]["args"]["name"] = "Блокнот"
    editor.save("my.yaml", "work_mode", current)
    text = (commands / "my.yaml").read_text(encoding="utf-8")
    assert (
        '    - open_app: {name: "Telegram"}\n    # потом редактор\n    - open_app: {name: "Блокнот"}\n'
        in text
    )
    assert text.endswith(
        '# Музыка\n- id: loud\n  phrases: ["погромче"]\n  steps:\n    - volume_change: {delta: 20}\n'
    )


def test_builtin_command_is_saved_as_copy_in_own_pack(editor: CommandEditor, commands: Path) -> None:
    media = (commands / "media.yaml").read_text(encoding="utf-8")
    current = editor.command("media.yaml", "volume_up")["command"]
    current["steps"] = [{"action": "volume_change", "args": {"delta": "15"}}]  # число из поля ввода
    editor.save("media.yaml", "volume_up", current)
    assert (commands / "media.yaml").read_text(encoding="utf-8") == media
    assert "    - volume_change: {delta: 15}\n" in (commands / "my.yaml").read_text(encoding="utf-8")
    editor.store.reload()
    active = next(c for c in editor.store.commands if c.id == "volume_up")
    assert active.source == "my.yaml" and active.steps[0].args == {"delta": 15}
    assert editor.command("media.yaml", "volume_up")["also_in"] == ["my.yaml"]


@pytest.mark.parametrize(
    ("fields", "error"),
    [
        ({"id": ""}, "Укажите id команды"),
        ({"id": "два слова"}, "id: только буквы, цифры"),
        ({"steps": []}, "Добавьте хотя бы одно действие"),
        ({"steps": [{"action": "fly", "args": {}}]}, "Шаг 1: неизвестное действие fly"),
        ({"steps": [{"action": "volume_set", "args": {"percent": "{level}"}}]}, "Шаг 1: слоты ['level']"),
        ({"phrases": [], "steps": [{"action": "get_time", "args": {}}]}, "Нет ни фраз, ни горячей клавиши"),
        ({"steps": [{"action": "volume_set", "args": {"percent": 300}}]}, "должно быть не больше 100"),
    ],
)
def test_errors_are_explained(editor: CommandEditor, fields: dict[str, Any], error: str) -> None:
    data = form(phrases=["тест"], steps=[{"action": "get_time", "args": {}}])
    with pytest.raises(EditorError) as info:
        editor.save("", "", {**data, **fields})
    assert any(error in message for message in info.value.errors), info.value.errors


def test_id_clash_in_own_pack(editor: CommandEditor) -> None:
    data = form(id="loud", phrases=["тише"], steps=[{"action": "get_time", "args": {}}])
    with pytest.raises(EditorError, match="уже есть команда «loud»"):
        editor.save("", "", data)


def test_unreadable_own_pack_is_not_overwritten(editor: CommandEditor, commands: Path) -> None:
    (commands / "my.yaml").write_text("- id: x\n  phrases: [незакрыто\n", encoding="utf-8")
    with pytest.raises(EditorError, match="не читается"):
        editor.save("", "", form(phrases=["тест"], steps=[{"action": "get_time", "args": {}}]))
    assert (commands / "my.yaml").read_text(encoding="utf-8") == "- id: x\n  phrases: [незакрыто\n"


def test_comments_only_pack_and_windows_line_endings(editor: CommandEditor, commands: Path) -> None:
    header = "# Ваши команды.\r\n# - id: example\r\n"
    (commands / "my.yaml").write_bytes(header.encode("utf-8"))
    editor.save("", "", form(id="t", phrases=["тест"], steps=[{"action": "get_time", "args": {}}]))
    data = (commands / "my.yaml").read_bytes().decode("utf-8")
    assert data == header + '\r\n- id: t\r\n  phrases: ["тест"]\r\n  steps:\r\n    - get_time: {}\r\n'


def test_delete_keeps_following_comment(editor: CommandEditor, commands: Path) -> None:
    editor.delete("my.yaml", "work_mode")
    assert (commands / "my.yaml").read_text(encoding="utf-8") == (
        '# Мои команды.\n# Пример — ниже.\n\n# Музыка\n- id: loud\n  phrases: ["погромче"]\n  steps:\n'
        "    - volume_change: {delta: 20}\n"
    )
    with pytest.raises(EditorError, match="только свои команды"):
        editor.delete("media.yaml", "volume_up")


def test_phrase_check_uses_unsaved_draft(editor: CommandEditor) -> None:
    current = editor.command("my.yaml", "loud")["command"]
    assert editor.test_phrase("очень громко", None)["status"] != "matched"
    current["phrases"] = ["очень громко"]
    result = editor.test_phrase("очень громко", current, "loud")
    assert result["status"] == "matched"
    assert [(m["id"], m["draft"]) for m in result["matches"]] == [("loud", True)]
    # Та же фраза у другой команды — неоднозначно, и видно у какой.
    current["phrases"] = ["громкость на максимум"]
    clash = editor.test_phrase("громкость на максимум", current, "loud")
    assert clash["status"] == "ambiguous" and {c["id"] for c in clash["candidates"][:2]} == {
        "loud",
        "volume_max",
    }
    # Черновик с условием when проверяется в окне, где условие выполнено.
    youtube = editor.command("media.yaml", "youtube_pause")["command"]
    assert editor.test_phrase("пауза", youtube, "youtube_pause")["matches"][0]["id"] == "youtube_pause"
    broken = editor.test_phrase("громче", {**current, "steps": []}, "loud")
    assert (
        broken["errors"] == ["Добавьте хотя бы одно действие"] and broken["matches"][0]["id"] == "volume_up"
    )


def test_run_takes_slot_values_from_phrase(editor: CommandEditor) -> None:
    draft = form(id="set", phrases=["громкость {percent}"], slots={"percent": "int"},
                 steps=[{"action": "volume_set", "args": {"percent": "{percent}"}}])  # fmt: skip
    command, slots, phrase = editor.prepare_run(draft, "громкость семьдесят")
    assert command.id == "set" and slots == {"percent": 70} and phrase == "громкость семьдесят"
    with pytest.raises(EditorError, match="впишите фразу"):
        editor.prepare_run(draft)
    with pytest.raises(EditorError, match="не подходит"):
        editor.prepare_run(draft, "открой ютуб")


def test_catalog_groups_actions_with_schemas(registry: ActionRegistry) -> None:
    catalog = action_catalog(registry)
    groups = list(dict.fromkeys(item["group"] for item in catalog))
    assert groups[:3] == ["Программы", "Окна", "Клавиатура и мышь"]
    volume = next(item for item in catalog if item["name"] == "volume_set")
    assert volume["group"] == "Звук" and volume["schema"]["properties"]["percent"]["maximum"] == 100
    assert next(item for item in catalog if item["name"] == "shutdown")["risk"] == "confirm"


def test_new_item_text(registry: ActionRegistry) -> None:
    from ruamel.yaml.comments import CommentedMap

    node = CommentedMap()
    when = {"active_process": ["chrome.exe", "msedge.exe"]}
    steps = [{"action": "get_time", "args": {}}]
    update_node(
        node, {"id": "a", "phrases": ["x", "y"], "when": when, "hotkey": "ctrl+alt+1", "steps": steps}
    )
    assert dump_item(node) == (
        "- id: a\n"
        '  phrases: ["x", "y"]\n'
        '  when: {active_process: ["chrome.exe", "msedge.exe"]}\n'
        '  hotkey: "ctrl+alt+1"\n'
        "  steps:\n"
        "    - get_time: {}\n"
    )
