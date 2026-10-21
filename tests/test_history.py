"""История: журнал → фразы с действиями; фильтры, поиск, страницы; журнал старого формата."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field

from assistant.actions.executor import ActionExecutor
from assistant.actions.registry import ActionError, ActionRegistry
from assistant.commands.loader import CommandStore
from assistant.config import MatcherConfig
from assistant.history import History, build_items, read_entries
from assistant.journal import ActionJournal
from assistant.logging_setup import SecretMasker
from assistant.nlu.matcher import Matcher
from assistant.nlu.normalize import Normalizer
from assistant.pipeline import Pipeline
from assistant.safety.confirm import StaticConfirmer
from assistant.safety.policy import SafetyPolicy
from tests.conftest import NO_WINDOW, RecordingSpeaker

PACK = """
- id: volume_set
  phrases: ["громкость на {percent}"]
  slots: {percent: int}
  reply: "Громкость {percent}"
  steps:
    - set_volume: {percent: "{percent}"}

- id: open_app
  phrases: ["открой {name}"]
  steps:
    - open_app: {name: "{name}"}
"""


def make_pipeline(tmp_path: Path) -> tuple[Pipeline, Path]:
    registry = ActionRegistry()

    @registry.action("set_volume", "Громкость")
    def set_volume(percent: Annotated[int, Field(ge=0, le=100)]) -> str:
        return f"{percent}%"

    @registry.action("open_app", "Открыть")
    def open_app(name: str) -> str:
        raise ActionError(f"Не нашёл приложение «{name}»")

    (tmp_path / "pack.yaml").write_text(PACK, encoding="utf-8")
    store = CommandStore(tmp_path, "my.yaml", registry)
    store.reload()
    matcher = Matcher(Normalizer(["джарвис"]), MatcherConfig())
    matcher.set_commands(store.commands)
    journal_path = tmp_path / "logs" / "actions.jsonl"
    journal = ActionJournal(journal_path, SecretMasker())
    executor = ActionExecutor(registry, SafetyPolicy(), StaticConfirmer(True), journal)
    return Pipeline(matcher, executor, RecordingSpeaker(), lambda: NO_WINDOW), journal_path


def test_phrases_from_pipeline_come_with_their_actions(tmp_path: Path) -> None:
    pipeline, journal = make_pipeline(tmp_path)
    pipeline.handle_text("громкость на пятьдесят", "voice")
    pipeline.handle_text("открой телеграм", "text")
    pipeline.handle_text("расскажи сказку", "text")  # нейросеть выключена
    pipeline.handle_text("", "voice")  # пустая фраза в историю не попадает
    items = History(journal).query()["items"]
    assert [item["phrase"] for item in items] == [
        "расскажи сказку",
        "открой телеграм",
        "громкость на пятьдесят",
    ]
    unknown, failed, volume = items
    assert volume == {
        **volume,
        "source": "voice",
        "route": "local",
        "commands": ["volume_set"],
        "reply": "Громкость 50",
        "ok": True,
        "error": "",
    }
    assert [(a["action"], a["args"], a["status"]) for a in volume["actions"]] == [
        ("set_volume", {"percent": 50}, "done")
    ]
    assert not failed["ok"] and failed["error"] == "Не нашёл приложение «телеграм»"
    assert failed["actions"][0]["status"] == "failed"
    assert unknown["route"] == "llm" and not unknown["ok"] and unknown["error"] == "нейросеть выключена"
    assert len({item["id"] for item in items}) == 3


def test_agent_actions_join_their_phrase_by_request(tmp_path: Path) -> None:
    journal_path = tmp_path / "actions.jsonl"
    journal = ActionJournal(journal_path, SecretMasker())
    step = {"source": "llm", "status": "done", "duration_ms": 5.0, "phrase": "создай папку и открой её"}
    journal.write(action="create_folder", args={"name": "Тест"}, result="Создал", request="r1", **step)
    journal.write(action="open_folder", args={"path": "Тест"}, result="Открыл", request="r1", **step)
    journal.write_phrase(request="r1", phrase="создай папку и открой её", source="voice", route="llm",
                         commands=[], reply="Готово", ok=True, duration_ms=2100)  # fmt: skip
    (item,) = History(journal_path).query()["items"]
    assert item["id"] == "r1" and item["route"] == "llm" and item["source"] == "voice"
    assert [a["action"] for a in item["actions"]] == ["create_folder", "open_folder"]


def test_old_journal_is_grouped_by_phrase_and_time(tmp_path: Path) -> None:
    def old(
        time: str, phrase: str, action: str, status: str = "done", command: str | None = "work_mode"
    ) -> dict[str, Any]:
        source = "voice" if command else "llm"
        result = f"{action}: {status}"
        return {"time": f"2026-10-20T{time}.000+03:00", "source": source, "command": command,
                "phrase": phrase, "action": action, "args": {}, "status": status, "result": result,
                "duration_ms": 10}  # fmt: skip

    entries = [
        old("10:00:00", "рабочий режим", "open_app"),
        old("10:00:01", "рабочий режим", "open_app"),
        old("10:00:02", "рабочий режим", "volume_set", status="failed"),
        old("10:30:00", "рабочий режим", "open_app"),  # тот же текст, но позже — другая фраза
        old("11:00:00", "найди погоду", "open_url", command=None),
    ]
    path = tmp_path / "actions.jsonl"
    path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n", encoding="utf-8")
    items = build_items(read_entries(path))
    assert [(item["phrase"], len(item["actions"])) for item in items] == [
        ("рабочий режим", 3),
        ("рабочий режим", 1),
        ("найди погоду", 1),
    ]
    assert not items[0]["ok"] and items[0]["error"] == "volume_set: failed"
    assert items[0]["duration_ms"] == 30 and items[0]["commands"] == ["work_mode"]
    assert items[2]["route"] == "llm" and items[2]["source"] == "" and items[2]["reply"] == ""
    assert all("_last" not in item for item in items)


def test_unfinished_phrase_is_shown_with_its_actions(tmp_path: Path) -> None:
    journal_path = tmp_path / "actions.jsonl"
    journal = ActionJournal(journal_path, SecretMasker())
    journal.write(source="llm", action="create_folder", args={}, status="done", result="Создал",
                  duration_ms=3, phrase="наведи порядок", request="r9")  # fmt: skip
    (item,) = History(journal_path).query()["items"]
    assert item["id"] == "r9" and item["phrase"] == "наведи порядок" and item["route"] == "llm" and item["ok"]


def test_filters_search_and_pages(tmp_path: Path) -> None:
    journal_path = tmp_path / "actions.jsonl"
    journal = ActionJournal(journal_path, SecretMasker())

    def phrase(request: str, text: str, route: str = "local", ok: bool = True, **extra: Any) -> None:
        fields = {"source": "text", "commands": [], "reply": "", "duration_ms": 1, **extra}
        journal.write_phrase(request=request, phrase=text, route=route, ok=ok, **fields)

    for number in range(5):  # в одну миллисекунду: страницы не должны терять фразы с одинаковым временем
        phrase(f"l{number}", f"громкость на {number}", commands=["volume_set"])
    phrase("e1", "открой телеграм", ok=False, commands=["open_app"], error="Не нашёл")
    phrase("n1", "расскажи сказку", route="llm", reply="Жили-были…")
    history = History(journal_path)
    assert [i["id"] for i in history.query(kind="errors")["items"]] == ["e1"]
    assert [i["id"] for i in history.query(kind="llm")["items"]] == ["n1"]
    assert [i["id"] for i in history.query(search="ЖИЛИ")["items"]] == ["n1"]
    assert [i["id"] for i in history.query(search="open_app")["items"]] == ["e1"]
    first = history.query(limit=3)
    assert [i["id"] for i in first["items"]] == ["n1", "e1", "l4"] and first["more"]
    last = first["items"][-1]
    rest = history.query(limit=3, before=last["id"], before_time=last["time"])
    assert [i["id"] for i in rest["items"]] == ["l3", "l2", "l1"] and rest["more"]
    # Фразы с курсором уже нет (старый журнал обрезан) — страница по времени.
    late = history.query(before="нет-такой", before_time="2100-01-01T00:00:00+00:00")
    assert len(late["items"]) == 7


def test_history_rereads_only_changed_journal(tmp_path: Path, monkeypatch: Any) -> None:
    journal_path = tmp_path / "actions.jsonl"
    journal = ActionJournal(journal_path, SecretMasker())
    history = History(journal_path)
    assert history.query()["items"] == []  # журнала ещё нет
    journal.write_phrase(request="a", phrase="раз", source="text", route="local", commands=[], reply="",
                         ok=True, duration_ms=1)  # fmt: skip
    reads: list[Path] = []
    monkeypatch.setattr(
        "assistant.history.read_entries", lambda path: reads.append(path) or read_entries(path)
    )
    assert len(history.query()["items"]) == 1
    assert len(history.query()["items"]) == 1
    assert reads == [journal_path]  # второй раз — из памяти
    journal.write_phrase(request="b", phrase="два", source="text", route="local", commands=[], reply="",
                         ok=True, duration_ms=1)  # fmt: skip
    assert [i["phrase"] for i in history.query()["items"]] == ["два", "раз"]


def test_tail_skips_cut_line_and_garbage(tmp_path: Path) -> None:
    path = tmp_path / "actions.jsonl"
    good = json.dumps({"time": "2026-10-21T10:00:00+03:00", "kind": "phrase", "request": "x", "phrase": "ок"})
    path.write_text('{"time": "обрезано\n' + "не json\n" + good + "\n[1, 2]\n", encoding="utf-8")
    assert [e["phrase"] for e in read_entries(path)] == ["ок"]
    assert [e["phrase"] for e in read_entries(path, tail_bytes=len(good.encode()) + 10)] == ["ок"]
