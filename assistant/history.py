"""История для окна: журнал (JSONL) → что сказано, куда ушла фраза, какие действия и чем кончилось.

В журнале этапа 5 действия фразы и запись о самой фразе (kind: phrase) связаны полем
request; запись о фразе идёт последней. В журнале до этапа 5 есть только действия —
они собираются во фразы по тексту фразы и времени.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

PAGE = 50
TAIL_BYTES = 8 * 1024 * 1024  # сколько последних байт журнала читать
FILTERS = ("all", "errors", "llm", "local")
_LEGACY_GAP = timedelta(minutes=2)  # действия старого журнала дальше друг от друга — разные фразы
_OK = frozenset({"done", "dry_run"})
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def read_entries(path: Path, tail_bytes: int = TAIL_BYTES) -> list[dict[str, Any]]:
    """Записи из последних tail_bytes байт журнала; битые строки пропускаются."""
    try:
        with path.open("rb") as file:
            size = file.seek(0, 2)
            start = max(0, size - tail_bytes)
            file.seek(start)
            data = file.read()
    except FileNotFoundError:
        return []
    except OSError:
        log.exception("Не удалось прочитать журнал %s", path)
        return []
    lines = data.split(b"\n")
    if start > 0:
        lines = lines[1:]  # первая строка начинается с середины
    entries = []
    for line in lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


def when(item: dict[str, Any]) -> datetime:
    """Время записи (с часовым поясом)."""
    try:
        moment = datetime.fromisoformat(str(item.get("time", "")))
    except ValueError:
        return _EPOCH
    return moment if moment.tzinfo else moment.astimezone()


def build_items(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Фразы из записей журнала, старые первыми."""
    items: list[dict[str, Any]] = []
    pending: dict[str, list[dict[str, Any]]] = {}  # действия, чья запись о фразе ещё впереди
    legacy: dict[str, Any] | None = None
    for entry in entries:
        request = str(entry.get("request") or "")
        if entry.get("kind") == "phrase":
            items.append(_phrase_item(entry, [_action(e) for e in pending.pop(request, [])]))
        elif request:
            pending.setdefault(request, []).append(entry)
        elif legacy is not None and _same_legacy_phrase(legacy, entry):
            _add_legacy_action(legacy, entry)
        else:
            legacy = _legacy_item(entry, f"old-{len(items)}")
            items.append(legacy)
    # Действия без записи о фразе: фраза ещё выполняется или программа закрылась посреди неё.
    items += [_unfinished_item(request, actions) for request, actions in pending.items()]
    for item in items:
        item.pop("_last", None)
    items.sort(key=when)
    return items


def matches(item: dict[str, Any], kind: str = "all", search: str = "") -> bool:
    """Подходит ли фраза под фильтр (all, errors, llm, local) и строку поиска."""
    if kind == "errors" and item["ok"]:
        return False
    if kind in ("llm", "local") and item["route"] != kind:
        return False
    needle = search.strip().lower()
    if not needle:
        return True
    texts = [item["phrase"], item["reply"], item["error"], *item["commands"]]
    texts += [part for action in item["actions"] for part in (action["action"], action["result"])]
    return any(needle in str(text).lower() for text in texts)


class History:
    """Чтение журнала для страницы «История»; прочитанное запоминается, пока файл не изменится.

    Args:
        path: файл журнала.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._stamp: tuple[int, int] | None = None
        self._items: list[dict[str, Any]] = []

    def items(self) -> list[dict[str, Any]]:
        """Все фразы из журнала, новые первыми."""
        try:
            stat = self.path.stat()
            stamp: tuple[int, int] | None = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            stamp = None
        with self._lock:
            if stamp is None:
                self._stamp, self._items = None, []
            elif stamp != self._stamp:
                self._items = build_items(read_entries(self.path))[::-1]
                self._stamp = stamp
            return self._items

    def query(
        self,
        *,
        limit: int = PAGE,
        before: str = "",
        before_time: str = "",
        kind: str = "all",
        search: str = "",
    ) -> dict[str, Any]:
        """Страница Истории под фильтром: фразы после before — id последней показанной.

        Если её уже нет в журнале (он длиннее, чем читается), страница начинается
        с фраз раньше before_time — её времени.
        """
        items = self.items()
        ids = [item["id"] for item in items]
        if before in ids:
            items = items[ids.index(before) + 1 :]
        elif before or before_time:
            cutoff = when({"time": before_time})
            items = [item for item in items if when(item) < cutoff]
        found = [item for item in items if matches(item, kind, search)]
        return {"items": found[:limit], "more": len(found) > limit}


# --- Сборка фраз -----------------------------------------------------------------------


def _action(entry: dict[str, Any]) -> dict[str, Any]:
    args = entry.get("args")
    return {
        "action": str(entry.get("action") or ""),
        "args": args if isinstance(args, dict) else {},
        "status": str(entry.get("status") or ""),
        "result": str(entry.get("result") or ""),
        "duration_ms": _number(entry.get("duration_ms")),
    }


def _number(value: Any) -> float:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0


def _phrase_item(entry: dict[str, Any], actions: list[dict[str, Any]]) -> dict[str, Any]:
    commands = entry.get("commands")
    return {
        "id": str(entry.get("request") or ""),
        "time": str(entry.get("time") or ""),
        "phrase": str(entry.get("phrase") or ""),
        "source": str(entry.get("source") or ""),
        "route": "llm" if entry.get("route") == "llm" else "local",
        "commands": [str(c) for c in commands] if isinstance(commands, list) else [],
        "reply": str(entry.get("reply") or ""),
        "ok": bool(entry.get("ok")),
        "error": str(entry.get("error") or ""),
        "duration_ms": _number(entry.get("duration_ms")),
        "actions": actions,
    }


def _legacy_item(entry: dict[str, Any], item_id: str) -> dict[str, Any]:
    llm = entry.get("source") == "llm"
    action = _action(entry)
    ok = action["status"] in _OK
    return {
        "id": item_id,
        "time": str(entry.get("time") or ""),
        "phrase": str(entry.get("phrase") or entry.get("command") or ""),
        "source": "" if llm else str(entry.get("source") or ""),
        "route": "llm" if llm else "local",
        "commands": [str(entry["command"])] if entry.get("command") else [],
        "reply": "" if llm else action["result"],
        "ok": ok,
        "error": "" if ok else action["result"],
        "duration_ms": action["duration_ms"],
        "actions": [action],
        "_last": str(entry.get("time") or ""),
    }


def _same_legacy_phrase(item: dict[str, Any], entry: dict[str, Any]) -> bool:
    llm = entry.get("source") == "llm"
    phrase = str(entry.get("phrase") or entry.get("command") or "")
    command = [str(entry["command"])] if entry.get("command") else []
    near = when(entry) - when({"time": item["_last"]}) <= _LEGACY_GAP
    return (
        near and phrase == item["phrase"] and (item["route"] == "llm") == llm and command == item["commands"]
    )


def _add_legacy_action(item: dict[str, Any], entry: dict[str, Any]) -> None:
    action = _action(entry)
    item["actions"].append(action)
    item["duration_ms"] += action["duration_ms"]
    item["_last"] = str(entry.get("time") or "")
    if item["route"] == "local":
        item["reply"] = action["result"]
    if action["status"] not in _OK and item["ok"]:
        item["ok"], item["error"] = False, action["result"]


def _unfinished_item(request: str, entries: list[dict[str, Any]]) -> dict[str, Any]:
    item = _legacy_item(entries[0], request)
    for entry in entries[1:]:
        _add_legacy_action(item, entry)
    return item
