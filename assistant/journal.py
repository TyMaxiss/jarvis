"""JSONL-журнал: строка на каждое действие (kind: action) и на каждую фразу (kind: phrase).

Действия одной фразы и сама фраза связаны полем request — по нему страница «История»
собирает, что было сказано, куда ушла фраза и что из этого вышло. Строки старого
формата (до этапа 5) — только действия, без kind и request.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from assistant.logging_setup import SecretMasker

log = logging.getLogger(__name__)


class ActionJournal:
    """Потокобезопасная запись журнала.

    Args:
        path: файл журнала (папка создаётся).
        masker: маскировщик секретов — ключи не должны попасть и в журнал.
    """

    def __init__(self, path: Path, masker: SecretMasker) -> None:
        self.path = path
        self._masker = masker
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        *,
        source: str,
        action: str,
        args: dict[str, Any],
        status: str,
        result: str,
        duration_ms: float,
        command_id: str | None = None,
        phrase: str | None = None,
        request: str | None = None,
    ) -> None:
        """Добавляет запись о действии. Ошибка записи логируется, но не прерывает работу."""
        self._append(
            {
                "kind": "action",
                "request": request,
                "source": source,
                "command": command_id,
                "phrase": phrase,
                "action": action,
                "args": args,
                "status": status,
                "result": result,
                "duration_ms": round(duration_ms, 1),
            }
        )

    def write_phrase(
        self,
        *,
        request: str,
        phrase: str,
        source: str,
        route: str,
        commands: list[str],
        reply: str,
        ok: bool,
        duration_ms: float,
        error: str = "",
    ) -> None:
        """Добавляет запись о фразе: её действия уже записаны с тем же request."""
        self._append(
            {
                "kind": "phrase",
                "request": request,
                "source": source,
                "phrase": phrase,
                "route": route,
                "commands": commands,
                "reply": reply,
                "ok": ok,
                "error": error,
                "duration_ms": round(duration_ms, 1),
            }
        )

    def _append(self, entry: dict[str, Any]) -> None:
        entry = {"time": datetime.now().astimezone().isoformat(timespec="milliseconds"), **entry}
        line = self._masker.mask(json.dumps(entry, ensure_ascii=False, default=str))
        try:
            with self._lock, self.path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")
        except OSError:
            log.exception("Не удалось записать журнал действий %s", self.path)
