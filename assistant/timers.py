"""Таймеры и напоминания: срабатывают в своё время и переживают перезапуск программы.

Поток проверяет время по часам (а не по счётчику с запуска): так сон компьютера и перевод
часов не сбивают напоминание «в 18:30». Список хранится в JSON-файле. Что сработало,
пока программа была закрыта, объявляется при запуске как пропущенное.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

log = logging.getLogger(__name__)

Kind = Literal["timer", "reminder"]
_CHECK_EVERY = 30.0  # даже без изменений поток просыпается: часы могли перевести


def local_now() -> datetime:
    """Текущее время с часовым поясом."""
    return datetime.now().astimezone()


def _plural(number: int, forms: tuple[str, str, str]) -> str:
    tens, ones = number % 100, number % 10
    if 10 < tens < 20:
        return forms[2]
    if ones == 1:
        return forms[0]
    return forms[1] if 2 <= ones <= 4 else forms[2]


def duration_text(seconds: float) -> str:
    """«5 минут», «1 час 30 минут», «45 секунд» — в форме «на сколько»."""
    total = max(1, round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    parts = []
    if hours:
        parts.append(f"{hours} {_plural(hours, ('час', 'часа', 'часов'))}")
    if minutes:
        parts.append(f"{minutes} {_plural(minutes, ('минуту', 'минуты', 'минут'))}")
    if secs and not hours:
        parts.append(f"{secs} {_plural(secs, ('секунду', 'секунды', 'секунд'))}")
    return " ".join(parts)


def moment_text(due: datetime, now: datetime) -> str:
    """Когда сработает: «в 18:30», «завтра в 9:00», «24.10 в 9:00»."""
    clock = f"{due.hour}:{due.minute:02d}"
    days = (due.date() - now.date()).days
    if days == 0:
        return f"в {clock}"
    if days == 1:
        return f"завтра в {clock}"
    return f"{due:%d.%m} в {clock}"


@dataclass
class Alarm:
    """Таймер или напоминание."""

    id: str
    kind: Kind
    due: datetime  # когда сработать (с часовым поясом)
    text: str  # подпись таймера или о чём напомнить
    seconds: float = 0.0  # на сколько поставлен таймер

    def title(self) -> str:
        """Как назвать: «таймер на 5 минут «чай»» или «напоминание: позвонить маме»."""
        if self.kind == "timer":
            label = f" «{self.text}»" if self.text else ""
            return f"таймер на {duration_text(self.seconds)}{label}"
        return f"напоминание: {self.text}"

    def message(self, missed: bool = False) -> str:
        """Что сказать, когда сработало."""
        if self.kind == "timer":
            label = f" «{self.text}»" if self.text else ""
            text = f"Таймер на {duration_text(self.seconds)}{label}: время вышло"
        else:
            text = f"Напоминаю: {self.text}"
        if missed:
            text = f"Пока программа была закрыта, в {self.due:%H:%M} сработало: {text[0].lower()}{text[1:]}"
        return text

    def as_dict(self) -> dict[str, object]:
        """Для файла и для окна (время — ISO с часовым поясом)."""
        return {**asdict(self), "due": self.due.isoformat(timespec="seconds")}


class Alarms:
    """Таймеры и напоминания с потоком, который их запускает.

    Args:
        path: файл со списком (None — только в памяти).
        now: текущее время с часовым поясом (подменяется в тестах).
    """

    def __init__(self, path: Path | None, now: Callable[[], datetime] = local_now) -> None:
        self.path = path
        self._now = now
        self._items: list[Alarm] = []
        self._cond = threading.Condition()
        self._stopped = False
        self._thread: threading.Thread | None = None
        # Что сделать, когда сработало (alarm, пропущено ли) и когда список изменился.
        self.on_fire: Callable[[Alarm, bool], None] | None = None
        self.on_change: Callable[[], None] | None = None

    # --- Список ----------------------------------------------------------------------

    def active(self) -> list[Alarm]:
        """Поставленные таймеры и напоминания, ближайшие первыми."""
        with self._cond:
            return sorted(self._items, key=lambda alarm: alarm.due)

    def add_timer(self, seconds: float, label: str = "") -> Alarm:
        """Таймер на столько-то секунд."""
        due = self._now() + timedelta(seconds=seconds)
        return self._add(Alarm(_new_id(), "timer", due, label.strip(), seconds))

    def add_reminder(self, text: str, due: datetime) -> Alarm:
        """Напоминание к сроку."""
        return self._add(Alarm(_new_id(), "reminder", due, text.strip()))

    def cancel(self, query: str = "", alarm_id: str = "", everything: bool = False) -> list[Alarm]:
        """Отменяет по id, по подписи или тексту, все сразу или (без условий) ближайший таймер."""
        with self._cond:
            needle = query.strip().lower()
            if everything:
                chosen = list(self._items)
            elif alarm_id:
                chosen = [a for a in self._items if a.id == alarm_id]
            elif needle:
                chosen = [a for a in self._items if needle in a.text.lower()]
            else:
                timers = sorted((a for a in self._items if a.kind == "timer"), key=lambda a: a.due)
                chosen = timers[:1] or sorted(self._items, key=lambda a: a.due)[:1]
            self._items = [a for a in self._items if a not in chosen]
            self._cond.notify_all()
        if chosen:
            self._changed()
        return chosen

    def _add(self, alarm: Alarm) -> Alarm:
        with self._cond:
            self._items.append(alarm)
            self._cond.notify_all()
        log.info("Поставлено: %s (%s)", alarm.title(), alarm.due.isoformat(timespec="seconds"))
        self._changed()
        return alarm

    def _changed(self) -> None:
        self._save()
        if self.on_change is not None:
            self.on_change()

    # --- Срабатывание ----------------------------------------------------------------

    def tick(self, missed: bool = False) -> list[Alarm]:
        """Запускает всё, чему пора; возвращает сработавшее."""
        now = self._now()
        with self._cond:
            due = sorted((a for a in self._items if a.due <= now), key=lambda a: a.due)
            self._items = [a for a in self._items if a not in due]
        if not due:
            return []
        self._changed()
        for alarm in due:
            log.info("Сработало%s: %s", " (пропущено)" if missed else "", alarm.title())
            if self.on_fire is not None:
                try:
                    self.on_fire(alarm, missed)
                except Exception:
                    log.exception("Ошибка при срабатывании: %s", alarm.title())
        return due

    def start(self) -> None:
        """Читает файл, объявляет пропущенное и запускает поток."""
        self._load()
        self.tick(missed=True)
        self._thread = threading.Thread(target=self._run, name="alarms", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Останавливает поток (список уже сохранён)."""
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while True:
            with self._cond:
                if self._stopped:
                    return
                upcoming = min((a.due for a in self._items), default=None)
                wait = _CHECK_EVERY
                if upcoming is not None:
                    wait = min(wait, max(0.0, (upcoming - self._now()).total_seconds()))
                self._cond.wait(timeout=wait)
                if self._stopped:
                    return
            self.tick()

    # --- Файл ------------------------------------------------------------------------

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            items = [
                Alarm(
                    id=str(item["id"]),
                    kind="timer" if item["kind"] == "timer" else "reminder",
                    due=datetime.fromisoformat(item["due"]),
                    text=str(item.get("text", "")),
                    seconds=float(item.get("seconds", 0)),
                )
                for item in data.get("alarms", [])
            ]
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            log.warning("Не удалось прочитать %s: %s — таймеров нет", self.path.name, exc)
            return
        with self._cond:
            self._items = items
        log.info("Таймеров и напоминаний из %s: %d", self.path.name, len(items))

    def _save(self) -> None:
        if self.path is None:
            return
        with self._cond:
            data = {"alarms": [alarm.as_dict() for alarm in self._items]}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(self.path.name + ".tmp")
            temporary.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            temporary.replace(self.path)
        except OSError:
            log.exception("Не удалось сохранить %s", self.path)


def _new_id() -> str:
    return uuid.uuid4().hex[:8]
