"""Действия: таймеры и напоминания."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Annotated

from pydantic import Field

from assistant.actions.registry import ActionError, ActionRegistry
from assistant.timers import Alarms, duration_text, local_now, moment_text

_MAX_TIMER = 24 * 3600
_MAX_AHEAD = timedelta(days=366)
_CLOCK = re.compile(r"^(\d{1,2})[:.](\d{2})$")
_DATE_CLOCK = re.compile(r"^(\d{4})-(\d{2})-(\d{2})[ T](\d{1,2})[:.](\d{2})$")


def parse_moment(text: str, now: datetime) -> datetime:
    """«18:30» — сегодня (или завтра, если уже прошло); «2026-10-24 09:00» — в этот день.

    Raises:
        ActionError: время не разобрать или оно уже прошло.
    """
    value = text.strip()
    if match := _CLOCK.match(value):
        hour, minute = int(match[1]), int(match[2])
        if hour > 23 or minute > 59:
            raise ActionError(f"Не понял время «{text}»")
        due = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return due if due > now else due + timedelta(days=1)
    if match := _DATE_CLOCK.match(value):
        try:
            due = now.replace(
                year=int(match[1]), month=int(match[2]), day=int(match[3]),
                hour=int(match[4]), minute=int(match[5]), second=0, microsecond=0,
            )  # fmt: skip
        except ValueError as exc:
            raise ActionError(f"Не понял дату «{text}»") from exc
        if due <= now:
            raise ActionError(f"Время «{text}» уже прошло")
        return due
    raise ActionError(f"Не понял время «{text}»: нужно ЧЧ:ММ или ГГГГ-ММ-ДД ЧЧ:ММ")


def register(registry: ActionRegistry, alarms: Alarms, now: Callable[[], datetime] = local_now) -> None:
    """Регистрирует действия таймеров и напоминаний.

    Args:
        registry: реестр действий.
        alarms: где хранятся таймеры и кто их запускает.
        now: текущее время (подменяется в тестах).
    """

    @registry.action("set_timer", "Поставить таймер: через сколько часов, минут и секунд подать сигнал")
    def set_timer(
        minutes: Annotated[float, Field(ge=0, le=1440, description="Минуты")] = 0,
        seconds: Annotated[float, Field(ge=0, le=86400, description="Секунды")] = 0,
        hours: Annotated[float, Field(ge=0, le=24, description="Часы")] = 0,
        label: Annotated[str, Field(max_length=100, description="Подпись, например «чай»")] = "",
    ) -> str:
        total = hours * 3600 + minutes * 60 + seconds
        if total <= 0:
            raise ActionError("На сколько поставить таймер?")
        if total > _MAX_TIMER:
            raise ActionError("Таймер — не больше чем на сутки; для дальнего срока поставьте напоминание")
        alarm = alarms.add_timer(total, label)
        return f"Поставил {alarm.title()}"

    @registry.action(
        "set_reminder",
        "Напомнить голосом о деле: через сколько минут или к сроку at (ЧЧ:ММ — сегодня или завтра; "
        "другой день — ГГГГ-ММ-ДД ЧЧ:ММ)",
    )
    def set_reminder(
        text: Annotated[str, Field(min_length=1, max_length=300, description="О чём напомнить")],
        minutes: Annotated[float | None, Field(gt=0, le=10080, description="Через сколько минут")] = None,
        at: Annotated[str | None, Field(description="Когда: «18:30» или «2026-10-24 09:00»")] = None,
    ) -> str:
        current = now()
        if (minutes is None) == (at is None or not at.strip()):
            raise ActionError("Когда напомнить: через сколько минут или во сколько?")
        due = current + timedelta(minutes=minutes) if minutes is not None else parse_moment(at or "", current)
        if due - current > _MAX_AHEAD:
            raise ActionError("Напоминание — не дальше чем через год")
        alarms.add_reminder(text, due)
        when = f"через {duration_text(minutes * 60)}" if minutes is not None else moment_text(due, current)
        return f"Напомню {when}: {text.strip()}"

    @registry.action("list_timers", "Узнать, какие таймеры и напоминания поставлены и сколько осталось")
    def list_timers() -> str:
        current = now()
        items = alarms.active()
        if not items:
            return "Таймеров и напоминаний нет"
        parts = []
        for alarm in items:
            if alarm.kind == "timer":
                left = (alarm.due - current).total_seconds()
                parts.append(f"{alarm.title()}, осталось {duration_text(left)}")
            else:
                parts.append(f"{moment_text(alarm.due, current)} {alarm.title()}")
        return "; ".join(parts)

    @registry.action(
        "cancel_timer",
        "Отменить таймер или напоминание по подписи или части текста; без неё — ближайший таймер",
    )
    def cancel_timer(
        what: Annotated[str, Field(max_length=100, description="Подпись или часть текста напоминания")] = "",
        everything: Annotated[bool, Field(description="Отменить все таймеры и напоминания")] = False,
    ) -> str:
        cancelled = alarms.cancel(what, everything=everything)
        if not cancelled:
            return "Нечего отменять" if not what else f"Не нашёл таймер или напоминание «{what}»"
        if len(cancelled) == 1:
            return f"Отменил {cancelled[0].title()}"
        return f"Отменил всё: {len(cancelled)}"
