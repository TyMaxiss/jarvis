"""Громкость системы через Core Audio (pycaw)."""

from __future__ import annotations

import logging
import threading
from typing import Annotated, Any

from pydantic import Field

from assistant.actions.registry import ActionError, ActionRegistry

log = logging.getLogger(__name__)


class VolumeController:
    """Громкость устройства вывода по умолчанию.

    Устройство запрашивается заново при каждом обращении: пользователь мог переключить
    вывод (наушники, колонки). COM инициализируется в каждом потоке, который вызывает
    контроллер.
    """

    def __init__(self) -> None:
        self._com_ready = threading.local()

    def _endpoint(self) -> Any:
        """Интерфейс IAudioEndpointVolume устройства вывода по умолчанию.

        С pycaw 2026 GetSpeakers() возвращает обёртку AudioDevice со свойством
        EndpointVolume; старый вызов device.Activate(...) больше не работает.
        """
        try:
            import comtypes
            from pycaw.pycaw import AudioUtilities
        except ImportError as exc:
            raise ActionError("Управление громкостью доступно только в Windows") from exc
        if not getattr(self._com_ready, "done", False):
            try:
                comtypes.CoInitialize()
            except OSError:
                # COM в этом потоке уже инициализирован в другом режиме — работать можно.
                log.debug("COM уже инициализирован в потоке %s", threading.current_thread().name)
            self._com_ready.done = True
        try:
            return AudioUtilities.GetSpeakers().EndpointVolume
        except Exception as exc:  # COMError, если устройства вывода нет
            log.exception("Не удалось получить устройство вывода звука")
            raise ActionError("Не нашёл устройство вывода звука") from exc

    def get(self) -> int:
        """Текущая громкость в процентах."""
        return round(self._endpoint().GetMasterVolumeLevelScalar() * 100)

    def set(self, percent: int) -> int:
        """Устанавливает громкость; при ненулевой громкости снимает «без звука»."""
        percent = max(0, min(100, percent))
        endpoint = self._endpoint()
        endpoint.SetMasterVolumeLevelScalar(percent / 100, None)
        if percent > 0 and endpoint.GetMute():
            endpoint.SetMute(0, None)
        return percent

    def is_muted(self) -> bool:
        """Включён ли режим «без звука»."""
        return bool(self._endpoint().GetMute())

    def set_muted(self, muted: bool) -> None:
        """Включает или выключает режим «без звука»."""
        self._endpoint().SetMute(1 if muted else 0, None)


def register(registry: ActionRegistry, controller: VolumeController | None = None) -> None:
    """Регистрирует действия громкости."""
    volume = controller or VolumeController()

    @registry.action("volume_set", "Установить громкость системы в процентах (0–100)")
    def volume_set(
        percent: Annotated[int, Field(ge=0, le=100, description="Громкость в процентах")],
    ) -> str:
        return f"Громкость {volume.set(percent)}%"

    @registry.action("volume_change", "Изменить громкость на указанное число процентов (минус — тише)")
    def volume_change(
        delta: Annotated[int, Field(ge=-100, le=100, description="На сколько процентов: 10 или -10")],
    ) -> str:
        return f"Громкость {volume.set(volume.get() + delta)}%"

    @registry.action("mute", "Выключить или включить звук; без параметра — переключить")
    def mute(
        state: Annotated[
            bool | None,
            Field(description="true — выключить звук, false — включить, не указано — переключить"),
        ] = None,
    ) -> str:
        muted = not volume.is_muted() if state is None else state
        volume.set_muted(muted)
        return "Звук выключен" if muted else "Звук включён"
