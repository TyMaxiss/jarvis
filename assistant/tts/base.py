"""Интерфейсы озвучки и очередь воспроизведения.

TTSEngine синтезирует звук, Player проигрывает его, Speaker связывает их: say()
возвращается сразу, а синтез и воспроизведение идут в отдельном потоке.
Пока Speaker говорит (speaking=True), голосовой цикл ищет в микрофоне только «стоп».
"""

from __future__ import annotations

import logging
import queue
import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)

_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")


class TTSError(RuntimeError):
    """Движок озвучки недоступен (нет модели, голоса или библиотеки)."""


class Speaker(Protocol):
    """То, что умеет произнести (или показать) ответ ассистента."""

    def say(self, text: str) -> None:
        """Произносит текст; не должен надолго блокировать вызывающий поток."""
        ...


class TTSEngine(Protocol):
    """Синтез речи."""

    name: str

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Возвращает звук (float32, моно, [-1, 1]) и частоту дискретизации."""
        ...


class Player(Protocol):
    """Воспроизведение звука."""

    def reset(self) -> None:
        """Снимает флаг остановки перед следующей фразой."""
        ...

    def play(self, audio: np.ndarray, samplerate: int) -> bool:
        """Проигрывает звук до конца; False, если воспроизведение прервали."""
        ...

    def stop(self) -> None:
        """Прерывает текущее воспроизведение (из любого потока)."""
        ...


def split_sentences(text: str) -> list[str]:
    """Делит текст на предложения: озвучка первого начинается, не дожидаясь остальных."""
    return [part.strip() for part in _SENTENCE_RE.split(text.strip()) if part.strip()]


@dataclass(frozen=True)
class _Item:
    generation: int
    text: str = ""
    audio: np.ndarray | None = None
    samplerate: int = 0


class SpeechQueue:
    """Очередь озвучки с фоновым потоком.

    Args:
        engine: основной движок синтеза.
        player: проигрыватель.
        fallback: запасной движок, если основной не смог синтезировать фразу.
        volume: множитель громкости.
    """

    def __init__(
        self,
        engine: TTSEngine,
        player: Player,
        fallback: TTSEngine | None = None,
        volume: float = 1.0,
    ) -> None:
        self.engine = engine
        self._player = player
        self._fallback = fallback
        self.volume = volume  # множитель громкости; меняется на лету из настроек
        self._queue: queue.Queue[_Item | None] = queue.Queue()
        self._cond = threading.Condition()
        self._pending = 0
        self._generation = 0
        self._thread = threading.Thread(target=self._worker, name="tts", daemon=True)
        self._thread.start()

    @property
    def speaking(self) -> bool:
        """Есть что произносить или идёт воспроизведение."""
        with self._cond:
            return self._pending > 0

    def say(self, text: str) -> None:
        """Ставит текст в очередь (по предложениям) и сразу возвращается."""
        self._enqueue(_Item(0, text=s) for s in split_sentences(text))

    def play_sound(self, audio: np.ndarray, samplerate: int) -> None:
        """Ставит в очередь готовый звук (например, сигнал «слушаю»)."""
        self._enqueue([_Item(0, audio=audio, samplerate=samplerate)])

    def _enqueue(self, items: Iterable[_Item]) -> None:
        with self._cond:
            batch = [_Item(self._generation, i.text, i.audio, i.samplerate) for i in items]
            self._pending += len(batch)
        for item in batch:
            self._queue.put(item)

    def stop(self) -> None:
        """Прерывает речь и выбрасывает всё, что ещё не произнесено."""
        with self._cond:
            self._generation += 1
        self._player.stop()

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Ждёт, пока всё будет произнесено. False — не дождались за timeout."""
        with self._cond:
            return self._cond.wait_for(lambda: self._pending == 0, timeout)

    def close(self) -> None:
        """Останавливает поток озвучки."""
        self.stop()
        self._queue.put(None)
        self._thread.join(timeout=5)

    def _is_current(self, item: _Item) -> bool:
        with self._cond:
            return item.generation == self._generation

    def _worker(self) -> None:
        while (item := self._queue.get()) is not None:
            try:
                # Сначала сброс, потом проверка: stop() между ними отменит фразу,
                # а stop() после проверки прервёт уже начатое воспроизведение.
                self._player.reset()
                if not self._is_current(item):
                    continue
                if item.audio is not None:
                    audio, rate = item.audio, item.samplerate
                else:
                    audio, rate = self._synthesize(item.text)
                if self._is_current(item) and len(audio):
                    self._player.play(np.clip(audio * self.volume, -1.0, 1.0), rate)
            except Exception:
                log.exception("Ошибка озвучки")
            finally:
                with self._cond:
                    self._pending -= 1
                    if self._pending == 0:
                        self._cond.notify_all()

    def _synthesize(self, text: str) -> tuple[np.ndarray, int]:
        try:
            return self.engine.synthesize(text)
        except Exception:
            if self._fallback is None:
                raise
            log.exception("%s не смог озвучить фразу — использую %s", self.engine.name, self._fallback.name)
            return self._fallback.synthesize(text)


class TeeSpeaker:
    """Передаёт ответ нескольким «говорящим»: например, в консоль и в озвучку."""

    def __init__(self, *speakers: Speaker) -> None:
        self._speakers = speakers

    def say(self, text: str) -> None:
        for speaker in self._speakers:
            speaker.say(text)


class SilentSpeech:
    """Заглушка озвучки, когда ни Piper, ни SAPI недоступны (ответы только в консоли)."""

    speaking = False

    def say(self, text: str) -> None:
        """Ничего не произносит."""

    def play_sound(self, audio: np.ndarray, samplerate: int) -> None:
        """Ничего не проигрывает."""

    def stop(self) -> None:
        """Нечего останавливать."""

    def wait_idle(self, timeout: float | None = None) -> bool:
        return True
