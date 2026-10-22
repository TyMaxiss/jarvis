"""Интерфейсы озвучки и очередь воспроизведения.

TTSEngine синтезирует звук, Player проигрывает его, Speaker связывает их: say()
возвращается сразу, а синтез и воспроизведение идут в фоновых потоках.
Пока Speaker говорит (speaking=True), голосовой цикл ищет в микрофоне только «стоп».
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from collections.abc import Callable, Iterable
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
    """Очередь озвучки: синтез и воспроизведение — в двух фоновых потоках.

    Следующая фраза синтезируется, пока звучит текущая (не больше одной впрок): у медленных
    движков (Silero, edge-tts) между предложениями нет пауз на синтез.

    Если основной движок не смог озвучить фразу, её произносит запасной, а основной
    пропускается следующие retry_after секунд: без сети edge-tts не ждёт таймаут на каждой фразе.

    Args:
        engine: основной движок синтеза.
        player: проигрыватель.
        fallback: запасной движок, если основной не смог синтезировать фразу.
        volume: множитель громкости.
        retry_after: сколько секунд после сбоя основного движка говорить запасным.
        clock: часы (тестам — свои).
    """

    def __init__(
        self,
        engine: TTSEngine,
        player: Player,
        fallback: TTSEngine | None = None,
        volume: float = 1.0,
        *,
        retry_after: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.engine = engine
        self._player = player
        self._fallback = fallback
        self.volume = volume  # множитель громкости; меняется на лету из настроек
        self._retry_after = retry_after
        self._clock = clock
        self._primary_down_until = 0.0
        self._texts: queue.Queue[_Item | None] = queue.Queue()
        self._sounds: queue.Queue[_Item | None] = queue.Queue()
        self._ahead = threading.Semaphore(1)  # сколько фраз можно синтезировать впрок
        self._cond = threading.Condition()
        self._pending = 0
        self._generation = 0
        self._synth_thread = threading.Thread(target=self._synth_worker, name="tts-synth", daemon=True)
        self._play_thread = threading.Thread(target=self._play_worker, name="tts", daemon=True)
        self._synth_thread.start()
        self._play_thread.start()

    def switch(self, engine: TTSEngine) -> None:
        """Новый основной движок (скачался голос Piper или Silero); прежний становится запасным."""
        self._fallback, self.engine = self.engine, engine
        self._primary_down_until = 0.0

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
            self._texts.put(item)

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
        """Останавливает потоки озвучки и движки, которым есть что останавливать (процесс Silero)."""
        self.stop()
        self._texts.put(None)
        self._synth_thread.join(timeout=5)
        self._play_thread.join(timeout=5)
        for engine in (self.engine, self._fallback):
            close = getattr(engine, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:
                    log.exception("Не удалось остановить %s", engine.name)  # type: ignore[union-attr]

    def _is_current(self, item: _Item) -> bool:
        with self._cond:
            return item.generation == self._generation

    def _done(self) -> None:
        with self._cond:
            self._pending -= 1
            if self._pending == 0:
                self._cond.notify_all()

    def _synth_worker(self) -> None:
        while (item := self._texts.get()) is not None:
            if not self._is_current(item):
                self._done()
                continue
            self._ahead.acquire()  # освобождает поток воспроизведения, когда берёт фразу
            if not self._is_current(item):  # пока ждали, речь прервали
                self._ahead.release()
                self._done()
                continue
            try:
                if item.audio is None:
                    audio, rate = self._synthesize(item.text)
                    item = _Item(item.generation, item.text, audio, rate)
            except Exception:
                log.exception("Ошибка озвучки")
                self._ahead.release()
                self._done()
                continue
            self._sounds.put(item)
        self._sounds.put(None)

    def _play_worker(self) -> None:
        while (item := self._sounds.get()) is not None:
            self._ahead.release()
            try:
                # Сначала сброс, потом проверка: stop() между ними отменит фразу,
                # а stop() после проверки прервёт уже начатое воспроизведение.
                self._player.reset()
                if self._is_current(item) and item.audio is not None and len(item.audio):
                    self._player.play(np.clip(item.audio * self.volume, -1.0, 1.0), item.samplerate)
            except Exception:
                log.exception("Ошибка воспроизведения")
            finally:
                self._done()

    def _synthesize(self, text: str) -> tuple[np.ndarray, int]:
        fallback = self._fallback
        if fallback is not None and self._clock() < self._primary_down_until:
            return fallback.synthesize(text)
        try:
            return self.engine.synthesize(text)
        except Exception:
            if fallback is None:
                raise
            log.warning(
                "%s не смог озвучить фразу — %d с говорю голосом %s",
                self.engine.name, self._retry_after, fallback.name, exc_info=True,
            )  # fmt: skip
            self._primary_down_until = self._clock() + self._retry_after
            return fallback.synthesize(text)


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
