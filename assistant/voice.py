"""Голосовой цикл: микрофон → активация → фраза → Whisper → команда → ответ.

Состояния:
* idle — ждём имя ассистента (Vosk слушает только во время речи) или горячую клавишу;
* listening — активированы, ждём начала фразы (после имени, клавиши или ответа);
* capturing — идёт фраза, ждём паузу;
* holding — push-to-talk в режиме удержания: пишем, пока клавиши зажаты;
* processing — распознаём и выполняем.

Время в цикле считается по полученному звуку (блоки по 32 мс), а не по часам —
с живым микрофоном это одно и то же, а тесты на записях становятся детерминированными.
Пока ассистент говорит, звук с микрофона выбрасывается (полудуплекс).
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

import numpy as np

from assistant.audio.capture import BLOCK_MS, BLOCK_SIZE, AudioSource
from assistant.audio.vad import SegmentEvent, SpeechSegmenter
from assistant.nlu.normalize import Normalizer
from assistant.safety.confirm import Confirmer
from assistant.stt.whisper_stt import Transcript

log = logging.getLogger(__name__)

# После окончания речи ассистента ещё немного не слушаем: эхо и реверберация.
_ECHO_GUARD_MS = 300


class VoiceState(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    CAPTURING = "capturing"
    HOLDING = "holding"
    PROCESSING = "processing"


class StatusSink(Protocol):
    """Куда сообщать о состоянии (консоль, а на этапе 4 — оверлей)."""

    def voice_status(self, state: str, detail: str = "") -> None: ...

    def voice_result(self, result: Any) -> None: ...


class SpeechOutput(Protocol):
    """Озвучка, которой управляет голосовой цикл."""

    @property
    def speaking(self) -> bool: ...

    def play_sound(self, audio: np.ndarray, samplerate: int) -> None: ...

    def stop(self) -> None: ...

    def wait_idle(self, timeout: float | None = None) -> bool: ...


class Spotter(Protocol):
    last_text: str

    def accept(self, block: np.ndarray) -> str | None: ...

    def finish(self) -> str | None: ...

    def reset(self) -> None: ...


class Transcriber(Protocol):
    def transcribe(self, audio: np.ndarray) -> Transcript: ...


class VAD(Protocol):
    def __call__(self, block: np.ndarray) -> float: ...

    def reset(self) -> None: ...


@dataclass(frozen=True)
class VoiceSettings:
    """Поведение голосового цикла."""

    wake_word: bool = True
    followup_seconds: float = 5.0
    listen_timeout: float = 6.0
    push_to_talk_mode: str = "press"
    max_phrase_seconds: float = 15.0


class VoiceLoop:
    """Голосовой цикл в отдельном потоке.

    Args:
        source: источник звука (микрофон).
        vad: детектор речи.
        segmenter: определение границ фразы.
        spotter: распознаватель имени ассистента (None — только push-to-talk).
        stt: распознавание речи.
        handle_text: обработка фразы (конвейер), принимает текст и способ подтверждения.
        speech: озвучка (для полудуплекса и сигналов).
        normalizer: чтобы понять, что во фразе было только имя.
        settings: настройки.
        status: куда сообщать о состоянии.
        confirmer: голосовое подтверждение рискованных действий.
        chime: сигнал «слушаю» (звук и частота) или None.
    """

    def __init__(
        self,
        source: AudioSource,
        vad: VAD,
        segmenter: SpeechSegmenter,
        spotter: Spotter | None,
        stt: Transcriber,
        handle_text: Callable[[str, Confirmer | None], Any],
        speech: SpeechOutput,
        normalizer: Normalizer,
        settings: VoiceSettings,
        status: StatusSink | None = None,
        confirmer: Confirmer | None = None,
        chime: tuple[np.ndarray, int] | None = None,
    ) -> None:
        self._source = source
        self._vad = vad
        self._segmenter = segmenter
        self._spotter = spotter
        self._stt = stt
        self._handle_text = handle_text
        self._speech = speech
        self._normalizer = normalizer
        self._settings = settings
        self._status = status
        self._confirmer = confirmer
        self._chime = chime
        self._events: queue.Queue[str] = queue.Queue()
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None
        self.state = VoiceState.IDLE
        self._clock_ms = 0
        self._deadline_ms = 0
        self._listen_window_ms = 0
        self._mute_until_ms = 0
        self._was_speaking = False
        self._hold_buffer: list[np.ndarray] = []

    # --- Управление (вызывается из потоков горячих клавиш) ---------------------------

    def push_to_talk(self) -> None:
        """Нажата клавиша push-to-talk (режим «нажал и говоришь»)."""
        self._events.put("ptt")

    def hold_start(self) -> None:
        """Клавиши push-to-talk зажаты (режим удержания)."""
        self._speech.stop()
        self._events.put("hold_start")

    def hold_end(self) -> None:
        """Клавиши push-to-talk отпущены."""
        self._events.put("hold_end")

    def stop_speech(self) -> None:
        """Прервать речь ассистента и прослушивание."""
        self._speech.stop()
        self._events.put("stop")

    # --- Поток ---------------------------------------------------------------------

    def start(self) -> None:
        """Запускает цикл в фоновом потоке."""
        self._thread = threading.Thread(target=self.run, name="voice", daemon=True)
        self._thread.start()

    def close(self) -> None:
        """Останавливает цикл."""
        self._stopped.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def run(self, stop_when_exhausted: bool = False) -> None:
        """Основной цикл. stop_when_exhausted — для тестов: выйти, когда звук кончился."""
        self._enter_idle()
        empty_reads = 0
        while not self._stopped.is_set():
            self.handle_events()
            block = self._source.read(timeout=0.1)
            if block is None:
                empty_reads += 1
                if stop_when_exhausted and empty_reads > 3:
                    return
                continue
            empty_reads = 0
            try:
                self.feed(block)
            except Exception:
                log.exception("Ошибка в голосовом цикле")
                self._enter_idle()

    def feed(self, block: np.ndarray) -> None:
        """Обрабатывает один блок звука (32 мс)."""
        self._clock_ms += BLOCK_MS
        if self._speech.speaking:
            self._was_speaking = True
            return
        if self._was_speaking:
            # Ассистент договорил: отбрасываем эхо и начинаем фразу с чистого листа.
            # Окно ожидания отсчитывается от конца речи, а не от начала ответа.
            self._was_speaking = False
            self._mute_until_ms = self._clock_ms + _ECHO_GUARD_MS
            self._reset_audio_state()
            if self.state is VoiceState.LISTENING:
                self._deadline_ms = self._mute_until_ms + self._listen_window_ms
        if self._clock_ms < self._mute_until_ms:
            return

        if self.state is VoiceState.HOLDING:
            self._hold_buffer.append(block)
            if len(self._hold_buffer) * BLOCK_MS >= self._settings.max_phrase_seconds * 1000:
                self._process(np.concatenate(self._hold_buffer), "hold")
            return

        event = self._segmenter.feed(block, self._vad(block))
        if self.state is VoiceState.IDLE:
            self._idle_block(block, event)
        elif self.state is VoiceState.LISTENING:
            if event is SegmentEvent.START:
                self._set_state(VoiceState.CAPTURING)
            elif self._clock_ms >= self._deadline_ms and not self._segmenter.in_speech:
                self._enter_idle()
        elif self.state is VoiceState.CAPTURING:
            if event is SegmentEvent.END:
                self._process(self._segmenter.segment_audio, "listen")
            elif event is SegmentEvent.DISCARD:
                self._set_state(VoiceState.LISTENING)

    # --- Состояния ------------------------------------------------------------------

    def _idle_block(self, block: np.ndarray, event: SegmentEvent) -> None:
        """Ожидание имени: Vosk получает только звук, в котором есть речь."""
        if self._spotter is None or not self._settings.wake_word:
            return
        heard = None
        if event is SegmentEvent.START:
            audio = self._segmenter.current_audio()
            for start in range(0, len(audio) - BLOCK_SIZE + 1, BLOCK_SIZE):
                heard = self._spotter.accept(audio[start : start + BLOCK_SIZE]) or heard
        elif self._segmenter.in_speech:
            heard = self._spotter.accept(block)
        elif event in (SegmentEvent.END, SegmentEvent.DISCARD):
            if self._spotter.finish():
                log.info("Услышано имя ассистента (в конце фразы)")
                self._on_wake_after_phrase()
            return
        if heard:
            # Имя услышано посреди фразы — дослушиваем её целиком и распознаём Whisper.
            log.info("Услышано имя ассистента")
            self._spotter.reset()
            self._set_state(VoiceState.CAPTURING)

    def _on_wake_after_phrase(self) -> None:
        """Имя распознано, когда фраза уже закончилась."""
        only_name = self._spotter is not None and self._normalizer.normalize(self._spotter.last_text) == ""
        if only_name:
            # Сказали только «Джарвис» — Whisper не нужен, сразу слушаем команду.
            self._listen(chime=True)
        else:
            self._process(self._segmenter.segment_audio, "wake")

    def _enter_idle(self) -> None:
        self._hold_buffer = []
        self._reset_audio_state()
        self._set_state(VoiceState.IDLE)

    def _listen(self, chime: bool, seconds: float | None = None) -> None:
        """Переход к ожиданию фразы без имени.

        Если ассистент сейчас говорит (ответ или сигнал), окно отсчитывается от конца речи.
        """
        self._listen_window_ms = int(
            (seconds if seconds is not None else self._settings.listen_timeout) * 1000
        )
        self._reset_audio_state()
        if chime and self._chime is not None:
            audio, rate = self._chime
            self._speech.play_sound(audio, rate)
        self._deadline_ms = self._clock_ms + self._listen_window_ms
        self._set_state(VoiceState.LISTENING)

    def _set_state(self, state: VoiceState) -> None:
        if state is not self.state:
            log.debug("Голос: %s", state)
        self.state = state
        self._report(state.value)

    def _report(self, kind: str, detail: str = "") -> None:
        if self._status is not None:
            self._status.voice_status(kind, detail)

    def _reset_audio_state(self) -> None:
        self._vad.reset()
        self._segmenter.reset()
        if self._spotter is not None:
            self._spotter.reset()

    def handle_events(self) -> None:
        """Обрабатывает накопившиеся события горячих клавиш (вызывается в потоке цикла)."""
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                return
            if event == "ptt" and self.state in (VoiceState.IDLE, VoiceState.LISTENING):
                self._speech.stop()
                self._source.flush()
                self._listen(chime=True)
            elif event == "hold_start" and self.state is not VoiceState.PROCESSING:
                self._source.flush()
                self._hold_buffer = []
                self._set_state(VoiceState.HOLDING)
            elif event == "hold_end" and self.state is VoiceState.HOLDING:
                audio = np.concatenate(self._hold_buffer) if self._hold_buffer else np.zeros(0, np.int16)
                self._process(audio, "hold")
            elif event == "stop" and self.state is not VoiceState.PROCESSING:
                self._enter_idle()

    # --- Обработка фразы ------------------------------------------------------------

    def _process(self, audio: np.ndarray, origin: str) -> None:
        """Распознаёт фразу, выполняет её и решает, что слушать дальше."""
        self._set_state(VoiceState.PROCESSING)
        self._report("recognizing")
        self._hold_buffer = []
        if len(audio) < BLOCK_SIZE * 4:  # меньше ~130 мс — не фраза
            self._enter_idle()
            return
        transcript = self._stt.transcribe(audio)
        log.info(
            "Конец речи → текст: %.0f мс (звук %.1f с): «%s»",
            transcript.duration_ms, len(audio) / 16_000, transcript.text,
        )  # fmt: skip
        if not transcript.text:
            self._enter_idle()
            return
        if not self._normalizer.normalize(transcript.text):
            # Во фразе было только имя — ждём саму команду.
            self._listen(chime=True)
            return
        self._report("recognized", transcript.text)
        started = time.perf_counter()
        result = self._handle_text(transcript.text, self._confirmer)
        log.info("Текст → результат: %.0f мс", (time.perf_counter() - started) * 1000)
        if self._status is not None and result is not None:
            self._status.voice_result(result)
        self._after_response()

    def _after_response(self) -> None:
        """После ответа: дождаться конца речи, затем слушать продолжение без имени."""
        self._source.flush()
        if self._settings.followup_seconds > 0:
            self._listen(chime=False, seconds=self._settings.followup_seconds)
        else:
            self._reset_audio_state()
            self._enter_idle()
