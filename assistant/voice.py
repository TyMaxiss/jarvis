"""Голосовой цикл: микрофон → активация → фраза → Whisper → команда → ответ.

Состояния:
* idle — ждём имя ассистента (Vosk слушает только во время речи) или горячую клавишу;
* listening — активированы, ждём начала фразы (после имени, клавиши или ответа);
* capturing — идёт фраза, ждём паузу;
* holding — push-to-talk в режиме удержания: пишем, пока клавиши зажаты;
* processing — распознаём и выполняем.

Время в цикле считается по полученному звуку (блоки по 32 мс), а не по часам —
с живым микрофоном это одно и то же, а тесты на записях становятся детерминированными.
Пока ассистент говорит, имя и команды не слушаются (полудуплекс). Короткий сигнал
«слушаю» — исключение: команду часто начинают говорить, не дождавшись его конца.

«Стоп» / «отмена» (STOP_PHRASES, можно с именем) останавливают ассистента в любом состоянии:
* пока выполняется задача (голосовая — в отдельном потоке, из текста — в конвейере) и пока
  ассистент говорит, слушаются только они: Vosk со своей грамматикой, решение в конце фразы;
  поверх речи ассистента (колонки) засчитывается и «Джарвис, стоп» посреди фразы — если Vosk
  его расслышал: без подавления эха голос ассистента обычно заглушает «стоп»;
* в ожидании команды и продолжения стоп-фразу распознаёт Whisper, и она не уходит в конвейер;
* «Джарвис, стоп» в покое Vosk слышит сразу, без Whisper.
Кнопка на главной и Ctrl+Alt+Q останавливают через interrupt() — и фразу, которая ещё
распознаётся.
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
from assistant.audio.wakeword import SpotResult
from assistant.nlu.names import skeleton
from assistant.nlu.normalize import Normalizer, basic_clean
from assistant.nlu.stop import STOP_PHRASES, is_stop_phrase
from assistant.safety.confirm import Confirmer
from assistant.stt.whisper_stt import Transcript

log = logging.getLogger(__name__)

# После окончания речи ассистента ещё немного не слушаем: эхо и реверберация.
_ECHO_GUARD_MS = 300
# Сколько после начала сигнала «слушаю» ещё может длиться его воспроизведение (задержка вывода).
_CHIME_MARGIN_MS = 500
# Слова-приманки, похожие на «стоп» по звучанию: их нет в грамматике стоп-слов. Калибровка
# на 4 голосах Piper: без них «стоп» слышен в 30 случаях из 40 вместо 28, ложных
# срабатываний на 148 обычных фразах и ответах ассистента — ноль, как и с ними.
# Совсем без приманок — 24 ложных. С этой грамматикой одиночное «Стоп.» эталонного голоса
# слышно с уверенностью 1.00 (раньше Vosk слышал «тот»).
STOP_SOUNDALIKES = ("то", "тот", "это", "этот", "что", "кто", "вот", "как", "сам", "там", "тут", "так",
                    "потом", "пока")  # fmt: skip
_VOWELS = frozenset("аеёиоуыэюяaeiouy")


def name_only_limit_ms(name: str) -> int:
    """Самая долгая речь, которую ещё можно считать одним именем без команды.

    «Джарвис» (2 слога) звучит ~0,5–0,7 с, предел — 800 мс; на каждый слог длинного
    имени — ещё 250 мс. Речь длиннее предела отправляется в Whisper.
    """
    syllables = sum(1 for char in name.lower() if char in _VOWELS)
    return 300 + 250 * max(2, syllables)


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
    def accept(self, block: np.ndarray) -> SpotResult | None: ...

    def finish(self) -> SpotResult: ...

    def reset(self) -> None: ...


class Dictator(Protocol):
    """Слова фразы по ходу речи (Dictation): accept отдаёт услышанное, когда оно изменилось."""

    def accept(self, block: np.ndarray) -> str | None: ...

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
    wake_confidence: float = 0.8  # минимальная уверенность Vosk в имени
    stop_confidence: float = 0.8  # минимальная уверенность Vosk в «стоп» / «отмена»
    verify_wake_word: bool = True  # имя должно быть и в тексте от Whisper
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
            Если у результата handled=False («Не знаю такой команды»), продолжение
            без имени не слушается.
        speech: озвучка (для полудуплекса и сигналов).
        normalizer: чтобы понять, что во фразе было только имя.
        settings: настройки.
        wake_names: имя ассистента и варианты его написания (для Vosk — первое).
        status: куда сообщать о состоянии.
        confirmer: голосовое подтверждение рискованных действий; пока оно слушает ответ
            (свойство listening), цикл не читает микрофон.
        chime: сигнал «слушаю» (звук и частота) или None.
        stop_spotter: распознаватель STOP_PHRASES (и имени: «Джарвис, стоп»), лучше без
            приманок STOP_SOUNDALIKES; None — голосом не прерывается ничего, фраза
            выполняется в потоке цикла.
        on_stop: что ещё сделать по «стоп», кроме остановки речи (прервать задачу).
        busy: конвейер выполняет задачу не из голоса (текст, горячая клавиша команды) —
            тогда цикл в покое слушает только «стоп».
        dictation: слова фразы по ходу речи для оверлея (сообщаются как partial); None — без них.
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
        wake_names: list[str] | None = None,
        status: StatusSink | None = None,
        confirmer: Confirmer | None = None,
        chime: tuple[np.ndarray, int] | None = None,
        stop_spotter: Spotter | None = None,
        on_stop: Callable[[], None] | None = None,
        busy: Callable[[], bool] | None = None,
        dictation: Dictator | None = None,
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
        names = [basic_clean(n) for n in (wake_names or ["джарвис"]) if basic_clean(n)]
        self._wake_phrase = names[0]
        self._wake_skeletons = {skeleton(n) for n in names}
        self._name_only_ms = name_only_limit_ms(self._wake_phrase)
        self._heard = SpotResult()
        self._status = status
        self._confirmer = confirmer
        self._chime = chime
        self._stop_spotter = stop_spotter
        self._on_stop = on_stop
        self._busy = busy
        self._dictation = dictation
        self._dictated = ""  # что уже показано из слов текущей фразы
        self._stop_words = frozenset(" ".join((*STOP_PHRASES, self._wake_phrase)).split())
        self._named_stops = [f"{self._wake_phrase} {phrase}" for phrase in STOP_PHRASES]
        self._events: queue.Queue[str] = queue.Queue()
        self._stop_requested = threading.Event()  # кнопка или Ctrl+Alt+Q, пока фраза распознаётся
        self._stopped = threading.Event()
        self._paused = False
        self._thread: threading.Thread | None = None
        self.state = VoiceState.IDLE
        self._clock_ms = 0
        self._deadline_ms = 0
        self._listen_window_ms = 0
        self._mute_until_ms = 0
        self._chime_until_ms = 0
        self._stop_only = False  # ассистент говорит или занят: слушаем только «стоп»
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

    def interrupt(self) -> None:
        """Остановить речь, прослушивание и фразу, которая ещё распознаётся (кнопка, Ctrl+Alt+Q)."""
        self._stop_requested.set()
        self._speech.stop()
        self._events.put("stop")

    def set_paused(self, paused: bool) -> None:
        """Пауза: микрофон не слушается — ни имя, ни горячая клавиша (трей, интерфейс)."""
        self._events.put("pause" if paused else "resume")

    @property
    def paused(self) -> bool:
        """Прослушивание на паузе."""
        return self._paused

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
        borrowed = False
        while not self._stopped.is_set():
            self.handle_events()
            if self._mic_borrowed():
                # Команда из текста или по горячей клавише спрашивает «да/нет» голосом:
                # микрофон слушает подтверждение, а цикл ждёт.
                borrowed = True
                self._stopped.wait(0.05)
                continue
            if borrowed:
                borrowed = False
                self._interrupt_phrase()
            block = self._source.read(timeout=0.1)
            if block is None:
                empty_reads += 1
                if stop_when_exhausted and empty_reads > 3:
                    return
                continue
            empty_reads = 0
            if self._paused:  # звук читается, чтобы не копился в очереди, но не слушается
                continue
            try:
                self.feed(block)
            except Exception:
                log.exception("Ошибка в голосовом цикле")
                self._enter_idle()

    def feed(self, block: np.ndarray) -> None:
        """Обрабатывает один блок звука (32 мс)."""
        self._clock_ms += BLOCK_MS
        speaking = self._speech.speaking and self._clock_ms > self._chime_until_ms
        if speaking or (self.state is VoiceState.IDLE and self._busy is not None and self._busy()):
            if not self._stop_only:
                self._stop_only = True
                self._interrupt_phrase()  # в речи ассистента и во время задачи ищем только «стоп»
            if self._stop_block(block, over_speech=speaking):
                self._enter_idle()
            return
        if self._stop_only:
            # Ассистент договорил (или задача кончилась): отбрасываем эхо и начинаем фразу
            # с чистого листа. Окно ожидания отсчитывается от конца речи, а не от начала ответа.
            self._stop_only = False
            self._mute_until_ms = self._clock_ms + _ECHO_GUARD_MS
            self._reset_audio_state()
            if self.state is VoiceState.LISTENING:
                self._deadline_ms = self._mute_until_ms + self._listen_window_ms
        if self._clock_ms < self._mute_until_ms:
            return

        if self.state is VoiceState.HOLDING:
            self._hold_buffer.append(block)
            self._dictate(block)
            if len(self._hold_buffer) * BLOCK_MS >= self._settings.max_phrase_seconds * 1000:
                self._process(np.concatenate(self._hold_buffer), "hold")
            return

        event = self._segmenter.feed(block, self._vad(block))
        if self.state is VoiceState.IDLE:
            self._idle_block(block, event)
        elif self.state is VoiceState.LISTENING:
            if event is SegmentEvent.START:
                self._set_state(VoiceState.CAPTURING)
                # Начало фразы (с запасом до неё) уже в сегментаторе — слова считаются и по нему.
                audio = self._segmenter.current_audio()
                for start in range(0, len(audio) - BLOCK_SIZE + 1, BLOCK_SIZE):
                    self._dictate(audio[start : start + BLOCK_SIZE])
            elif self._clock_ms >= self._deadline_ms and not self._segmenter.in_speech:
                self._enter_idle()
        elif self.state is VoiceState.CAPTURING:
            if event is SegmentEvent.END:
                self._process(self._segmenter.segment_audio, "listen")
            elif event is SegmentEvent.DISCARD:
                self._forget_dictation()  # это был шум: показанные «слова» убираем
                self._set_state(VoiceState.LISTENING)
            else:
                self._dictate(block)

    # --- Состояния ------------------------------------------------------------------

    def _idle_block(self, block: np.ndarray, event: SegmentEvent) -> None:
        """Ожидание имени: Vosk получает только звук, в котором есть речь.

        Решение принимается в конце фразы по окончательному результату Vosk
        с уверенностью не ниже wake_confidence.
        """
        if self._spotter is None or not self._settings.wake_word:
            return
        heard = self._spot(self._spotter, block, event)
        if heard is None:
            return
        log.debug("Vosk: %s", heard.describe())
        if event is SegmentEvent.END and heard.find([self._wake_phrase], self._settings.wake_confidence):
            log.info("Услышано имя ассистента: %s", heard.describe())
            if self.is_stop(heard):  # «Джарвис, стоп»: «стоп» есть в грамматике имени, Whisper не нужен
                self._stop_heard(heard.text)
                self._enter_idle()
                return
            self._on_wake(heard)

    def _spot(self, spotter: Spotter, block: np.ndarray, event: SegmentEvent) -> SpotResult | None:
        """Подаёт речь распознавателю; в конце фразы возвращает всё услышанное в ней, иначе None.

        Vosk получает только звук, в котором есть речь (с запасом до её начала).
        """
        if event is SegmentEvent.START:
            self._heard = SpotResult()
            audio = self._segmenter.current_audio()
            for start in range(0, len(audio) - BLOCK_SIZE + 1, BLOCK_SIZE):
                self._collect(spotter.accept(audio[start : start + BLOCK_SIZE]))
        elif self._segmenter.in_speech:
            self._collect(spotter.accept(block))
        elif event in (SegmentEvent.END, SegmentEvent.DISCARD):
            heard = self._heard + spotter.finish()
            self._heard = SpotResult()
            return heard
        return None

    def _collect(self, result: SpotResult | None) -> None:
        if result is not None:
            self._heard = self._heard + result

    def _stop_block(self, block: np.ndarray, over_speech: bool = False) -> bool:
        """Ищет «стоп» / «отмена» во время задачи или речи ассистента. True — услышано.

        over_speech — ассистент говорит: его голос из колонок попадает в микрофон, и фраза
        пользователя тонет в нём. Тогда «Джарвис, стоп» засчитывается и посреди фразы — уже
        по куску, который Vosk распознал до паузы.
        """
        if self._stop_spotter is None:
            return False
        event = self._segmenter.feed(block, self._vad(block))
        heard = self._spot(self._stop_spotter, block, event)
        if heard is None:
            if not (over_speech and self._heard.words and self._named_stop(self._heard)):
                return False
            heard = self._heard
        else:
            log.debug("Vosk (стоп): %s", heard.describe())
            if not self.is_stop(heard, over_speech):
                return False
        self._stop_heard(heard.text)
        return True

    def _stop_heard(self, text: str) -> None:
        """Услышан «стоп». Сначала прерывается задача, потом речь: так прерванная задача
        не успеет поставить в очередь новую фразу."""
        log.info("Услышано «%s» — останавливаю", text)
        if self._on_stop is not None:
            self._on_stop()
        self._speech.stop()
        self._report("stopped", text)

    def is_stop(self, heard: SpotResult, over_speech: bool = False) -> bool:
        """Во фразе только стоп-слова (и, может быть, имя), а стоп-слово распознано уверенно.

        Поэтому «стоп» посреди разговора или ответа ассистента ничего не прерывает.
        Короткое «стоп» детектор речи иногда считает шумом — такая фраза тоже проверяется.
        over_speech — ассистент говорит: «Джарвис, стоп» засчитывается и среди других слов.
        """
        words = {word for word, _ in heard.words}
        if (
            words
            and words <= self._stop_words
            and heard.find(STOP_PHRASES, self._settings.stop_confidence) is not None
        ):
            return True
        return over_speech and self._named_stop(heard)

    def _named_stop(self, heard: SpotResult) -> bool:
        """Имя и сразу стоп-слово, оба уверенно: «Джарвис, стоп» среди речи ассистента."""
        return heard.find(self._named_stops, self._settings.stop_confidence) is not None

    def _on_wake(self, heard: SpotResult) -> None:
        """Имя распознано в только что закончившейся фразе.

        Если сказали только «Джарвис», Whisper не нужен: сразу сигнал и ожидание команды.
        Одних слов Vosk для этого мало: с маленькой грамматикой он может вернуть
        «джарвис» и на «Джарвис, открой ютуб», не отметив остальное как [unk].
        Поэтому «только имя» — это ещё и короткая речь; всё длиннее распознаёт Whisper.
        """
        only_name = heard.known_words == self._wake_phrase.split() and not heard.has_unknown
        span_ms = self._segmenter.speech_span_ms
        if only_name and span_ms <= self._name_only_ms:
            self._listen(chime=True)
        else:
            log.debug("Речь с именем %d мс — распознаю целиком", span_ms)
            self._process(self._segmenter.segment_audio, "wake")

    def mentions_name(self, text: str) -> bool:
        """Есть ли имя ассистента в тексте (с учётом вариантов написания: «Jarvis», «Джервис»)."""
        from rapidfuzz import fuzz

        for word in basic_clean(text).split():
            word_skeleton = skeleton(word)
            if any(fuzz.ratio(word_skeleton, name) >= 80 for name in self._wake_skeletons):
                return True
        return False

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
            # Пока звучит сигнал, микрофон не выключаем: на короткий тон детектор речи
            # не реагирует, а начало команды, сказанной сразу после имени, иначе пропадает.
            self._chime_until_ms = self._clock_ms + len(audio) * 1000 // rate + _CHIME_MARGIN_MS
        self._deadline_ms = self._clock_ms + self._listen_window_ms
        self._set_state(VoiceState.LISTENING)

    def _set_state(self, state: VoiceState) -> None:
        """Меняет состояние и сообщает о нём (только о смене, без повторов)."""
        if state is self.state:
            return
        log.debug("Голос: %s", state)
        self.state = state
        self._report(state.value)

    def _report(self, kind: str, detail: str = "") -> None:
        if self._status is not None:
            self._status.voice_status(kind, detail)

    def _reset_audio_state(self) -> None:
        self._vad.reset()
        self._segmenter.reset()
        self._heard = SpotResult()
        for spotter in (self._spotter, self._stop_spotter):
            if spotter is not None:
                spotter.reset()
        # Фраза оборвалась (ассистент заговорил, микрофон занят подтверждением): слова — с экрана.
        self._forget_dictation()

    def _interrupt_phrase(self) -> None:
        """Звук оборвался посреди фразы (ассистент заговорил, микрофон занят подтверждением): фраза
        ждётся заново. Иначе цикл остался бы в CAPTURING без её конца и без срока ожидания — а с ним
        и оверлей «Слушаю…» на экране.
        """
        self._reset_audio_state()
        if self.state is VoiceState.CAPTURING:
            self._deadline_ms = self._clock_ms + self._listen_window_ms
            self._set_state(VoiceState.LISTENING)

    def _dictate(self, block: np.ndarray) -> None:
        """Слова текущей фразы по ходу речи — в оверлей (partial), когда они изменились."""
        if self._dictation is None:
            return
        try:
            text = self._dictation.accept(block)
        except Exception:
            log.debug("Слова по ходу речи не распознались", exc_info=True)
            return
        if text is not None and text != self._dictated:
            self._dictated = text
            self._report("partial", text)

    def _forget_dictation(self) -> None:
        if self._dictation is not None:
            self._dictation.reset()
        if self._dictated:
            self._dictated = ""
            self._report("partial", "")

    def handle_events(self) -> None:
        """Обрабатывает накопившиеся события горячих клавиш (вызывается в потоке цикла)."""
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                return
            if event in ("pause", "resume"):
                self._pause(event == "pause")
            elif event == "stop":
                self._stop_requested.clear()
                if self.state is not VoiceState.PROCESSING:
                    self._enter_idle()
            elif self._paused:
                continue  # на паузе горячие клавиши не активируют прослушивание
            elif event == "ptt" and self.state in (VoiceState.IDLE, VoiceState.LISTENING):
                self._speech.stop()
                self._source.flush()
                self._listen(chime=True)
            elif event == "hold_start" and self.state is not VoiceState.PROCESSING:
                self._source.flush()
                self._hold_buffer = []
                if self._dictation is not None:
                    self._dictation.reset()
                self._dictated = ""
                self._set_state(VoiceState.HOLDING)
            elif event == "hold_end" and self.state is VoiceState.HOLDING:
                audio = np.concatenate(self._hold_buffer) if self._hold_buffer else np.zeros(0, np.int16)
                self._process(audio, "hold")

    def _pause(self, paused: bool) -> None:
        if paused == self._paused:
            return
        self._paused = paused
        log.info("Прослушивание %s", "на паузе" if paused else "продолжено")
        self._enter_idle()
        if paused:
            self._report("paused")
        else:
            self._source.flush()
            self._report(VoiceState.IDLE.value)

    # --- Обработка фразы ------------------------------------------------------------

    def _process(self, audio: np.ndarray, origin: str) -> None:
        """Распознаёт фразу, выполняет её и решает, что слушать дальше.

        Задержка считается от конца речи: фраза закончена, когда VAD услышал паузу
        (voice.min_silence_ms), поэтому пауза входит в задержку. В режиме удержания
        клавиш конец фразы — отпускание клавиш, паузы нет.
        """
        self._set_state(VoiceState.PROCESSING)
        self._report("recognizing")
        self._hold_buffer = []
        self._dictated = ""  # показанные слова дальше сменит текст Whisper — стирать их не нужно
        if len(audio) < BLOCK_SIZE * 4:  # меньше ~130 мс — не фраза
            self._enter_idle()
            return
        pause_ms = 0 if origin == "hold" else self._segmenter.settings.min_silence_ms
        transcript = self._stt.transcribe(audio)
        log.info(
            "Конец речи → текст: %.0f мс (пауза %d мс + распознавание %.0f мс, звук %.1f с): «%s»",
            pause_ms + transcript.duration_ms, pause_ms, transcript.duration_ms, len(audio) / 16_000,
            transcript.text,
        )  # fmt: skip
        if self._stop_requested.is_set():
            # Пока фраза распознавалась, нажали «Стоп» или Ctrl+Alt+Q — её не выполняем.
            self._stop_requested.clear()
            log.info("Остановлено во время распознавания — «%s» не выполняю", transcript.text)
            self._enter_idle()
            return
        if not transcript.text:
            self._enter_idle()
            return
        if origin == "wake" and self._settings.verify_wake_word and not self.mentions_name(transcript.text):
            # Vosk «услышал» имя, а Whisper его не слышит — скорее всего, ложное срабатывание.
            log.info("Имени нет в распознанном «%s» — игнорирую", transcript.text)
            self._enter_idle()
            return
        normalized = self._normalizer.normalize(transcript.text)
        if not normalized:
            # Во фразе было только имя — ждём саму команду.
            self._listen(chime=True)
            return
        if is_stop_phrase(normalized):
            # «Стоп» вместо команды: остановить ассистента, а не искать команду.
            self._stop_heard(transcript.text)
            self._enter_idle()
            return
        self._report("recognized", transcript.text)
        started = time.perf_counter()
        result, stopped = self._run_task(transcript.text)
        action_ms = (time.perf_counter() - started) * 1000
        total_ms = pause_ms + transcript.duration_ms + action_ms
        log.info("Текст → результат: %.0f мс; конец речи → результат: %.0f мс", action_ms, total_ms)
        self._report(
            "latency",
            f"{total_ms:.0f} мс от конца речи: пауза {pause_ms} + распознавание "
            f"{transcript.duration_ms:.0f} + выполнение {action_ms:.0f}",
        )
        if self._status is not None and result is not None:
            self._status.voice_result(result)
        self._after_response(followup=not stopped and getattr(result, "handled", True))

    def _run_task(self, text: str) -> tuple[Any, bool]:
        """Выполняет фразу; если задан stop_spotter, тем временем слушает «стоп».

        Returns:
            Результат конвейера и то, прервал ли пользователь задачу голосом.
        """
        if self._stop_spotter is None:
            return self._handle_text(text, self._confirmer), False
        outcome: list[Any] = []
        failure: list[BaseException] = []

        def work() -> None:
            try:
                outcome.append(self._handle_text(text, self._confirmer))
            except BaseException as exc:  # ошибку разбирает голосовой цикл, как без отдельного потока
                failure.append(exc)

        worker = threading.Thread(target=work, name="voice-task", daemon=True)
        worker.start()
        stopped = self._watch_task(worker)
        if failure:
            raise failure[0]
        return (outcome[0] if outcome else None), stopped

    def _watch_task(self, worker: threading.Thread) -> bool:
        """Ждёт конца задачи и слушает «стоп» / «отмена». True — задачу прервали голосом.

        Пока голосовое подтверждение спрашивает «да/нет», микрофон слушает оно.
        """
        stopped = False
        self._reset_audio_state()
        while worker.is_alive() and not self._stopped.is_set():
            if stopped:
                worker.join(timeout=0.1)  # задача прервана — ждём, пока она это заметит
            elif self._mic_borrowed():
                self._reset_audio_state()  # после подтверждения — с чистого листа
                worker.join(timeout=0.05)
            elif (block := self._source.read(timeout=0.1)) is not None:
                self._clock_ms += BLOCK_MS
                # Нейросеть говорит ответ по предложениям, пока задача ещё идёт.
                stopped = self._stop_block(block, over_speech=self._speech.speaking)
            else:
                worker.join(timeout=0.01)
        return stopped

    def _mic_borrowed(self) -> bool:
        """Голосовое подтверждение сейчас само слушает микрофон."""
        return bool(getattr(self._confirmer, "listening", False))

    def _after_response(self, followup: bool) -> None:
        """После ответа: дождаться конца речи, затем слушать продолжение без имени.

        Если фраза не обработана («Не знаю такой команды»), продолжения не ждём:
        иначе разговор рядом или эхо ответа получали бы ответ за ответом.
        """
        self._source.flush()
        if followup and self._settings.followup_seconds > 0:
            self._listen(chime=False, seconds=self._settings.followup_seconds)
        else:
            self._enter_idle()
