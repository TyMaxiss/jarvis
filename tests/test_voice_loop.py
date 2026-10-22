"""Тесты голосового цикла на поддельных компонентах."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest

from assistant.audio.capture import SAMPLE_RATE
from assistant.audio.vad import SegmenterSettings, SileroVAD, SpeechSegmenter
from assistant.audio.wakeword import SpotResult
from assistant.nlu.normalize import Normalizer
from assistant.voice import VoiceLoop, VoiceSettings, VoiceState, name_only_limit_ms
from tests.conftest import wait_until
from tests.voice_fakes import (
    STOP,
    WAKE,
    FakeSource,
    FakeSpeech,
    FakeSpotter,
    FakeSTT,
    FakeVAD,
    RecordingStatus,
    WordsSpotter,
    blocks,
    silence,
    speech,
)

SCRIPT = {
    5001: "Джарвис, открой ютуб.",
    5002: "Сделай громче.",
    5003: "Который час?",
    5004: "Сделай тише.",  # без имени: Vosk «ослышался», Whisper имени не слышит
    5005: "Джарвис.",
    5006: "Стоп.",
    5007: "Хватит!",
    5008: "Джарвис, отмена.",
    WAKE: "Джарвис.",
    6666: "Джарвис, субтитры сделал DimaTorzok",
}
ECHO = 4444  # голос ассистента из колонок в микрофоне


@dataclass(frozen=True)
class Unhandled:
    """Результат конвейера для фразы, которую он не понял («Не знаю такой команды»)."""

    text: str
    handled: bool = False


class Harness:
    """Собирает цикл и запоминает, какие фразы дошли до конвейера.

    understood=False — конвейер отвечает «не понял» на любую фразу.
    stop=True — цикл слушает «стоп» (блоки STOP); stopped — услышан ли он.
    task — что ещё делает конвейер с фразой (например, ждёт, пока его прервут).
    """

    def __init__(
        self,
        audio: np.ndarray,
        spotter: FakeSpotter | None = None,
        speech_out: FakeSpeech | None = None,
        settings: VoiceSettings | None = None,
        reply: str | None = "Готово",
        understood: bool = True,
        stop: bool = False,
        confirmer: object | None = None,
        stop_spotter: object | None = None,
        busy: Callable[[], bool] | None = None,
    ) -> None:
        self.handled: list[str] = []
        self.source = FakeSource(audio)
        self.speech = speech_out or FakeSpeech()
        self.status = RecordingStatus()
        self.stt = FakeSTT(SCRIPT)
        self.stopped = threading.Event()
        self.task: Callable[[str], object] | None = None
        self._reply = reply

        def handle(text: str, confirmer: object) -> object:
            self.handled.append(text)
            if self._reply:
                self.speech.say(self._reply)
            if self.task is not None:
                self.task(text)
            return text if understood else Unhandled(text)

        self.loop = VoiceLoop(
            source=self.source,
            vad=FakeVAD(),
            segmenter=SpeechSegmenter(SegmenterSettings(min_silence_ms=500)),
            spotter=spotter if spotter is not None else FakeSpotter(),
            stt=self.stt,
            handle_text=handle,
            speech=self.speech,
            normalizer=Normalizer(["джарвис"]),
            settings=settings or VoiceSettings(followup_seconds=3, listen_timeout=3),
            wake_names=["джарвис", "jarvis"],
            status=self.status,
            confirmer=confirmer,  # type: ignore[arg-type]
            chime=(np.zeros(2205, dtype=np.float32), 22050),
            stop_spotter=stop_spotter or (FakeSpotter(word="стоп", value=STOP) if stop else None),  # type: ignore[arg-type]
            on_stop=self.stopped.set,
            busy=busy,
        )

    def run(self) -> Harness:
        self.loop.run(stop_when_exhausted=True)
        return self


def test_wake_word_inside_phrase() -> None:
    audio = np.concatenate([silence(500), speech(WAKE, 300), speech(5001, 700), silence(1200)])
    h = Harness(audio).run()
    assert h.handled == ["Джарвис, открой ютуб."]
    assert ("recognized", "Джарвис, открой ютуб.") in h.status.events


def test_only_name_then_command() -> None:
    audio = np.concatenate([silence(300), speech(WAKE, 500), silence(1000), speech(5002, 800), silence(1200)])
    h = Harness(audio).run()
    assert h.handled == ["Сделай громче."]
    assert h.speech.sounds == 1  # сигнал «слушаю» после имени
    assert h.stt.calls == [5002]  # для одного имени Whisper не нужен


def test_command_said_during_chime_is_not_cut() -> None:
    # Пауза после имени чуть дольше min_silence_ms (500 мс в Harness): имя — отдельная фраза,
    # звучит сигнал, а команду начинают сразу, не дожидаясь его конца.
    audio = np.concatenate([silence(300), speech(WAKE, 500), silence(550), speech(5002, 800), silence(1200)])
    h = Harness(audio).run()
    assert h.handled == ["Сделай громче."]
    assert h.speech.sounds == 1
    assert h.stt.speech_ms[-1] >= 790  # команда целиком, без потерянного начала


def test_name_and_command_when_vosk_hears_only_name() -> None:
    # Vosk вернул только «джарвис» без [unk], но речь длинная — значит, была и команда.
    audio = np.concatenate([silence(300), speech(WAKE, 400), speech(5001, 900), silence(1200)])
    h = Harness(audio, spotter=FakeSpotter(unk=False)).run()
    assert h.handled == ["Джарвис, открой ютуб."]
    assert h.speech.sounds == 0  # без сигнала «слушаю»: команда уже прозвучала


def test_long_name_only_goes_through_whisper() -> None:
    # Протяжное «Джаааарвис» дольше предела: его распознаёт Whisper, и ассистент ждёт команду.
    audio = np.concatenate(
        [silence(300), speech(WAKE, 1100), silence(1000), speech(5002, 800), silence(1200)]
    )
    h = Harness(audio).run()
    assert h.handled == ["Сделай громче."]
    assert h.stt.calls == [WAKE, 5002]
    assert h.speech.sounds == 1


@pytest.mark.parametrize(("name", "limit"), [("джарвис", 800), ("jarvis", 800), ("эй компьютер", 1300)])
def test_name_only_limit(name: str, limit: int) -> None:
    assert name_only_limit_ms(name) == limit


def test_only_name_recognized_by_whisper_then_command() -> None:
    # Vosk услышал имя и ещё что-то, а Whisper — только «Джарвис.»: ждём команду.
    audio = np.concatenate(
        [silence(300), speech(WAKE, 300), speech(5005, 400), silence(1000), speech(5003, 800), silence(1200)]
    )
    h = Harness(audio).run()
    assert h.handled == ["Который час?"]
    assert h.stt.calls == [5005, 5003]


def test_low_confidence_name_is_ignored() -> None:
    audio = np.concatenate([silence(300), speech(WAKE, 300), speech(5001, 700), silence(1200)])
    h = Harness(audio, spotter=FakeSpotter(conf=0.5)).run()
    assert h.handled == [] and h.stt.calls == []


def test_false_wake_rejected_by_whisper() -> None:
    # Vosk уверен, что слышал имя, но в тексте от Whisper его нет — команда не выполняется.
    audio = np.concatenate([silence(300), speech(WAKE, 200), speech(5004, 900), silence(1200)])
    h = Harness(audio).run()
    assert h.stt.calls == [5004]
    assert h.handled == []


def test_wake_check_can_be_disabled() -> None:
    audio = np.concatenate([silence(300), speech(WAKE, 200), speech(5004, 900), silence(1200)])
    h = Harness(audio, settings=VoiceSettings(verify_wake_word=False)).run()
    assert h.handled == ["Сделай тише."]


@pytest.mark.parametrize(
    ("text", "expected"),
    [("Джарвис, открой", True), ("Jarvis, open", True), ("Джервис, привет", True), ("Сделай тише", False)],
)
def test_mentions_name(text: str, expected: bool) -> None:
    loop = Harness(silence(100)).loop
    assert loop.mentions_name(text) is expected


def test_speech_without_name_is_ignored() -> None:
    audio = np.concatenate([silence(300), speech(5002, 800), silence(1500)])
    h = Harness(audio).run()
    assert h.handled == []
    assert h.stt.calls == []  # без имени Whisper не запускается


def test_push_to_talk() -> None:
    audio = np.concatenate([silence(300), speech(5002, 800), silence(1500)])
    h = Harness(audio)
    h.loop.push_to_talk()
    h.run()
    assert h.handled == ["Сделай громче."]
    assert h.speech.stops == 1 and h.speech.sounds == 1


def test_push_to_talk_timeout() -> None:
    audio = np.concatenate([silence(4000), speech(5002, 800), silence(1500)])
    h = Harness(audio)
    h.loop.push_to_talk()
    h.run()
    assert h.handled == []  # за 3 секунды ожидания речи не было
    assert h.loop.state is VoiceState.IDLE


def test_follow_up_without_name() -> None:
    audio = np.concatenate(
        [speech(WAKE, 300), speech(5001, 700), silence(1500), speech(5002, 700), silence(1500)]
    )
    h = Harness(audio).run()
    assert h.handled == ["Джарвис, открой ютуб.", "Сделай громче."]


def test_no_follow_up_after_unknown_command() -> None:
    # После «Не знаю такой команды» окно продолжения не открывается: иначе разговор
    # рядом или эхо ответа получали бы ответ за ответом.
    audio = np.concatenate(
        [speech(WAKE, 300), speech(5001, 700), silence(1500), speech(5002, 700), silence(1500)]
    )
    h = Harness(audio, understood=False).run()
    assert h.handled == ["Джарвис, открой ютуб."]
    assert h.loop.state is VoiceState.IDLE


def test_end_of_listening_is_reported_once() -> None:
    # Когда окно продолжения закрылось, консоль должна узнать, что ассистент снова ждёт имя.
    audio = np.concatenate([speech(WAKE, 300), speech(5001, 700), silence(5000)])
    h = Harness(audio).run()
    kinds = [kind for kind, _ in h.status.events]
    assert kinds[-2:] == ["listening", "idle"]
    assert all(a != b for a, b in zip(kinds, kinds[1:], strict=False))  # без повторов подряд


def test_follow_up_window_expires() -> None:
    audio = np.concatenate(
        [speech(WAKE, 300), speech(5001, 700), silence(4500), speech(5002, 700), silence(1500)]
    )
    h = Harness(audio).run()
    assert h.handled == ["Джарвис, открой ютуб."]


def test_follow_up_counts_from_end_of_reply() -> None:
    # Ответ звучит ~2,5 с (80 блоков); окно в 3 с начинается после него.
    audio = np.concatenate(
        [speech(WAKE, 300), speech(5001, 700), silence(4500), speech(5002, 700), silence(1500)]
    )
    h = Harness(audio, speech_out=FakeSpeech(speak_blocks=80)).run()
    assert h.handled == ["Джарвис, открой ютуб.", "Сделай громче."]


def test_half_duplex_ignores_own_speech() -> None:
    # Пока ассистент говорит (100 блоков ≈ 3,2 с), фраза с именем не слушается.
    audio = np.concatenate(
        [
            speech(WAKE, 300),
            speech(5001, 700),
            silence(700),
            speech(WAKE, 300),
            speech(5003, 700),
            silence(5000),
        ]
    )
    h = Harness(audio, speech_out=FakeSpeech(speak_blocks=100)).run()
    assert h.handled == ["Джарвис, открой ютуб."]


def test_hallucination_and_silence_are_not_commands() -> None:
    audio = np.concatenate([speech(WAKE, 300), speech(6666, 700), silence(1500)])
    h = Harness(audio).run()
    # Фейковый STT не фильтрует; фильтр «галлюцинаций» проверяется в test_voice_components.
    assert h.handled == ["Джарвис, субтитры сделал DimaTorzok"]
    empty = Harness(np.concatenate([speech(WAKE, 300), speech(4999, 700), silence(1500)])).run()
    assert empty.handled == []  # STT ничего не распознал


def test_hold_mode() -> None:
    audio = np.concatenate([speech(5002, 900), silence(300)])
    h = Harness(audio, settings=VoiceSettings(push_to_talk_mode="hold"))
    h.loop.hold_start()
    h.loop.handle_events()
    for block in blocks(audio):
        h.loop.feed(block)
    h.loop.hold_end()
    h.loop.handle_events()
    assert h.handled == ["Сделай громче."]


def test_interrupt_returns_to_idle() -> None:
    audio = silence(500)
    h = Harness(audio)
    h.loop.push_to_talk()
    h.loop.interrupt()
    h.run()
    assert h.loop.state is VoiceState.IDLE
    assert h.speech.stops >= 1


def test_status_events() -> None:
    audio = np.concatenate([speech(WAKE, 300), speech(5001, 700), silence(1500)])
    h = Harness(audio).run()
    kinds = [kind for kind, _ in h.status.events]
    assert kinds.index("recognizing") < kinds.index("recognized") < kinds.index("latency")
    assert kinds.index("latency") < kinds.index("listening")
    latency = dict(h.status.events)["latency"]
    assert "мс от конца речи: пауза 500 + распознавание" in latency  # min_silence_ms в Harness — 500
    assert h.status.results == ["Джарвис, открой ютуб."]


def test_waiting_for_wake_word_uses_little_cpu() -> None:
    """ТЗ: в ожидании имени нагрузка < 5%. 10 с тишины должны обрабатываться за < 0,5 с CPU."""
    audio = silence(10_000)
    loop = VoiceLoop(
        source=FakeSource(audio),
        vad=SileroVAD(),
        segmenter=SpeechSegmenter(),
        spotter=FakeSpotter(),
        stt=FakeSTT({}),
        handle_text=lambda text, conf: None,
        speech=FakeSpeech(),
        normalizer=Normalizer(["джарвис"]),
        settings=VoiceSettings(),
    )
    started = time.process_time()
    loop.run(stop_when_exhausted=True)
    cpu = time.process_time() - started
    assert cpu < 0.05 * len(audio) / SAMPLE_RATE, f"CPU {cpu:.2f} с на 10 с звука"


@pytest.mark.parametrize("value", [0, 300])
def test_silence_and_quiet_noise_do_nothing(value: int) -> None:
    h = Harness(np.full(SAMPLE_RATE * 3, value, dtype=np.int16)).run()
    assert h.handled == [] and h.stt.calls == []


# --- «Стоп» / «отмена» ----------------------------------------------------------------


def test_stop_word_cancels_running_task() -> None:
    # Задача ждёт, пока её прервут. «Стоп» прерывает её; продолжение без имени после этого
    # не слушается, а следующее обращение по имени снова работает.
    audio = np.concatenate(
        [
            speech(WAKE, 300), speech(5001, 700), silence(800),
            speech(STOP, 400), silence(1000),
            speech(5002, 700), silence(1500),
            speech(WAKE, 300), speech(5001, 700), silence(1500),
        ]
    )  # fmt: skip
    h = Harness(audio, stop=True)
    interrupted: list[bool] = []

    def task(text: str) -> None:
        if not interrupted:
            interrupted.append(h.stopped.wait(timeout=5))

    h.task = task
    h.run()
    assert interrupted == [True]
    assert ("stopped", "стоп") in h.status.events and h.speech.stops >= 1
    assert h.handled == ["Джарвис, открой ютуб.", "Джарвис, открой ютуб."]
    assert 5002 not in h.stt.calls


def test_stop_inside_other_speech_does_not_cancel() -> None:
    # «Стоп» посреди фразы (разговор рядом, ответ ассистента) задачу не прерывает.
    audio = np.concatenate(
        [
            speech(WAKE, 300), speech(5001, 700), silence(800),
            speech(STOP, 300), speech(5003, 600), silence(1000),
        ]
    )  # fmt: skip
    h = Harness(audio, stop=True)
    h.task = lambda text: wait_until(lambda: h.source.exhausted, timeout=5)
    h.run()
    assert h.source.exhausted and not h.stopped.is_set()
    assert all(kind != "stopped" for kind, _ in h.status.events)


class BorrowingConfirmer:
    """Голосовое подтверждение: пока listening, оно само слушает микрофон."""

    def __init__(self, listening: bool = False) -> None:
        self.listening = listening

    def confirm(self, question: str) -> bool:
        return False


def test_confirmation_keeps_microphone() -> None:
    # Пока подтверждение ждёт «да/нет», цикл не читает микрофон; потом снова слушает «стоп».
    audio = np.concatenate(
        [speech(WAKE, 300), speech(5001, 700), silence(800), speech(STOP, 400), silence(1000)]
    )
    confirmer = BorrowingConfirmer()
    h = Harness(audio, stop=True, confirmer=confirmer)
    taken: list[int] = []

    def task(text: str) -> None:
        confirmer.listening = True  # задача спрашивает «да/нет» голосом
        time.sleep(0.05)  # цикл замечает это на следующем блоке
        before = h.source.reads
        time.sleep(0.2)
        taken.append(h.source.reads - before)
        confirmer.listening = False
        h.stopped.wait(timeout=5)

    h.task = task
    h.run()
    assert taken == [0]
    assert h.stopped.is_set()


@pytest.mark.parametrize(("stop", "handled"), [(True, 1), (False, 2)], ids=["слушает стоп", "без стоп-слов"])
def test_stop_while_assistant_speaks(stop: bool, handled: int) -> None:
    # Ответ звучит 60 блоков (~1,9 с); «Стоп» посреди него обрывает речь, и продолжение без
    # имени уже не слушается. Без распознавателя стоп-слов «Сделай громче» — продолжение.
    audio = np.concatenate(
        [
            speech(WAKE, 300), speech(5001, 700), silence(900),
            speech(STOP, 400), silence(2000),
            speech(5002, 700), silence(1500),
        ]
    )  # fmt: skip
    h = Harness(audio, speech_out=FakeSpeech(speak_blocks=60), stop=stop).run()
    assert len(h.handled) == handled
    assert (("stopped", "стоп") in h.status.events) is stop
    assert h.loop.state is (VoiceState.IDLE if stop else VoiceState.LISTENING)


@pytest.mark.parametrize(
    ("words", "expected"),
    [
        ([("стоп", 0.95)], True),
        ([("отмена", 0.9)], True),
        ([("джарвис", 0.9), ("стоп", 0.95)], True),
        ([("хватит", 0.85), ("хватит", 0.9)], True),
        ([("стоп", 0.6)], False),  # неуверенно
        ([("стоп", 0.95), ("[unk]", 1.0)], False),  # и что-то ещё
        ([("хорошо", 0.9), ("стоп", 0.95)], False),  # слово-приманка: обычная речь
        ([("джарвис", 0.95)], False),
        ([], False),
    ],
)
def test_is_stop(words: list[tuple[str, float]], expected: bool) -> None:
    loop = Harness(silence(100)).loop
    heard = SpotResult(" ".join(w for w, _ in words), tuple(words))
    assert loop.is_stop(heard) is expected


def test_stop_threshold_is_separate_from_name() -> None:
    settings = VoiceSettings(wake_confidence=0.95, stop_confidence=0.7)
    loop = Harness(silence(100), settings=settings).loop
    assert loop.is_stop(SpotResult("стоп", (("стоп", 0.75),)))
    assert not loop.is_stop(SpotResult("стоп", (("стоп", 0.65),)))


def test_pause_ignores_name_and_hotkey() -> None:
    audio = np.concatenate([silence(300), speech(WAKE, 300), speech(5001, 700), silence(1200)])
    h = Harness(audio)
    h.loop.set_paused(True)
    h.loop.push_to_talk()
    h.run()
    assert h.handled == [] and h.stt.calls == []
    assert h.loop.paused
    assert ("paused", "") in h.status.events
    assert h.source.exhausted  # звук читается, чтобы не копился в очереди


def test_resume_after_pause_listens_again() -> None:
    audio = np.concatenate([silence(300), speech(WAKE, 300), speech(5001, 700), silence(1200)])
    h = Harness(audio)
    h.loop.set_paused(True)
    h.loop.set_paused(False)
    h.run()
    assert h.handled == ["Джарвис, открой ютуб."]
    assert not h.loop.paused
    states = [state for state, _ in h.status.events]
    assert states.index("paused") < len(states) - 1 - states[::-1].index("idle")


# --- Стоп-фраза в ожидании команды, в покое, кнопкой, во время задачи из текста ----------


def test_stop_phrase_while_waiting_for_command() -> None:
    # «Джарвис» → сигнал → «Стоп»: прослушивание кончается, в конвейер ничего не уходит
    # (раньше «стоп» ставил музыку на паузу), продолжение без имени не слушается.
    audio = np.concatenate(
        [
            silence(300), speech(WAKE, 500), silence(1000),
            speech(5006, 500), silence(1500),
            speech(5002, 700), silence(1500),
        ]
    )  # fmt: skip
    h = Harness(audio).run()
    assert h.handled == [] and h.stopped.is_set()
    assert ("stopped", "Стоп.") in h.status.events
    assert all(kind != "recognized" for kind, _ in h.status.events)
    assert h.loop.state is VoiceState.IDLE and 5002 not in h.stt.calls


def test_stop_phrase_ends_follow_up() -> None:
    # После ответа ассистент ждёт продолжение; «Хватит!» его обрывает.
    audio = np.concatenate(
        [
            speech(WAKE, 300), speech(5001, 700), silence(1500),
            speech(5007, 500), silence(1500),
            speech(5002, 700), silence(1500),
        ]
    )  # fmt: skip
    h = Harness(audio).run()
    assert h.handled == ["Джарвис, открой ютуб."]
    assert ("stopped", "Хватит!") in h.status.events and h.speech.stops >= 1
    assert 5002 not in h.stt.calls


def test_name_and_stop_without_whisper() -> None:
    # «Джарвис, стоп»: «стоп» есть в грамматике имени — Vosk слышит всё сразу, Whisper не нужен.
    audio = np.concatenate([silence(300), speech(WAKE, 300), speech(STOP, 400), silence(1200)])
    spotter = WordsSpotter({WAKE: ("джарвис", 0.95), STOP: ("стоп", 0.9)})
    h = Harness(audio, spotter=spotter).run()  # type: ignore[arg-type]
    assert h.stopped.is_set() and ("stopped", "джарвис стоп") in h.status.events
    assert h.stt.calls == [] and h.handled == []


def test_name_and_stop_word_outside_name_grammar() -> None:
    # «Джарвис, отмена»: «отмены» нет в грамматике имени — фразу распознаёт Whisper.
    audio = np.concatenate([silence(300), speech(WAKE, 300), speech(5008, 500), silence(1200)])
    h = Harness(audio).run()
    assert h.stt.calls == [5008] and h.handled == []
    assert ("stopped", "Джарвис, отмена.") in h.status.events


def test_stop_command_is_not_a_stop_phrase() -> None:
    loop = Harness(silence(100)).loop
    assert not loop.is_stop(SpotResult("отмени [unk]", (("отмени", 0.95), ("[unk]", 1.0))))


class InterruptingSTT(FakeSTT):
    """Пока фраза распознаётся, пользователь нажимает «Стоп»."""

    def __init__(self, loop: VoiceLoop | None = None) -> None:
        super().__init__(SCRIPT)
        self.loop = loop

    def transcribe(self, audio: np.ndarray) -> object:  # type: ignore[override]
        if self.loop is not None:
            self.loop.interrupt()
        return super().transcribe(audio)


def test_stop_button_while_phrase_is_recognized() -> None:
    audio = np.concatenate(
        [
            speech(WAKE, 300), speech(5001, 700), silence(1500),
            speech(WAKE, 300), speech(5001, 700), silence(1500),
        ]
    )  # fmt: skip
    h = Harness(audio)
    stt = InterruptingSTT(h.loop)
    h.loop._stt = stt
    stt_calls: list[int] = []

    def once(audio: np.ndarray) -> object:
        # Стоп нажимают только во время первой фразы.
        stt.loop = None if stt_calls else h.loop
        stt_calls.append(1)
        return InterruptingSTT.transcribe(stt, audio)

    stt.transcribe = once  # type: ignore[method-assign]
    h.run()
    assert h.handled == ["Джарвис, открой ютуб."]  # первая фраза распознана, но не выполнена; вторая — да
    assert len(stt_calls) == 2
    assert h.speech.stops >= 1


def test_busy_pipeline_listens_only_for_stop() -> None:
    # Задача из текста ещё выполняется: голосовой цикл в покое слушает только «стоп»,
    # обращение по имени ждёт конца задачи.
    busy = [True]
    audio = np.concatenate(
        [
            silence(300), speech(WAKE, 300), speech(5001, 700), silence(1000),
            speech(STOP, 400), silence(1000),
        ]
    )  # fmt: skip
    h = Harness(audio, stop=True, busy=lambda: busy[0]).run()
    assert h.handled == [] and h.stt.calls == []
    assert h.stopped.is_set() and ("stopped", "стоп") in h.status.events


def test_name_works_again_after_task() -> None:
    # Пока задача из текста идёт (первые 2 с), обращение по имени не слушается; потом — снова да.
    audio = np.concatenate(
        [
            speech(WAKE, 300), speech(5001, 700), silence(1500),
            speech(WAKE, 300), speech(5001, 700), silence(1500),
        ]
    )  # fmt: skip
    h = Harness(audio, stop=True)
    h.loop._busy = lambda: h.loop._clock_ms < 2000
    h.run()
    assert h.handled == ["Джарвис, открой ютуб."] and h.stt.calls == [5001]
    assert not h.stopped.is_set()


def test_idle_loop_leaves_microphone_to_confirmation() -> None:
    # Команда по горячей клавише спрашивает «да/нет» голосом, а голосовой цикл в покое:
    # пока подтверждение слушает, цикл микрофон не читает.
    confirmer = BorrowingConfirmer(listening=True)
    audio = np.concatenate([speech(WAKE, 300), speech(5001, 700), silence(1500)])
    h = Harness(audio, confirmer=confirmer)
    thread = threading.Thread(target=h.run, daemon=True)
    thread.start()
    time.sleep(0.2)
    assert h.source.reads == 0
    confirmer.listening = False
    thread.join(timeout=10)
    assert h.handled == ["Джарвис, открой ютуб."]


# --- «Джарвис, стоп» поверх речи ассистента (колонки) --------------------------------------


def echo_spotter(chunks: bool) -> WordsSpotter:
    return WordsSpotter({ECHO: ("[unk]", 1.0), WAKE: ("джарвис", 0.9), STOP: ("стоп", 0.92)}, chunks=chunks)


@pytest.mark.parametrize("chunks", [False, True], ids=["в конце фразы", "по кускам"])
def test_name_and_stop_over_assistant_speech(chunks: bool) -> None:
    # Голос ассистента из колонок не прерывается: VAD видит одну длинную фразу, в которой
    # «Джарвис, стоп» среди чужих слов. Оно засчитывается — сразу по куску, если Vosk его отдал.
    audio = np.concatenate(
        [speech(ECHO, 600), speech(WAKE, 300), speech(STOP, 300), speech(ECHO, 900), silence(800)]
    )
    h = Harness(audio, speech_out=FakeSpeech(speak_blocks=10_000), stop_spotter=echo_spotter(chunks))
    h.speech.say("Длинный ответ")
    h.run()
    assert h.stopped.is_set() and h.speech.stops >= 1
    stopped = [detail for kind, detail in h.status.events if kind == "stopped"]
    assert stopped and "джарвис стоп" in stopped[0]


def test_bare_stop_inside_assistant_speech_is_ignored() -> None:
    # Одно «стоп» среди речи ассистента может быть его же словом — не засчитывается.
    audio = np.concatenate([speech(ECHO, 600), speech(STOP, 300), speech(ECHO, 900), silence(800)])
    h = Harness(audio, speech_out=FakeSpeech(speak_blocks=10_000), stop_spotter=echo_spotter(True))
    h.speech.say("Длинный ответ")
    h.run()
    assert not h.stopped.is_set()


@pytest.mark.parametrize(
    ("words", "over_speech", "expected"),
    [
        ([("[unk]", 1.0), ("джарвис", 0.9), ("стоп", 0.9), ("[unk]", 1.0)], True, True),
        ([("[unk]", 1.0), ("джарвис", 0.9), ("стоп", 0.9), ("[unk]", 1.0)], False, False),
        ([("джарвис", 0.9), ("[unk]", 1.0), ("стоп", 0.9)], True, False),  # не подряд
        ([("джарвис", 0.6), ("стоп", 0.9), ("[unk]", 1.0)], True, False),  # имя неуверенно
        ([("хорошо", 0.9), ("стоп", 0.95)], True, False),
    ],
)
def test_is_stop_over_speech(words: list[tuple[str, float]], over_speech: bool, expected: bool) -> None:
    loop = Harness(silence(100)).loop
    heard = SpotResult(" ".join(w for w, _ in words), tuple(words))
    assert loop.is_stop(heard, over_speech) is expected
