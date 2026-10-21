"""Тесты голосового цикла на поддельных компонентах."""

from __future__ import annotations

import time

import numpy as np
import pytest

from assistant.audio.capture import SAMPLE_RATE
from assistant.audio.vad import SegmenterSettings, SileroVAD, SpeechSegmenter
from assistant.nlu.normalize import Normalizer
from assistant.voice import VoiceLoop, VoiceSettings, VoiceState
from tests.voice_fakes import (
    WAKE,
    FakeSource,
    FakeSpeech,
    FakeSpotter,
    FakeSTT,
    FakeVAD,
    RecordingStatus,
    blocks,
    silence,
    speech,
)

SCRIPT = {
    5001: "Джарвис, открой ютуб.",
    5002: "Сделай громче.",
    5003: "Который час?",
    WAKE: "Джарвис.",
    6666: "Субтитры сделал DimaTorzok",
}


class Harness:
    """Собирает цикл и запоминает, какие фразы дошли до конвейера."""

    def __init__(
        self,
        audio: np.ndarray,
        spotter: FakeSpotter | None = None,
        speech_out: FakeSpeech | None = None,
        settings: VoiceSettings | None = None,
        reply: str | None = "Готово",
    ) -> None:
        self.handled: list[str] = []
        self.source = FakeSource(audio)
        self.speech = speech_out or FakeSpeech()
        self.status = RecordingStatus()
        self.stt = FakeSTT(SCRIPT)
        self._reply = reply

        def handle(text: str, confirmer: object) -> str:
            self.handled.append(text)
            if self._reply:
                self.speech.say(self._reply)
            return text

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
            status=self.status,
            chime=(np.zeros(2205, dtype=np.float32), 22050),
        )

    def run(self) -> Harness:
        self.loop.run(stop_when_exhausted=True)
        return self


def test_wake_word_inside_phrase() -> None:
    audio = np.concatenate([silence(500), speech(WAKE, 300), speech(5001, 700), silence(1200)])
    h = Harness(audio).run()
    assert h.handled == ["Джарвис, открой ютуб."]
    assert ("recognized", "Джарвис, открой ютуб.") in h.status.events


def test_wake_word_detected_only_at_phrase_end() -> None:
    audio = np.concatenate([silence(500), speech(WAKE, 300), speech(5002, 700), silence(1200)])
    h = Harness(audio, spotter=FakeSpotter(mid_phrase=False)).run()
    assert h.handled == ["Сделай громче."]


def test_only_name_then_command() -> None:
    audio = np.concatenate([silence(300), speech(WAKE, 500), silence(1000), speech(5002, 800), silence(1200)])
    h = Harness(audio, spotter=FakeSpotter(mid_phrase=False)).run()
    assert h.handled == ["Сделай громче."]
    assert h.speech.sounds == 1  # сигнал «слушаю» после имени
    assert h.stt.calls == [5002]  # для одного имени Whisper не нужен


def test_only_name_recognized_by_whisper_then_command() -> None:
    # Имя услышано посреди фразы, Whisper вернул только «Джарвис.» — ждём команду.
    audio = np.concatenate([silence(300), speech(WAKE, 500), silence(1000), speech(5003, 800), silence(1200)])
    h = Harness(audio, spotter=FakeSpotter(mid_phrase=True)).run()
    assert h.handled == ["Который час?"]
    assert h.stt.calls == [WAKE, 5003]


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
    # Фейковый STT не фильтрует; фильтр «галлюцинаций» проверяется в test_stt.
    assert h.handled == ["Субтитры сделал DimaTorzok"]
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


def test_stop_speech_returns_to_idle() -> None:
    audio = silence(500)
    h = Harness(audio)
    h.loop.push_to_talk()
    h.loop.stop_speech()
    h.run()
    assert h.loop.state is VoiceState.IDLE
    assert h.speech.stops >= 1


def test_status_events() -> None:
    audio = np.concatenate([speech(WAKE, 300), speech(5001, 700), silence(1500)])
    h = Harness(audio).run()
    kinds = [kind for kind, _ in h.status.events]
    assert "capturing" in kinds and "recognizing" in kinds and "recognized" in kinds
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
