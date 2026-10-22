"""Сквозная проверка голоса на настоящих моделях: Piper → VAD → Vosk → Whisper → матчер.

Речь синтезируется голосами Piper и подаётся в голосовой цикл вместо микрофона.
Нужны скачанные модели (uv run python -m assistant --download-models), поэтому тест
включается переменной JARVIS_E2E=1 (так сделано в CI на Windows).

Логика цикла и безопасность подтверждения проверяются на голосах dmitri и irina.
Точность распознавания — на эталонном голосе irina: синтетический dmitri произносит
отдельные слова так, что Whisper small и Vosk их путают («Открой ютуб» → «Пробою тут»),
это свойство синтезатора, а не конвейера. Его результаты видны в калибровке.

Калибровка печатает:
* для слова активации — сколько фраз с именем Vosk услышал и сколько фраз без имени
  принял за имя, с приманками и без (все скачанные голоса);
* длительность речи по VAD и текст Whisper с временем распознавания;
* что Vosk слышит в ответах «да/нет», с приманками и без;
* какие фразы прерывают ассистента как «стоп» (все голоса);
* слышен ли «стоп» поверх речи ассистента из колонок (эхо другим голосом).
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from assistant.audio.capture import ArraySource, resample, to_int16
from assistant.audio.vad import SegmentEvent, SileroVAD, SpeechSegmenter
from assistant.audio.wakeword import GrammarSpotter, SpotResult, load_vosk_model, phrase_spotter, wake_spotter
from assistant.config import AppConfig
from assistant.nlu.matcher import Matcher
from assistant.nlu.normalize import Normalizer
from assistant.safety.confirm import NO_PHRASES, YES_PHRASES, VoiceConfirmer
from assistant.stt.whisper_stt import WhisperSTT
from assistant.tts.piper_tts import PiperTTS
from assistant.voice import (
    STOP_PHRASES,
    STOP_SOUNDALIKES,
    Spotter,
    VoiceLoop,
    VoiceSettings,
    name_only_limit_ms,
)
from tests.conftest import NO_WINDOW, ROOT
from tests.voice_fakes import FakeSpeech, FakeSTT, RecordingStatus

pytestmark = pytest.mark.skipif(os.environ.get("JARVIS_E2E") != "1", reason="включается через JARVIS_E2E=1")
log = logging.getLogger(__name__)
CONFIG = AppConfig()
NAME = CONFIG.general.name
NORMALIZER = Normalizer(CONFIG.wake_words, CONFIG.matcher.filler_words)
PIPER_DIR = ROOT / "models" / "piper"
TEST_VOICES = ("ru_RU-dmitri-medium", "ru_RU-irina-medium")
REFERENCE_VOICE = "ru_RU-irina-medium"


def voice_paths(names: tuple[str, ...] | None = None) -> list[Path]:
    """Скачанные голоса Piper (все или только из names)."""
    paths = sorted(PIPER_DIR.glob("ru_RU-*.onnx")) if PIPER_DIR.is_dir() else []
    return [p for p in paths if names is None or p.stem in names]


@pytest.fixture(
    scope="module", params=voice_paths(TEST_VOICES) or [None], ids=lambda p: p.stem if p else "нет"
)
def piper(request: pytest.FixtureRequest) -> PiperTTS:
    if request.param is None:
        pytest.skip("голоса Piper не скачаны")
    return PiperTTS(request.param)


@pytest.fixture(scope="module", params=voice_paths() or [None], ids=lambda p: p.stem if p else "нет")
def any_piper(request: pytest.FixtureRequest) -> PiperTTS:
    """Любой из скачанных голосов — для быстрых проверок только на Vosk."""
    if request.param is None:
        pytest.skip("голоса Piper не скачаны")
    return PiperTTS(request.param)


@pytest.fixture(scope="module")
def reference_piper() -> PiperTTS:
    paths = voice_paths((REFERENCE_VOICE,)) or voice_paths()
    if not paths:
        pytest.skip("голоса Piper не скачаны")
    log.warning("Эталонный голос: %s", paths[0].stem)
    return PiperTTS(paths[0])


@pytest.fixture(scope="module")
def vosk_model() -> object:
    return load_vosk_model(ROOT / CONFIG.voice.vosk_model)


@pytest.fixture(scope="module")
def stt() -> WhisperSTT:
    prompt = "Джарвис, открой, сделай, громче, громкость, найди, ютуб."
    stt = WhisperSTT(CONFIG.stt, ROOT / CONFIG.stt.download_dir, lambda: prompt)
    stt.load()
    log.warning("Whisper: %s на %s, beam %d", stt.model_name, stt.device, CONFIG.stt.beam_size)
    return stt


def say(piper: PiperTTS, text: str) -> np.ndarray:
    """Синтезированная фраза в формате микрофона: 16 кГц, int16."""
    audio, rate = piper.synthesize(text)
    return to_int16(resample(audio, rate))


def gap(seconds: float) -> np.ndarray:
    return np.zeros(int(16_000 * seconds), dtype=np.int16)


def spot(spotter: Spotter, audio: np.ndarray) -> SpotResult:
    """Всё, что распознаватель услышал во фразе."""
    result = SpotResult()
    for start in range(0, len(audio) - 511, 512):
        part = spotter.accept(audio[start : start + 512])
        if part is not None:
            result = result + part
    return result + spotter.finish()


def speech_span_ms(audio: np.ndarray) -> int:
    """Длительность речи так, как её видит голосовой цикл (VAD и границы фразы)."""
    vad, segmenter = SileroVAD(), SpeechSegmenter()
    padded = np.concatenate([audio, gap(1.0)])
    for start in range(0, len(padded) - 511, 512):
        block = padded[start : start + 512]
        if segmenter.feed(block, vad(block)) is SegmentEvent.END:
            return segmenter.speech_span_ms
    return -1


def run_loop(audio: np.ndarray, vosk_model: object, stt: WhisperSTT) -> tuple[list[str], int]:
    """Прогоняет звук через голосовой цикл: что ушло в конвейер и сколько раз прозвучал сигнал."""
    handled: list[str] = []
    speech = FakeSpeech()
    loop = VoiceLoop(
        source=ArraySource(audio, silence_after=2.0),
        vad=SileroVAD(),
        segmenter=SpeechSegmenter(),
        spotter=wake_spotter(vosk_model, NAME),
        stt=stt,
        handle_text=lambda text, conf: handled.append(text),
        speech=speech,
        normalizer=NORMALIZER,
        wake_names=CONFIG.wake_words,
        settings=VoiceSettings(
            followup_seconds=3, listen_timeout=4, wake_confidence=CONFIG.voice.wake_confidence
        ),
        status=RecordingStatus(),
        chime=(np.zeros(1600, dtype=np.float32), 16_000),
    )
    loop.run(stop_when_exhausted=True)
    log.warning("Цикл передал в конвейер: %s, сигналов «слушаю»: %d", handled, speech.sounds)
    return handled, speech.sounds


def stop_spotter(vosk_model: object) -> GrammarSpotter:
    """Распознаватель стоп-слов, как в main.py: «стоп», «отмена»… и имя («Джарвис, стоп»)."""
    return phrase_spotter(vosk_model, [*STOP_PHRASES, NAME], exclude_decoys=STOP_SOUNDALIKES)


def stop_heard(audio: np.ndarray, spotter: GrammarSpotter) -> tuple[bool, list[SpotResult]]:
    """Прерывает ли фраза ассистента, пока он говорит (так же «стоп» слушается во время задачи).

    Returns:
        Остановил ли цикл речь и что Vosk услышал в каждой фразе.
    """
    speech = FakeSpeech(speak_blocks=100_000)
    speech.say("Длинный ответ ассистента")
    stopped = threading.Event()
    loop = VoiceLoop(
        source=ArraySource(np.concatenate([gap(0.3), audio, gap(1.0)]), silence_after=0.5),
        vad=SileroVAD(),
        segmenter=SpeechSegmenter(),
        spotter=None,
        stt=FakeSTT({}),
        handle_text=lambda text, conf: None,
        speech=speech,
        normalizer=NORMALIZER,
        wake_names=CONFIG.wake_words,
        settings=VoiceSettings(wake_word=False, stop_confidence=CONFIG.voice.stop_confidence),
        stop_spotter=spotter,
        on_stop=stopped.set,
    )
    heard: list[SpotResult] = []
    is_stop = loop.is_stop

    def remember(result: SpotResult) -> bool:
        heard.append(result)
        return is_stop(result)

    loop.is_stop = remember  # type: ignore[method-assign]
    loop.run(stop_when_exhausted=True)
    return stopped.is_set(), heard


def describe_heard(heard: list[SpotResult]) -> str:
    return " | ".join(result.describe() for result in heard) or "—"


def confirm_by_voice(piper: PiperTTS, vosk_model: object, answer: str) -> bool:
    source = ArraySource(np.concatenate([gap(0.5), say(piper, answer), gap(1.0)]), silence_after=0.5)
    confirmer = VoiceConfirmer(
        FakeSpeech(),
        source,
        lambda: phrase_spotter(vosk_model, YES_PHRASES + NO_PHRASES),
        timeout=5,
        min_confidence=CONFIG.voice.confirm_confidence,
    )
    return confirmer.confirm("Выключить компьютер?")


def test_wake_word_is_in_vosk_vocabulary(vosk_model: object) -> None:
    spotter = wake_spotter(vosk_model, NAME)
    log.warning("Слово «%s» в словаре Vosk: %s, слов-приманок: %d", NAME, spotter.usable, len(spotter.decoys))
    assert spotter.usable
    confirm = phrase_spotter(vosk_model, YES_PHRASES + NO_PHRASES)
    assert confirm.missing_words == []
    assert stop_spotter(vosk_model).missing_words == []


WITH_NAME = [
    "Джарвис.",
    "Джарвис, открой ютуб.",
    "Джарвис, сделай громче.",
    "Джарвис, который час?",
    "Джарвис, найди котиков на ютубе.",
    "Джарвис, пауза.",
    "Эй, Джарвис!",
    "Привет, Джарвис.",
]
WITHOUT_NAME = [
    "Сделай громче.",
    "Открой ютуб.",
    "Громкость на пятьдесят.",
    "Который час?",
    "Сегодня хорошая погода.",
    "Я пошёл в магазин за хлебом.",
    "Давай посмотрим кино вечером.",
    "Привет, как дела?",
    "Ты где сейчас?",
    "Позвони мне завтра утром.",
    "Что будем есть на ужин?",
    "Мне нужно закончить работу.",
    "Включи свет на кухне.",
    "Где мои ключи?",
    "Это очень интересная статья.",
    "Сколько стоит этот телефон?",
    "Пойдём гулять в парк.",
    "Он сказал, что придёт позже.",
    "Хорошо, договорились.",
    "Нет, я не согласен.",
    "Да, конечно.",
    "Спасибо большое.",
    "Подожди минутку.",
    "Какая сегодня дата?",
    # Похожие на имя по звучанию.
    "Жарко сегодня, правда?",
    "Джинсы лежат в шкафу.",
    "Чарльз Дарвин написал книгу.",
    "Марвин, иди сюда.",
]


def wake_variants(model: object) -> dict[str, Spotter]:
    return {"имя+[unk]": GrammarSpotter(model, [NAME]), "приманки": wake_spotter(model, NAME)}


def test_calibration_wake_word(vosk_model: object) -> None:
    """Полнота и ложные срабатывания слова активации для вариантов распознавателя."""
    voices = voice_paths()
    if not voices:
        pytest.skip("голоса Piper не скачаны")
    limit = name_only_limit_ms(NAME)
    totals: dict[str, list[int]] = {}
    for path in voices:
        piper = PiperTTS(path)
        audio = {phrase: say(piper, phrase) for phrase in WITH_NAME + WITHOUT_NAME}
        spans = {phrase: speech_span_ms(sound) for phrase, sound in audio.items()}
        for variant, spotter in wake_variants(vosk_model).items():
            found, false, fast, notes = 0, 0, 0, []
            for phrase, sound in audio.items():
                heard = spot(spotter, np.concatenate([gap(0.3), sound, gap(0.5)]))
                woke = heard.find([NAME], CONFIG.voice.wake_confidence) is not None
                only_name = woke and heard.known_words == [NAME] and not heard.has_unknown
                if phrase in WITH_NAME:
                    found += woke
                    if not woke:
                        notes.append(f"пропуск «{phrase}»: {heard.describe()}")
                elif woke:
                    false += 1
                    quick = only_name and spans[phrase] <= limit
                    fast += quick
                    notes.append(f"ложно «{phrase}»{' (сразу сигнал)' if quick else ''}: {heard.describe()}")
            total = totals.setdefault(variant, [0, 0, 0, 0, 0])
            for index, value in enumerate((found, len(WITH_NAME), false, len(WITHOUT_NAME), fast)):
                total[index] += value
            log.warning(
                "КАЛИБРОВКА имени %s | %s: услышано %d/%d, ложно %d/%d, из них сразу сигнал %d%s",
                path.stem, variant, found, len(WITH_NAME), false, len(WITHOUT_NAME), fast,
                "".join(f"\n    {note}" for note in notes),
            )  # fmt: skip
    for variant, (found, positives, false, negatives, fast) in totals.items():
        log.warning(
            "КАЛИБРОВКА имени ИТОГО %s: услышано %d/%d, ложно %d/%d, из них сразу сигнал %d",
            variant, found, positives, false, negatives, fast,
        )  # fmt: skip


COMMAND_PHRASES = [
    "Джарвис.",
    "Джарвис, открой ютуб.",
    "Джарвис, сделай громче.",
    "Сделай громче.",
    "Открой ютуб.",
    "Громкость на пятьдесят.",
    "Который час?",
]
CALIBRATION_ANSWERS = [
    "Да.",
    "Нет.",
    "Подтверждаю.",
    "Конечно.",
    "Не надо.",
    "Отмена.",
    "Давай.",
    "Да ладно.",
    "Сделай громче.",
    "Сегодня хорошая погода.",
]


def test_calibration_whisper(piper: PiperTTS, stt: WhisperSTT) -> None:
    """Длительность речи по VAD, текст Whisper и время распознавания."""
    for phrase in COMMAND_PHRASES:
        speech = say(piper, phrase)
        started = time.perf_counter()
        text = stt.transcribe(np.concatenate([gap(0.3), speech, gap(0.5)])).text
        log.warning(
            "КАЛИБРОВКА Whisper «%s» | речь %d мс | «%s» за %.0f мс",
            phrase, speech_span_ms(speech), text, (time.perf_counter() - started) * 1000,
        )  # fmt: skip


def test_calibration_answers(vosk_model: object) -> None:
    """Что Vosk слышит в ответах на вопрос подтверждения, без приманок и с ними (все голоса)."""
    variants = {
        "без приманок": GrammarSpotter(vosk_model, YES_PHRASES + NO_PHRASES),
        "приманки": phrase_spotter(vosk_model, YES_PHRASES + NO_PHRASES),
    }
    for path in voice_paths():
        piper = PiperTTS(path)
        audio = {
            answer: np.concatenate([gap(0.3), say(piper, answer), gap(0.5)]) for answer in CALIBRATION_ANSWERS
        }
        for variant, spotter in variants.items():
            heard = " | ".join(f"«{a}» → {spot(spotter, sound).describe()}" for a, sound in audio.items())
            log.warning("КАЛИБРОВКА ответы %s, %s | %s", path.stem, variant, heard)


@pytest.mark.parametrize(
    ("phrase", "command_id"),
    [
        ("Сделай громче.", "volume_up"),
        ("Громкость на пятьдесят.", "volume_set"),
        ("Открой ютуб.", "open_youtube"),
        ("Который час?", "time"),
    ],
)
def test_whisper_recognizes_commands(
    reference_piper: PiperTTS, stt: WhisperSTT, matcher: Matcher, phrase: str, command_id: str
) -> None:
    transcript = stt.transcribe(say(reference_piper, phrase))
    log.warning("Whisper: «%s» → «%s» за %.0f мс", phrase, transcript.text, transcript.duration_ms)
    result = matcher.match(transcript.text, NO_WINDOW)
    assert result.matched and result.matches[0].command.id == command_id, transcript.text


def test_wake_word_and_command_in_one_phrase(piper: PiperTTS, vosk_model: object, stt: WhisperSTT) -> None:
    # Без запятой: после неё Piper иногда делает паузу дольше 700 мс, и фраза делится на две
    # (тогда ассистент правильно подаёт сигнал после имени, но проверяется другой сценарий).
    audio = np.concatenate([gap(1.0), say(piper, "Джарвис открой ютуб."), gap(1.5)])
    handled, chimes = run_loop(audio, vosk_model, stt)
    assert len(handled) == 1 and chimes == 0, handled


@pytest.mark.parametrize("pause", [1.2, 0.8], ids=["пауза 1.2 с", "команда во время сигнала"])
def test_name_then_pause_then_command(
    piper: PiperTTS, vosk_model: object, stt: WhisperSTT, pause: float
) -> None:
    audio = np.concatenate(
        [gap(1.0), say(piper, "Джарвис."), gap(pause), say(piper, "Сделай громче."), gap(1.5)]
    )
    handled, chimes = run_loop(audio, vosk_model, stt)
    assert len(handled) == 1 and chimes == 1, handled


@pytest.mark.parametrize(
    "phrase",
    ["Сделай громче.", "Который час?", "Сегодня хорошая погода.", "Давай посмотрим кино вечером."],
)
def test_phrase_without_name_is_ignored(
    piper: PiperTTS, vosk_model: object, stt: WhisperSTT, phrase: str
) -> None:
    audio = np.concatenate([gap(1.0), say(piper, phrase), gap(1.5)])
    assert run_loop(audio, vosk_model, stt) == ([], 0)  # ни команды, ни сигнала «слушаю»


@pytest.mark.parametrize(
    ("answer", "expected"), [("Да.", True), ("Подтверждаю.", True), ("Нет.", False), ("Не надо.", False)]
)
def test_voice_confirmation(
    reference_piper: PiperTTS, vosk_model: object, answer: str, expected: bool
) -> None:
    assert confirm_by_voice(reference_piper, vosk_model, answer) is expected


@pytest.mark.parametrize("phrase", ["Сделай громче.", "Сегодня хорошая погода.", "Да ладно."])
def test_unrelated_speech_is_not_consent(any_piper: PiperTTS, vosk_model: object, phrase: str) -> None:
    """Посторонняя фраза во время подтверждения не должна выполнить рискованное действие."""
    assert confirm_by_voice(any_piper, vosk_model, phrase) is False


@pytest.mark.skipif(sys.platform != "win32", reason="SAPI есть только в Windows")
def test_sapi_voice_synthesizes() -> None:
    """Голос Windows синтезирует речь (в CI установлены только английские голоса)."""
    import threading

    from assistant.tts.sapi_tts import SapiTTS

    result: dict[str, Any] = {}

    def run() -> None:
        try:
            sapi = SapiTTS()
            result["voices"] = sapi.available_voices()
            result["text"] = "Проверка голоса" if sapi.has_russian_voice else "Voice check"
            result["audio"] = sapi.synthesize(result["text"])
        except Exception as exc:  # ошибку из потока показываем в тексте теста
            result["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=30)
    log.warning("Голоса SAPI: %s", result.get("voices"))
    assert not worker.is_alive(), "SAPI завис при синтезе"
    assert "error" not in result, result.get("error")
    audio, rate = result["audio"]
    log.warning("SAPI: «%s» → %.1f с звука, %d Гц", result["text"], len(audio) / rate, rate)
    assert len(audio) > rate * 0.3
    assert float(np.abs(audio).max()) > 0.01  # не тишина


# --- «Стоп» / «отмена» ----------------------------------------------------------------

STOP_SAID = ["Стоп.", "Отмена.", "Хватит.", "Джарвис, стоп."]
NOT_STOP = ["Сегодня хорошая погода.", "Сделай громче.", "Открываю ютуб.", "Не знаю такой команды."]
STOP_TRIES = ["Стоп.", "Стоп!", "Отмена.", "Хватит.", "Прекрати.", "Отмени.",
              "Джарвис, стоп.", "Стоп, хватит."]  # fmt: skip
ASSISTANT_REPLIES = ["Отменено.", "Готово.", "Открываю ютуб.", "Не знаю такой команды.",
                     "Громкость пятьдесят процентов.", "Нейросеть недоступна.",
                     "Выключаю компьютер через пять секунд.", "Хорошо.", "Сейчас посмотрю."]  # fmt: skip
# Piper синтезирует с шумом: одна и та же фраза каждый раз звучит немного иначе.
STOP_TAKES = 3


def stop_takes(piper: PiperTTS, phrase: str, spotter: GrammarSpotter) -> tuple[int, list[str]]:
    """Сколько из STOP_TAKES синтезов фразы прервали ассистента и что услышал Vosk."""
    hits, heard_all = 0, []
    for _ in range(STOP_TAKES):
        stopped, heard = stop_heard(say(piper, phrase), spotter)
        hits += stopped
        heard_all.append(describe_heard(heard))
    return hits, heard_all


def test_calibration_stop_words(vosk_model: object) -> None:
    """Полнота «стоп» и ложные срабатывания на обычных фразах и ответах ассистента (все голоса).

    «Почти» — фраза, в которой Vosk услышал одни стоп-слова, но с уверенностью ниже
    voice.stop_confidence: по ним видно, насколько безопасно снижать порог.
    """
    voices = voice_paths()
    if not voices:
        pytest.skip("голоса Piper не скачаны")
    spotter = stop_spotter(vosk_model)
    stop_words = set(" ".join((*STOP_PHRASES, NAME)).split())
    others = [*WITHOUT_NAME, *ASSISTANT_REPLIES]
    total_found = total_false = 0
    for path in voices:
        piper = PiperTTS(path)
        lines, found, false = [], 0, 0
        for phrase in STOP_TRIES:
            hits, heard = stop_takes(piper, phrase, spotter)
            found += hits
            lines.append(f"«{phrase}» {hits}/{STOP_TAKES}: {' | '.join(heard)}")
        for phrase in others:
            stopped, heard = stop_heard(say(piper, phrase), spotter)
            false += stopped
            only_stop = any(h.words and all(w in stop_words for w, _ in h.words) for h in heard)
            if stopped or only_stop:
                lines.append(f"{'ложно' if stopped else 'почти'} «{phrase}»: {describe_heard(heard)}")
        total_found += found
        total_false += false
        log.warning(
            "КАЛИБРОВКА стоп %s: услышано %d/%d, ложно %d/%d%s",
            path.stem, found, len(STOP_TRIES) * STOP_TAKES, false, len(others),
            "".join(f"\n    {line}" for line in lines),
        )  # fmt: skip
    log.warning(
        "КАЛИБРОВКА стоп ИТОГО (порог %.2f): услышано %d/%d, ложно %d/%d",
        CONFIG.voice.stop_confidence, total_found, len(STOP_TRIES) * STOP_TAKES * len(voices),
        total_false, len(others) * len(voices),
    )  # fmt: skip


@pytest.mark.parametrize("phrase", STOP_SAID)
def test_stop_word_interrupts(reference_piper: PiperTTS, vosk_model: object, phrase: str) -> None:
    # Синтез случаен, поэтому проверяется большинство из STOP_TAKES попыток.
    hits, heard = stop_takes(reference_piper, phrase, stop_spotter(vosk_model))
    assert hits * 2 > STOP_TAKES, heard


@pytest.mark.parametrize("phrase", NOT_STOP)
def test_ordinary_speech_does_not_interrupt(piper: PiperTTS, vosk_model: object, phrase: str) -> None:
    stopped, heard = stop_heard(say(piper, phrase), stop_spotter(vosk_model))
    assert not stopped, describe_heard(heard)


# --- «Стоп» поверх речи ассистента (колонки) -----------------------------------------------

ECHO_REPLY = "Сейчас расскажу подробнее. Это займёт немного времени, потому что вопрос интересный."
OVER_SPEECH = ["Джарвис, стоп.", "Стоп."]
ECHO_GAINS = (0.3, 0.6)  # громкость голоса ассистента в микрофоне относительно пользователя


def over_echo(echo: np.ndarray, phrase: np.ndarray, gain: float, at: float = 1.0) -> np.ndarray:
    """Фраза пользователя поверх речи ассистента: так их слышит микрофон рядом с колонками."""
    start = int(16_000 * at)
    length = max(len(echo), start + len(phrase))
    mixed = np.zeros(length, dtype=np.float32)
    mixed[: len(echo)] += echo.astype(np.float32) * gain
    mixed[start : start + len(phrase)] += phrase.astype(np.float32)
    return np.clip(mixed, -32768, 32767).astype(np.int16)


def test_calibration_stop_over_speech(reference_piper: PiperTTS, vosk_model: object) -> None:
    """Слышен ли «стоп», когда его перекрывает голос ассистента (эхо — другие голоса Piper).

    Одно «стоп» среди чужой речи не засчитывается (это может быть слово ассистента),
    «Джарвис, стоп» — засчитывается. Печатает, что сработало и что услышал Vosk.
    """
    echoes = [path for path in voice_paths() if path.stem != REFERENCE_VOICE]
    if not echoes:
        pytest.skip("нет второго голоса Piper для эха")
    spotter = stop_spotter(vosk_model)
    lines, found = [], {phrase: 0 for phrase in OVER_SPEECH}
    tries = 0
    for path in echoes:
        echo = say(PiperTTS(path), ECHO_REPLY)
        for gain in ECHO_GAINS:
            tries += 1
            for phrase in OVER_SPEECH:
                stopped, heard = stop_heard(over_echo(echo, say(reference_piper, phrase), gain), spotter)
                found[phrase] += stopped
                mark = "да" if stopped else "нет"
                lines.append(f"{path.stem} эхо {gain}: «{phrase}» {mark}: {describe_heard(heard)}")
            # Одна речь ассистента (без пользователя) не должна прерывать саму себя.
            stopped, heard = stop_heard((echo.astype(np.float32) * gain).astype(np.int16), spotter)
            if stopped:
                lines.append(f"ЛОЖНО {path.stem} эхо {gain}: {describe_heard(heard)}")
    log.warning(
        "КАЛИБРОВКА стоп поверх речи: %s%s",
        ", ".join(f"«{phrase}» {count}/{tries}" for phrase, count in found.items()),
        "".join(f"\n    {line}" for line in lines),
    )


def test_assistant_speech_does_not_stop_itself(any_piper: PiperTTS, vosk_model: object) -> None:
    # Ответ ассистента в микрофоне (колонки) — его голос, без пользователя — его не прерывает.
    stopped, heard = stop_heard(say(any_piper, ECHO_REPLY), stop_spotter(vosk_model))
    assert not stopped, describe_heard(heard)
