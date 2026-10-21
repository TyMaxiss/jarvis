"""Самопроверка (--self-check): загружаются ли нативные части программы.

Без моделей и микрофона проверяется то, что ломается в собранном exe, если в папке
_internal чего-то не хватает: DLL, данных пакетов, обёрток COM, сборок .NET.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def _vosk() -> str:
    import vosk

    vosk.SetLogLevel(-1)  # вызов в libvosk.dll
    return "libvosk загружена"


def _ctranslate2() -> str:
    import ctranslate2

    return f"ctranslate2 {ctranslate2.__version__}, видеокарт CUDA: {ctranslate2.get_cuda_device_count()}"


def _vad() -> str:
    from assistant.audio.capture import BLOCK_SIZE
    from assistant.audio.vad import SileroVAD

    probability = SileroVAD()(np.zeros(BLOCK_SIZE, dtype=np.int16))
    return f"onnxruntime, речь в тишине: {probability:.2f}"


def _piper() -> str:
    from piper.phonemize_espeak import EspeakPhonemizer

    phonemes = EspeakPhonemizer().phonemize("ru", "проверка")
    return f"espeak-ng: {''.join(phonemes[0])}"


def _sound() -> str:
    import sounddevice

    return f"PortAudio, устройств: {len(sounddevice.query_devices())}"


def _windows_ui() -> str:
    import webview.platforms.edgechromium  # noqa: F401 — загружает .NET и сборки WebView2

    from assistant.ui import win32

    return f"WebView2 Runtime: {win32.webview2_version() or 'не установлен'}"


def _ui_automation() -> str:
    import comtypes.client

    from assistant.actions.screen import CLSID_CUIAUTOMATION

    module = comtypes.client.GetModule("UIAutomationCore.dll")
    automation = comtypes.client.CreateObject(CLSID_CUIAUTOMATION, interface=module.IUIAutomation)
    return f"корень: «{automation.GetRootElement().CurrentName}»"


def _sapi() -> str:
    import comtypes.client

    voice = comtypes.client.CreateObject("SAPI.SpVoice")
    return f"голосов Windows: {voice.GetVoices().Count}"


def _llm() -> str:
    import anthropic
    import openai

    # Клиенты создаются без запросов: проверяется, что SDK и их зависимости собраны целиком.
    openai.OpenAI(api_key="self-check", base_url="http://127.0.0.1:9")
    anthropic.Anthropic(api_key="self-check", base_url="http://127.0.0.1:9")
    return f"openai {openai.__version__}, anthropic {anthropic.__version__}"


def _screenshot() -> str:
    from PIL import ImageGrab

    image = ImageGrab.grab(bbox=(0, 0, 8, 8), all_screens=True)
    return f"снимок {image.width}×{image.height}"


CHECKS: dict[str, Callable[[], str]] = {
    "Vosk (слово активации)": _vosk,
    "CTranslate2 (Whisper)": _ctranslate2,
    "Silero VAD": _vad,
    "Piper (espeak-ng)": _piper,
    "Звук": _sound,
    "Окна (WebView2, .NET)": _windows_ui,
    "UI Automation": _ui_automation,
    "SAPI": _sapi,
    "Нейросети (SDK)": _llm,
    "Снимок экрана": _screenshot,
}


def run_self_check(out: Any, checks: dict[str, Callable[[], str]] | None = None) -> int:
    """Выполняет проверки и пишет итог каждой; 0 — всё на месте, 1 — что-то не работает."""
    failed = 0
    for title, check in (checks or CHECKS).items():
        try:
            detail = check()
        except Exception as exc:
            failed += 1
            log.info("Самопроверка: %s", title, exc_info=True)  # подробности — в файл лога
            out.write(f"✗ {title}: {type(exc).__name__}: {exc}\n")
        else:
            out.write(f"✓ {title}: {detail}\n")
    out.write("Всё на месте.\n" if not failed else f"Не работает: {failed}.\n")
    return 1 if failed else 0
