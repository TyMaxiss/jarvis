"""Живая проверка провайдера нейросети: настоящий запрос с вызовом инструмента.

Включается переменной JARVIS_LIVE_LLM=1. Профиль — JARVIS_LIVE_PROFILE, а без неё —
llm.profile из config.yaml. Ключ берётся из .env или из переменной окружения.
Каждый запуск — несколько коротких запросов: платный провайдер возьмёт за них немного денег.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from assistant.actions.executor import ActionExecutor
from assistant.actions.registry import ActionRegistry
from assistant.config import AppConfig, LlmConfig, load_config
from assistant.journal import ActionJournal
from assistant.llm.agent import Agent, AgentStatus
from assistant.logging_setup import SecretMasker
from assistant.safety.confirm import StaticConfirmer
from assistant.safety.policy import SafetyPolicy
from tests.conftest import ROOT, RecordingSpeaker

pytestmark = pytest.mark.skipif(
    os.environ.get("JARVIS_LIVE_LLM") != "1", reason="включается через JARVIS_LIVE_LLM=1"
)
log = logging.getLogger(__name__)


def live_settings() -> LlmConfig:
    """Раздел llm из config.yaml (если он есть) с профилем из JARVIS_LIVE_PROFILE."""
    load_dotenv(ROOT / ".env", override=False)
    path = ROOT / "config.yaml"
    llm = load_config(path).llm if path.exists() else AppConfig().llm
    profile = os.environ.get("JARVIS_LIVE_PROFILE") or llm.profile
    return LlmConfig.model_validate({**llm.model_dump(), "profile": profile})


def test_provider_calls_tool_and_answers(tmp_path: Path) -> None:
    settings = live_settings()
    registry = ActionRegistry()
    calls: list[str] = []

    @registry.action("get_time", "Узнать текущее время и дату")
    def get_time() -> str:
        calls.append("get_time")
        return "Сейчас 18:30, среда, 21 октября 2026 года"

    journal = ActionJournal(tmp_path / "journal.jsonl", SecretMasker.from_environ())
    executor = ActionExecutor(registry, SafetyPolicy(), StaticConfirmer(False), journal)
    speaker = RecordingSpeaker()
    result = Agent(settings, executor, speaker).run("Который сейчас час? Узнай с помощью инструмента.")
    log.warning(
        "Нейросеть %s: %s, запросов %d, нейросеть %.0f мс, всего %.0f мс; инструменты %s; ответ «%s»",
        settings.profile, result.status, result.requests, result.llm_ms, result.total_ms, calls, result.text,
    )  # fmt: skip
    assert result.status is AgentStatus.DONE, result.error
    assert "get_time" in calls
    assert result.text and speaker.said


def test_model_sees_screenshot(tmp_path: Path) -> None:
    """Снимок экрана доходит до модели в формате её API (если в профиле vision: true)."""
    from PIL import Image

    settings = live_settings()
    if not settings.active.vision:
        pytest.skip(f"профиль {settings.profile} не принимает картинки (vision: false)")
    registry = ActionRegistry()

    @registry.action("describe_screen", "Посмотреть, что на экране")
    def describe_screen() -> str:
        return "Окно «Проверка». На снимке экрана — один цветной квадрат."

    buffer = io.BytesIO()
    Image.new("RGB", (320, 240), (220, 20, 20)).save(buffer, "JPEG", quality=90)
    shots = [buffer.getvalue()]
    journal = ActionJournal(tmp_path / "journal.jsonl", SecretMasker.from_environ())
    executor = ActionExecutor(registry, SafetyPolicy(), StaticConfirmer(False), journal)
    agent = Agent(settings, executor, RecordingSpeaker(), attachments=lambda: shots.pop() if shots else None)
    result = agent.run("Посмотри на экран через describe_screen и ответь одним словом: какого цвета квадрат?")
    log.warning("Нейросеть %s, снимок: %s, ответ «%s»", settings.profile, result.status, result.text)
    assert result.status is AgentStatus.DONE, result.error
    assert not shots, "describe_screen не вызван"
    assert "красн" in result.text.lower()
