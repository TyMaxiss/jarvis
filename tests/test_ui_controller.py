"""UiController: что делают кнопки и поля интерфейса (без окон pywebview)."""

from __future__ import annotations

import shutil
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from assistant.actions.executor import ActionExecutor
from assistant.actions.registry import ActionRegistry
from assistant.commands.loader import CommandStore
from assistant.config import LlmConfig
from assistant.journal import ActionJournal
from assistant.llm.providers import CheckResult
from assistant.logging_setup import SecretMasker
from assistant.nlu.matcher import Matcher
from assistant.pipeline import Pipeline
from assistant.safety.confirm import StaticConfirmer
from assistant.safety.policy import SafetyPolicy
from assistant.ui import controller as controller_module
from assistant.ui.activity import Activity
from assistant.ui.confirm import UiConfirmer
from assistant.ui.controller import UiController
from assistant.ui.settings import SettingsStore
from tests.conftest import NO_WINDOW, ROOT, RecordingSpeaker, wait_until
from tests.ui_fakes import FakeAgent, FakeHost, FakeVoice, FakeVolume, RecordingBus

KEYS = ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENCODE_API_KEY", "CUSTOM_LLM_API_KEY")
SCENARIO = """
- id: work_mode
  phrases: ["рабочий режим"]
  reply: "Все системы готовы"
  steps:
    - volume_set: {percent: 30}
    - say: {text: "Работаем"}
"""


@pytest.fixture
def world(
    tmp_path: Path,
    registry: ActionRegistry,
    matcher: Matcher,
    speaker: RecordingSpeaker,
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    for name in KEYS:  # ключи, которые сохранит тест, после него удаляются
        monkeypatch.delenv(name, raising=False)
    commands = tmp_path / "commands"
    shutil.copytree(ROOT / "commands", commands)
    (commands / "scenarios.yaml").write_text(SCENARIO, encoding="utf-8")
    store = CommandStore(commands, "my.yaml", registry)
    store.reload()
    matcher.set_commands(store.commands)
    config_path = tmp_path / "config.yaml"
    shutil.copyfile(ROOT / "config.example.yaml", config_path)
    settings = SettingsStore(config_path, tmp_path / ".env")
    bus = RecordingBus()
    journal_file = tmp_path / "actions.jsonl"
    executor = ActionExecutor(
        registry,
        SafetyPolicy(),
        StaticConfirmer(True),
        ActionJournal(journal_file, SecretMasker()),
        dry_run=True,
    )
    pipeline = Pipeline(matcher, executor, speaker, lambda: NO_WINDOW)
    agents: list[FakeAgent] = []

    def make_agent(llm: LlmConfig) -> Any:
        agents.append(FakeAgent(llm))
        return agents[-1]

    controller = UiController(
        config=settings.load(),
        store=settings,
        pipeline=pipeline,
        executor=executor,
        commands=store,
        make_agent=make_agent,
        speech=None,
        volume=FakeVolume(),
        bus=bus,
        activity=Activity(bus),
        confirmer=UiConfirmer(bus, timeout=0.3),
        log_file=tmp_path / "assistant.log",
        journal_file=journal_file,
    )
    host = FakeHost()
    controller.host = host
    return SimpleNamespace(
        controller=controller, bus=bus, host=host, settings=settings, pipeline=pipeline, executor=executor,
        agents=agents, config_path=config_path, env_path=tmp_path / ".env",
    )  # fmt: skip


def test_state_matches_interface(world: SimpleNamespace) -> None:
    state = world.controller.state()
    # Ключи — как в типе AppState в ui/src/lib/types.ts.
    assert set(state) == {"name", "version", "voice", "llm", "dry_run", "volume", "recent", "scenarios", "ui"}
    assert set(state["voice"]) == {
        "enabled",
        "state",
        "paused",
        "wake_word",
        "push_to_talk_key",
        "stop_key",
        "loading",
        "problem",
    }
    assert set(state["llm"]) == {"enabled", "profile", "title", "model", "ready", "problem"}
    assert state["name"] == "Джарвис" and state["voice"]["state"] == "off"
    assert state["llm"]["title"] == "DeepSeek" and not state["llm"]["ready"]  # ключа нет
    assert state["volume"] == {"percent": 40, "muted": False}
    assert state["scenarios"] == [
        {"id": "work_mode", "title": "Рабочий режим", "phrase": "рабочий режим", "steps": 2}
    ]


def test_window_ready_starts_events(world: SimpleNamespace) -> None:
    world.controller.startup_issues = ["my.yaml: шаг 1: неизвестное действие"]
    world.controller.window_ready("main")
    world.controller.window_ready("main")  # перезагрузка страницы
    assert world.bus.attached("main")
    assert world.bus.of("commands") == [{"count": len(world.controller.commands.commands),
                                         "issues": ["my.yaml: шаг 1: неизвестное действие"]}]  # fmt: skip
    world.controller.window_ready("evil")
    assert not world.bus.attached("evil")


def test_live_settings_apply_without_restart(world: SimpleNamespace) -> None:
    result = world.controller.save_settings({"general": {"dry_run": False}, "ui": {"theme": "light"}})
    assert result["ok"] and not result["restart"] and result["settings"]["ui"]["theme"] == "light"
    assert world.executor.dry_run is False
    assert world.bus.of("state")[-1]["ui"]["theme"] == "light"
    assert not result["settings"]["restart_pending"]
    result = world.controller.save_settings({"voice": {"sounds": False}})
    assert result["ok"] and result["restart"] and world.controller.restart_pending
    assert world.controller.settings()["restart_pending"]  # баннер «Перезапустить» не пропадёт


def test_command_line_dry_run_is_not_turned_off(world: SimpleNamespace) -> None:
    world.controller._force_dry_run = True
    world.controller.save_settings({"general": {"dry_run": False}})
    assert world.executor.dry_run is True


def test_bad_settings_are_reported(world: SimpleNamespace) -> None:
    result = world.controller.save_settings({"ui": {"accent": "red"}})
    assert not result["ok"] and result["errors"] and result["settings"]["ui"]["accent"] == "#7c6cff"
    assert not world.controller.save_settings(["не словарь"])["ok"]
    assert not world.controller.save_settings({"llm": {"profiles": {}}})["ok"]


def test_llm_save_stores_key_and_switches_agent(world: SimpleNamespace) -> None:
    request = {
        "profile": "openai",
        "model": "gpt-5.4-mini",
        "key": " sk-test-openai-1234567890 ",
        "base_url": "",
    }
    result = world.controller.llm_save(request)
    assert result["ok"], result["errors"]
    assert world.settings.key("OPENAI_API_KEY") == "sk-test-openai-1234567890"
    assert "OPENAI_API_KEY='sk-test-openai-1234567890'" in world.env_path.read_text(encoding="utf-8")
    saved = world.settings.load().llm
    assert saved.profile == "openai" and saved.active.model == "gpt-5.4-mini"
    assert world.pipeline.agent is world.agents[0] and world.agents[0].settings.profile == "openai"
    openai = next(p for p in result["settings"]["llm"]["profiles"] if p["name"] == "openai")
    assert openai["key_set"] and openai["key_hint"] == "••••7890"
    assert "sk-test-openai" not in str(result)  # ключ обратно не уходит
    world.controller.llm_save({"profile": "openai", "model": "gpt-6-luna", "key": "", "base_url": ""})
    assert world.agents[0].reconfigured[-1].active.model == "gpt-6-luna"
    assert world.controller.state()["llm"]["ready"]


def test_llm_save_custom_service(world: SimpleNamespace) -> None:
    request = {"profile": "custom", "model": "local-model", "key": "", "base_url": "http://localhost:1234/v1"}
    assert world.controller.llm_save(request)["ok"]
    custom = world.settings.load().llm.profiles["custom"]
    assert custom.base_url == "http://localhost:1234/v1" and custom.api_key_env == ""
    request = {**request, "key": "lm-studio-key-123456"}
    assert world.controller.llm_save(request)["ok"]
    assert world.settings.load().llm.profiles["custom"].api_key_env == "CUSTOM_LLM_API_KEY"


@pytest.mark.parametrize(
    ("request_", "error"),
    [
        ({"profile": "nope", "model": "m"}, "Нет профиля"),
        ({"profile": "openai", "model": ""}, "Укажите модель"),
        (
            {"profile": "openai", "model": "m", "key": "ключ с пробелами"},
            "Ключ не подходит: в ключе символ «к»",
        ),
        ({"profile": "custom", "model": "m", "base_url": ""}, "base_url"),
    ],
)
def test_llm_save_rejects_bad_requests(world: SimpleNamespace, request_: dict[str, str], error: str) -> None:
    result = world.controller.llm_save(request_)
    assert not result["ok"] and error in result["errors"][0]
    assert world.settings.load().llm.profile == "deepseek"


def test_llm_test_uses_typed_key_without_saving(
    world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[LlmConfig, str | None]] = []

    def fake_check(settings: LlmConfig, api_key: str | None = None, timeout: float = 30) -> CheckResult:
        seen.append((settings, api_key))
        return CheckResult(True, "Готово", 812.6)

    monkeypatch.setattr(controller_module, "check_connection", fake_check)
    request = {"profile": "anthropic", "model": "claude-haiku-5-5", "key": "sk-ant-test-1234567890"}
    assert world.controller.llm_test(request) == {"ok": True, "text": "Готово", "ms": 813, "error": ""}
    settings, key = seen[0]
    assert settings.profile == "anthropic" and settings.active.model == "claude-haiku-5-5"
    assert key == "sk-ant-test-1234567890"
    assert not world.env_path.exists() and world.settings.load().llm.profile == "deepseek"


def test_llm_models_explains_errors(world: SimpleNamespace) -> None:
    result = world.controller.llm_models({"profile": "openai", "key": ""})
    assert not result["ok"] and "OPENAI_API_KEY" in result["error"]


def test_forget_key(world: SimpleNamespace) -> None:
    world.controller.llm_save(
        {"profile": "deepseek", "model": "deepseek-flash", "key": "sk-deepseek-1234567890"}
    )
    assert world.controller.llm_forget_key("deepseek")["ok"]
    assert world.settings.key("DEEPSEEK_API_KEY") == ""
    assert not world.controller.llm_forget_key("ollama")["ok"]  # ключа у неё и нет


def test_llm_can_be_turned_off_and_on(world: SimpleNamespace) -> None:
    world.controller.save_settings({"llm": {"enabled": True}})
    assert world.pipeline.agent is world.agents[0]
    world.controller.save_settings({"llm": {"enabled": False}})
    assert world.pipeline.agent is None and not world.controller.state()["llm"]["enabled"]


def test_text_command_runs_in_background(world: SimpleNamespace) -> None:
    world.controller.send_text("  сделай   громче ")
    assert wait_until(lambda: world.controller.activity.recent)
    item = world.controller.activity.recent[0]
    assert item["phrase"] == "сделай громче" and item["route"] == "local" and item["ok"]
    assert world.bus.of("recognized")[0] == {"text": "сделай громче", "source": "text"}
    world.controller.send_text("   ")
    assert len(world.bus.of("recognized")) == 1


def test_scenario_runs_by_id(world: SimpleNamespace) -> None:
    scenario = world.controller.scenarios()[0]
    world.controller.run_scenario(scenario["id"])
    assert wait_until(lambda: world.controller.activity.recent)
    assert world.controller.activity.recent[0]["phrase"] == scenario["phrase"]
    world.controller.run_scenario("нет-такой")


def test_window_controls(world: SimpleNamespace) -> None:
    controller, host = world.controller, world.host
    controller.window_action("main", "minimize")
    controller.window_action("main", "maximize")
    controller.window_action("main", "close")
    controller.window_action("chat", "hide")
    controller.window_resize(1000.4, 700, 2)
    controller.window_resize(1200, 800, 1)  # устаревший запрос
    controller.window_resize(300, 200, 3)  # меньше минимального
    assert host.calls == [
        ("minimize",), ("maximize",), ("hide", "main"), ("hide", "chat"),
        ("resize", 1000, 700), ("resize", 880, 600),
    ]  # fmt: skip


def test_quick_chat_toggles_and_focuses_input(world: SimpleNamespace) -> None:
    controller, host = world.controller, world.host
    controller.toggle_chat()
    assert host.visible("chat") and world.bus.events[-1] == ("chat_shown", {}, ("chat",))
    controller.toggle_chat()
    assert not host.visible("chat")
    controller.pause_hotkeys(True)  # в настройках записывают сочетание
    controller.toggle_chat()
    assert not host.visible("chat")


def test_voice_controls(world: SimpleNamespace) -> None:
    controller = world.controller
    voice = FakeVoice()
    controller.voice_loading = True
    assert controller.state()["voice"]["loading"]
    controller.attach_voice(voice)
    state = world.bus.of("state")[-1]["voice"]
    assert state["enabled"] and state["state"] == "idle" and not state["loading"] and state["wake_word"]
    controller.voice_listen()
    controller.voice_pause(True)
    assert voice.loop.listen_requests == 1 and voice.loop.paused
    pushes = len(world.bus.of("state"))
    controller.on_voice_state("paused")
    assert len(world.bus.of("state")) == pushes + 1 and world.bus.of("voice")[-1]["state"] == "paused"
    controller.toggle_pause()
    assert not voice.loop.paused


def test_level_meter_goes_to_settings_and_overlay(world: SimpleNamespace) -> None:
    controller = world.controller
    controller.attach_voice(FakeVoice())
    controller.level_meter(True)
    thread = threading.Thread(target=controller.pump_levels, args=(0.01,), daemon=True)
    thread.start()
    assert wait_until(lambda: any(kind == "level" for kind, _, _ in world.bus.events))
    controller.stop()
    thread.join(1)
    kind, payload, to = next(event for event in world.bus.events if event[0] == "level")
    assert payload == {"value": 0.42} and to == ("main",)


def test_volume_buttons(world: SimpleNamespace) -> None:
    assert world.controller.set_volume(55.4)["volume"]["percent"] == 55
    assert world.controller.toggle_mute()["volume"]["muted"] is True
    assert world.controller.set_volume("громко")["volume"]["percent"] == 55


def test_key_page_comes_from_provider_list(world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr(controller_module.webbrowser, "open", opened.append)
    world.controller.open_key_page("openai")
    world.controller.open_key_page("ollama")  # ключ не нужен — страницы нет
    assert opened == ["https://platform.openai.com/settings/organization/api-keys"]


def test_open_missing_file_shows_toast(world: SimpleNamespace) -> None:
    world.controller.open_path("log")
    world.controller.open_path("unknown")
    assert world.bus.of("toast") == [{"kind": "info", "text": "Файла assistant.log пока нет"}]


def test_preview_without_speech_is_explained(world: SimpleNamespace) -> None:
    world.controller.preview_voice(
        {"engine": "piper", "piper_voice": "models/piper/x.onnx", "rate": 1, "volume": 1}
    )
    assert wait_until(lambda: world.bus.of("toast"))
    assert "Озвучка" in world.bus.of("toast")[0]["text"]


def test_restart_closes_windows(world: SimpleNamespace) -> None:
    world.controller.restart()
    assert world.controller.restart_requested and world.host.calls == [("quit",)]


def test_reload_commands_reports_count(world: SimpleNamespace) -> None:
    rebinds: list[int] = []
    world.controller.on_commands = lambda: rebinds.append(1)
    result = world.controller.reload_commands()
    assert result["count"] == len(world.controller.commands.commands) and result["issues"] == []
    assert world.bus.of("commands")[-1] == result and rebinds == [1]


def test_new_key_for_same_profile_recreates_client(world: SimpleNamespace) -> None:
    request = {"profile": "deepseek", "model": "deepseek-flash", "key": "sk-deepseek-old-1234567890"}
    world.controller.llm_save(request)
    agent = world.pipeline.agent
    assert agent is world.agents[0] and agent.reconfigured == []  # агент создан с нужным профилем
    world.controller.llm_save({**request, "key": "sk-deepseek-new-1234567890"})
    # Профиль и модель те же, а ключ новый: клиент с прежним ключом не должен остаться.
    assert len(agent.reconfigured) == 1
    world.controller.llm_save({**request, "key": ""})
    assert len(agent.reconfigured) == 1  # без нового ключа пересоздавать нечего
