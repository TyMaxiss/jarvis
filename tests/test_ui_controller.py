"""UiController: что делают кнопки и поля интерфейса (без окон pywebview)."""

from __future__ import annotations

import json
import shutil
import threading
from datetime import timedelta
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
from assistant.timers import Alarms, local_now
from assistant.ui import controller as controller_module
from assistant.ui.activity import Activity, VoiceStatus
from assistant.ui.confirm import UiConfirmer
from assistant.ui.controller import UiController
from assistant.ui.settings import SettingsStore
from assistant.updates import Release, UpdateError
from tests.conftest import NO_WINDOW, ROOT, RecordingSpeaker, wait_until
from tests.ui_fakes import FakeAgent, FakeHost, FakeVoice, FakeVolume, RecordingBus

KEYS = (
    "DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENCODE_API_KEY", "CUSTOM_LLM_API_KEY",
    "YANDEX_API_KEY",
)  # fmt: skip
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
    assert set(state) == {
        "name",
        "version",
        "voice",
        "busy",
        "llm",
        "dry_run",
        "volume",
        "recent",
        "scenarios",
        "timers",
        "update",
        "ui",
    }
    assert set(state["update"]) == {
        "supported",
        "version",
        "release",
        "checked",
        "checking",
        "installing",
        "step",
        "progress",
        "error",
    }
    assert set(state["voice"]) == {
        "enabled",
        "state",
        "paused",
        "wake_word",
        "push_to_talk_key",
        "stop_key",
        "loading",
        "problem",
        "failed",
        "step",
        "progress",
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


def test_alarms_reach_window_and_tray(world: SimpleNamespace, tmp_path: Path) -> None:
    path = tmp_path / "timers.json"
    Alarms(path).add_reminder(
        "позвонить маме", local_now() - timedelta(minutes=5)
    )  # пока программа была закрыта
    controller = world.controller
    controller.alarms = Alarms(path)
    tray: list[str] = []
    controller.on_alarm = tray.append
    controller.start_alarms()
    try:
        assert len(tray) == 1 and tray[0].startswith("Пока программа была закрыта, в ")
        assert tray[0].endswith("сработало: напоминаю: позвонить маме")
        assert world.bus.of("toast") == []  # страница окна ещё грузится — уведомление ждёт её
        controller.window_ready("main")
        assert world.bus.of("toast") == [{"kind": "alarm", "text": tray[0]}]

        timer = controller.alarms.add_timer(600, "чай")
        assert [item["id"] for item in world.bus.of("state")[-1]["timers"]] == [timer.id]
        controller.alarm_fired(timer, missed=False)
        assert world.bus.of("toast")[-1] == {"kind": "alarm", "text": "Таймер на 10 минут «чай»: время вышло"}
        assert len(controller.timer_cancel("")["timers"]) == 1  # без id ничего не снимается
        assert controller.timer_cancel(timer.id)["timers"] == [] and controller.alarms.active() == []
    finally:
        controller.stop()


class FakeAutostart:
    def __init__(self, fail: bool = False) -> None:
        self.on = False
        self.fail = fail

    def enabled(self) -> bool:
        return self.on

    def set(self, enabled: bool) -> None:
        if self.fail:
            raise OSError("нет доступа к папке")
        self.on = enabled


def test_autostart_toggle_is_not_in_config(world: SimpleNamespace) -> None:
    controller = world.controller
    assert controller.settings()["general"]["autostart"] is None  # переключателя нет
    controller.autostart = FakeAutostart()
    assert controller.settings()["general"]["autostart"] is False
    before = world.config_path.read_text(encoding="utf-8")
    result = controller.save_settings({"general": {"autostart": True}})
    assert result["ok"] and not result["restart"] and result["settings"]["general"]["autostart"] is True
    assert world.config_path.read_text(encoding="utf-8") == before  # config.yaml не тронут
    result = controller.save_settings({"general": {"autostart": False, "dry_run": False}})
    assert result["ok"] and controller.autostart.on is False and not controller.executor.dry_run
    controller.autostart = FakeAutostart(fail=True)
    assert controller.save_settings({"general": {"autostart": True}})["errors"] == [
        "Автозапуск: нет доступа к папке"
    ]


def test_live_settings_apply_without_restart(world: SimpleNamespace) -> None:
    result = world.controller.save_settings({"general": {"dry_run": False}, "ui": {"theme": "light"}})
    assert result["ok"] and not result["restart"] and result["settings"]["ui"]["theme"] == "light"
    assert world.executor.dry_run is False
    assert world.bus.of("state")[-1]["ui"]["theme"] == "light"
    assert not result["settings"]["restart_pending"]
    result = world.controller.save_settings({"voice": {"sounds": False}})
    assert result["ok"] and result["restart"] and world.controller.restart_pending
    assert world.controller.settings()["restart_pending"]  # баннер «Перезапустить» не пропадёт


def test_overlay_style_applies_without_restart(world: SimpleNamespace) -> None:
    result = world.controller.save_settings({"ui": {"overlay_style": "orb"}})
    assert result["ok"] and not result["restart"] and result["settings"]["ui"]["overlay_style"] == "orb"
    assert world.host.calls == [("overlay_style", "orb")]
    assert world.bus.of("state")[-1]["ui"]["overlay_style"] == "orb"
    assert "overlay_style: orb" in world.config_path.read_text(encoding="utf-8")
    world.controller.save_settings({"ui": {"theme": "light"}})  # оформление не менялось — окну не сообщают
    assert world.host.calls == [("overlay_style", "orb")]
    assert not world.controller.save_settings({"ui": {"overlay_style": "neon"}})["ok"]


def test_overlay_page_learns_when_overlay_shows_and_hides(world: SimpleNamespace) -> None:
    controller, bus = world.controller, world.bus
    world.host.next_glass = True  # с «Эффектами прозрачности» Windows страница рисует сферу полупрозрачной
    controller.window_ready("overlay")  # страница загрузилась, пока оверлей спрятан: готовит сферу заранее
    world.host.glass = True
    controller.overlay.voice_state("listening")
    world.host.next_glass = False  # «Эффекты прозрачности» выключили, пока сфера на экране
    controller.overlay_dismiss()  # клик по оверлею
    presence = [(payload, to) for kind, payload, to in bus.events if kind == "overlay"]
    assert presence == [
        ({"visible": False, "glass": True}, ("overlay",)),
        ({"visible": True, "glass": True}, ("overlay",)),
        ({"visible": False, "glass": False}, ("overlay",)),  # следующий показ — непрозрачный
    ]
    assert world.host.calls == [("show", "overlay"), ("hide", "overlay")]


def voice_command(controller: UiController) -> None:
    """Голосовая команда от «Джарвис» до конца ожидания продолжения."""
    overlay = controller.overlay
    status = VoiceStatus(controller.activity, controller.on_voice_state, overlay.command_done, overlay.phrase)
    for state in ("listening", "capturing", "recognizing", "processing"):
        status.voice_status(state)
    status.voice_status("recognized", "сделай громче")
    status.voice_result(controller.pipeline.handle_text("сделай громче", "voice"))
    status.voice_status("listening")  # продолжение без имени
    status.voice_status("idle")


def test_overlay_hides_after_voice_command_by_default(world: SimpleNamespace) -> None:
    # «Иногда после 5 секунд сфера не пропадает, а висит» — по умолчанию итог не держится.
    controller = world.controller
    assert controller.config.ui.overlay_keep is False
    voice_command(controller)
    assert controller.overlay.visible and controller.overlay._timer is not None  # уйдёт через 3 с


def test_voice_command_result_stays_until_next_phrase(world: SimpleNamespace) -> None:
    controller, host = world.controller, world.host
    assert controller.save_settings({"ui": {"overlay_keep": True}})["ok"]
    voice_command(controller)
    assert controller.overlay.visible and host.calls == [("show", "overlay")]  # итог на экране
    assert controller.overlay._timer is None  # и прятать его никто не собирается
    # Выключили в настройках — итог больше не держится: оверлей уйдёт через 3 с.
    result = controller.save_settings({"ui": {"overlay_keep": False}})
    assert result["ok"] and not result["restart"] and result["settings"]["ui"]["overlay_keep"] is False
    assert "overlay_keep: false" in world.config_path.read_text(encoding="utf-8")
    assert controller.overlay._timer is not None
    controller.overlay_dismiss()
    assert host.calls == [("show", "overlay"), ("hide", "overlay")]


def test_failed_voice_command_result_is_kept_too(world: SimpleNamespace) -> None:
    controller = world.controller
    overlay = controller.overlay
    controller.save_settings({"ui": {"overlay_keep": True}})
    status = VoiceStatus(controller.activity, controller.on_voice_state, overlay.command_done, overlay.phrase)
    for state in ("listening", "capturing", "recognizing", "processing"):
        status.voice_status(state)
    status.voice_status("recognized", "сделай кофе")
    status.voice_failed("сделай кофе", "Ошибка — подробности в логе")  # конвейер упал с исключением
    status.voice_status("idle")
    assert overlay.visible and overlay._timer is None  # ошибка видна до следующей фразы, как любой итог


def test_overlay_style_change_hides_overlay_that_holds_result(world: SimpleNamespace) -> None:
    controller, host = world.controller, world.host
    controller.save_settings({"ui": {"overlay_keep": True}})
    for state in ("listening", "recognizing", "processing"):
        controller.on_voice_state(state)
    controller.overlay.command_done()
    controller.on_voice_state("idle")
    assert controller.overlay.visible
    controller.save_settings({"ui": {"overlay_style": "orb"}})
    # Новое оформление — со следующего показа, а не после клика по оверлею.
    assert host.calls == [("show", "overlay"), ("overlay_style", "orb"), ("hide", "overlay")]
    assert not controller.overlay.visible


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


def test_failed_voice_offers_restart(world: SimpleNamespace) -> None:
    controller = world.controller
    controller.voice_loading = True
    assert not controller.state()["voice"]["failed"]
    controller.voice_failed("Голос недоступен: не удалось загрузить Whisper (Unable to open file)")
    voice = world.bus.of("state")[-1]["voice"]
    assert voice["failed"] and not voice["loading"] and "Unable to open file" in voice["problem"]
    controller.attach_voice(FakeVoice())
    assert not world.bus.of("state")[-1]["voice"]["failed"]


def test_voice_progress_while_loading(world: SimpleNamespace) -> None:
    from assistant.ui.app import UiOut

    controller = world.controller
    controller.voice_loading = True
    out = UiOut(controller)
    out.write("Загружаю распознавание речи (Whisper)…\n")
    assert world.bus.events[-1] == (
        "voice_progress",
        {"text": "Загружаю распознавание речи (Whisper)…", "fraction": None},
        ("main",),
    )
    out.progress("Скачиваю библиотеки CUDA: 40% (550 МБ из 1,3 ГБ)", 0.4)
    voice = controller.state()["voice"]
    assert (voice["step"], voice["progress"]) == ("Скачиваю библиотеки CUDA: 40% (550 МБ из 1,3 ГБ)", 0.4)
    out.write("! Не удалось скачать голос Piper (нет сети)\n")
    assert world.bus.events[-1] == (
        "toast",
        {"kind": "error", "text": "Не удалось скачать голос Piper (нет сети)"},
        None,
    )
    assert controller.state()["voice"]["progress"] == 0.4  # уведомление ход загрузки не сбрасывает
    controller.attach_voice(FakeVoice())
    voice = world.bus.of("state")[-1]["voice"]
    assert (voice["step"], voice["progress"], voice["loading"]) == ("", None, False)


def test_level_meter_goes_to_settings_and_overlay(world: SimpleNamespace) -> None:
    controller = world.controller
    controller.attach_voice(FakeVoice())
    controller.level_meter(True)
    thread = threading.Thread(target=controller.pump, args=(0.01,), daemon=True)
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


def test_voice_keys_are_saved_and_shared_with_chatgpt(world: SimpleNamespace) -> None:
    controller = world.controller
    controller.llm_save({"profile": "openai", "model": "gpt-6-luna", "key": "sk-old-openai-1234567890"})
    agent = world.pipeline.agent
    result = controller.tts_save_key("openai", " sk-new-openai-0987654321 ")
    assert result["ok"] and result["settings"]["tts_keys"]["openai"] == {
        "set": True, "hint": "••••4321", "from_system": False,
    }  # fmt: skip
    assert world.settings.key("OPENAI_API_KEY") == "sk-new-openai-0987654321"  # тот же ключ, что у ChatGPT
    assert len(agent.reconfigured) == 1  # нейросеть ChatGPT — клиент с новым ключом
    assert "sk-new-openai" not in str(result)  # ключ обратно не уходит
    assert controller.tts_save_key("yandex", "AQVNtest-yandex-key")["ok"]
    assert world.settings.key("YANDEX_API_KEY") == "AQVNtest-yandex-key" and len(agent.reconfigured) == 1
    assert not controller.tts_save_key("yandex", "мой ключ")["ok"]  # не похож на ключ
    assert not controller.tts_save_key("piper", "sk-anything-12345")["ok"]
    assert controller.tts_forget_key("yandex")["settings"]["tts_keys"]["yandex"]["set"] is False
    assert "YANDEX_API_KEY" not in world.env_path.read_text(encoding="utf-8")


def test_voice_key_pages(world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr(controller_module.webbrowser, "open", opened.append)
    for engine in ("openai", "yandex", "piper"):
        world.controller.open_tts_key_page(engine)
    assert opened == [
        "https://platform.openai.com/settings/organization/api-keys",
        "https://yandex.cloud/ru/docs/iam/operations/authentication/manage-api-keys",
    ]


def test_preview_of_keyed_voice_uses_unsaved_settings(
    world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np

    from assistant.tts.openai_tts import OpenAITTS
    from assistant.tts.yandex_tts import YandexTTS

    heard: list[tuple[str, ...]] = []

    def openai_synthesize(engine: OpenAITTS, text: str) -> tuple[np.ndarray, int]:
        heard.append((engine.voice, engine.style, engine.base_url, str(engine.rate)))
        return np.zeros(10, np.float32), 24_000

    def yandex_synthesize(engine: YandexTTS, text: str) -> tuple[np.ndarray, int]:
        heard.append((engine.voice, engine.role, str(engine.rate)))
        return np.zeros(10, np.float32), 48_000

    monkeypatch.setattr(OpenAITTS, "synthesize", openai_synthesize)
    monkeypatch.setattr(YandexTTS, "synthesize", yandex_synthesize)
    preview = world.controller._preview_engine
    assert (
        preview({"engine": "openai", "openai_voice": "onyx", "openai_style": "Бодро"}, 1.2, "Привет")[1]
        == 24_000
    )
    assert preview({"engine": "yandex", "yandex_voice": "alexander"}, 0.9, "Привет")[1] == 48_000
    assert heard == [("onyx", "Бодро", "https://api.openai.com/v1", "1.2"), ("alexander", "", "0.9")]


def test_failed_voice_is_explained_in_a_toast(world: SimpleNamespace) -> None:
    from assistant.tts.base import SpeechQueue

    class Silent:
        name = "yandex"

        def synthesize(self, text: str) -> Any:
            raise RuntimeError("нет ключа Yandex SpeechKit")

    class Player:
        def reset(self) -> None: ...

        def play(self, audio: Any, samplerate: int) -> bool:
            return True

        def stop(self) -> None: ...

    queue = SpeechQueue(Silent(), Player())
    UiController(
        config=world.controller.config, store=world.settings, pipeline=world.pipeline,
        executor=world.executor, commands=world.controller.commands, make_agent=FakeAgent, speech=queue,
        volume=FakeVolume(), bus=world.bus, activity=Activity(world.bus),
        confirmer=UiConfirmer(world.bus, timeout=0.3), log_file=Path("a.log"), journal_file=Path("j.jsonl"),
    )  # fmt: skip
    assert queue.on_failure is not None
    queue.on_failure("yandex", "нет ключа Yandex SpeechKit")
    text = world.bus.of("toast")[-1]["text"]
    assert text.startswith("Не удалось озвучить голосом Яндекса: нет ключа Yandex SpeechKit.")
    assert text.endswith("Пока говорю запасным голосом")
    queue.close()


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


def test_editor_save_reloads_quietly_and_tells_watcher(world: SimpleNamespace) -> None:
    written: list[bool] = []
    world.controller.on_commands_written = lambda: written.append(True)
    steps = [{"action": "get_time", "args": {}}]
    form = {"id": "tea", "phrases": ["завари чай"], "reply": "Ставлю", "hotkey": "", "steps": steps}
    result = world.controller.command_save("", "", form)
    assert result["ok"] and result["id"] == "tea" and written == [True]
    mine = next(pack for pack in result["packs"] if pack["user"])
    assert [c["id"] for c in mine["commands"]] == ["tea"]
    assert world.pipeline.handle_text("завари чай").reply == "Ставлю"  # матчер уже знает команду
    assert not world.bus.of("commands")  # без всплывающего «Команды обновлены»
    assert world.controller.command_delete("my.yaml", "tea")["ok"]
    assert world.controller.command_get("my.yaml", "tea") == {
        "ok": False,
        "errors": ["В my.yaml нет команды «tea»"],
    }


def test_editor_reports_errors_and_runs_draft_with_slots(world: SimpleNamespace) -> None:
    bad = world.controller.command_save("", "", {"id": "x", "phrases": ["икс"], "steps": []})
    assert not bad["ok"] and bad["errors"] == ["Добавьте хотя бы одно действие"]
    draft = {"id": "loud", "phrases": ["уровень звука {percent}"], "slots": {"percent": "int"},
             "steps": [{"action": "volume_set", "args": {"percent": "{percent}"}}]}  # fmt: skip
    test = world.controller.command_test("уровень звука сорок", draft, "")
    assert test["matches"][0]["slots"] == {"percent": 40} and test["matches"][0]["draft"]
    assert world.controller.command_run(draft, "уровень звука сорок") == {"ok": True, "errors": []}
    assert wait_until(lambda: world.controller.activity.recent)
    assert world.controller.activity.recent[0]["phrase"] == "уровень звука сорок"
    entries = [
        json.loads(line) for line in world.executor.journal.path.read_text(encoding="utf-8").splitlines()
    ]
    assert entries[0]["action"] == "volume_set" and entries[0]["args"] == {"percent": 40}
    assert not world.controller.command_run(draft, "")["ok"]  # слотам неоткуда взять значения
    names = [action["name"] for action in world.controller.actions()]
    assert "open_app" in names and "volume_set" in names


# --- Обновления ---------------------------------------------------------------------------

RELEASE = Release(
    version="0.7.0",
    title="Джарвис 0.7.0",
    notes="Новые голоса",
    page="https://github.com/TyMaxiss/jarvis/releases/tag/v0.7.0",
    installer="https://github.com/TyMaxiss/jarvis/releases/download/v0.7.0/Jarvis-Setup-0.7.0.exe",
    size=2 << 20,
    sha256="0" * 64,
)


class FakeUpdater:
    """Вместо assistant.updates.Updater: без GitHub и без запуска установщика."""

    def __init__(
        self, release: Release | None = RELEASE, *, supported: bool = True, due: bool = True
    ) -> None:
        self.release = release
        self.current = "0.6.0"
        self.supported = supported
        self.is_due = due
        self.error: UpdateError | None = None
        self.download_error: Exception | None = None
        self.checks = 0
        self.installed: list[Path] = []

    def due(self) -> bool:
        return self.is_due

    def check(self) -> Release | None:
        self.checks += 1
        if self.error is not None:
            raise self.error
        return self.release

    def download(self, release: Release, out: Any, folder: Path | None = None) -> Path:
        out.write(f"Скачиваю версию {release.version}\n")
        out.progress("Скачиваю обновление: 50% (1 МБ из 2 МБ)", 0.5)
        if self.download_error is not None:
            raise self.download_error
        return Path("Jarvis-Setup-0.7.0.exe")

    def install(self, installer: Path, folder: Path | None = None) -> None:
        self.installed.append(installer)


def test_update_check_shows_new_version(world: SimpleNamespace) -> None:
    controller = world.controller
    assert controller.update_check()["error"] == "Проверка обновлений недоступна"  # без проверяльщика
    controller.updater = FakeUpdater(supported=False)
    view = controller.update_check()
    assert view["checked"] and not view["checking"] and not view["error"] and not view["supported"]
    assert view["version"] == "0.6.0"  # с какой версией сравнивали
    assert view["release"]["version"] == "0.7.0" and view["release"]["notes"] == "Новые голоса"
    assert [event["checking"] for event in world.bus.of("update")] == [True, False]
    assert controller.state()["update"] == view
    controller.updater.error = UpdateError("нет сети")
    view = controller.update_check()
    assert view["error"] == "Не удалось проверить обновления: нет сети"
    assert view["release"]["version"] == "0.7.0"  # найденное раньше не забывается


def test_updates_on_start_notify_window_and_tray(world: SimpleNamespace) -> None:
    controller = world.controller
    tray: list[str] = []
    controller.on_update = tray.append
    controller.updater = FakeUpdater()
    controller.check_updates_on_start()
    text = "Вышла новая версия Джарвиса — 0.7.0. Обновить можно на главной."
    assert world.bus.of("toast") == [{"kind": "info", "text": text}] and tray == [text]
    controller.updater = FakeUpdater(supported=False)
    controller.check_updates_on_start()
    assert tray[-1] == "Вышла новая версия Джарвиса — 0.7.0. Обновите исходники: git pull и uv sync."


def test_updates_on_start_respect_settings_and_schedule(world: SimpleNamespace) -> None:
    controller = world.controller
    controller.updater = FakeUpdater(due=False)
    controller.check_updates_on_start()  # проверяли меньше суток назад
    assert controller.updater.checks == 0
    controller.updater = FakeUpdater(release=None)
    controller.check_updates_on_start()  # новой версии нет — молчим
    assert controller.updater.checks == 1 and not world.bus.of("toast")
    result = controller.save_settings({"general": {"check_updates": False}})
    assert result["ok"] and not result["restart"] and not result["settings"]["general"]["check_updates"]
    controller.updater = FakeUpdater()
    controller.check_updates_on_start()
    assert controller.updater.checks == 0


def test_update_install_downloads_installs_and_quits(world: SimpleNamespace) -> None:
    controller = world.controller
    updater = controller.updater = FakeUpdater()
    assert not controller.update_install()["installing"]  # новой версии ещё не нашли
    controller.update_check()
    assert controller.update_install()["installing"]
    assert wait_until(lambda: ("quit",) in world.host.calls)
    assert updater.installed == [Path("Jarvis-Setup-0.7.0.exe")]
    steps = [(event["step"], event["progress"]) for event in world.bus.of("update")]
    assert ("Скачиваю версию 0.7.0", None) in steps
    assert ("Скачиваю обновление: 50% (1 МБ из 2 МБ)", 0.5) in steps
    assert steps[-1] == ("Ставлю обновление — Джарвис перезапустится сам", None)


def test_failed_update_keeps_program_running(world: SimpleNamespace) -> None:
    controller = world.controller
    updater = controller.updater = FakeUpdater()
    controller.update_check()
    updater.download_error = UpdateError("не удалось скачать обновление: нет сети")
    controller.update_install()
    assert wait_until(lambda: not world.bus.of("update")[-1]["installing"])
    view = world.bus.of("update")[-1]
    assert view["error"] == "Обновление не удалось: не удалось скачать обновление: нет сети"
    assert not view["step"] and view["progress"] is None
    updater.download_error = PermissionError("папка занята")  # неожиданная ошибка — тоже не зависает
    controller.update_install()
    assert wait_until(lambda: not world.bus.of("update")[-1]["installing"])
    assert world.bus.of("update")[-1]["error"] == "Обновление не удалось: папка занята"
    assert ("quit",) not in world.host.calls and updater.installed == []


def test_source_checkout_does_not_install_updates(world: SimpleNamespace) -> None:
    controller = world.controller
    controller.updater = FakeUpdater(supported=False)
    controller.update_check()
    assert not controller.update_install()["installing"]


def test_update_page_opens_release(world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr(controller_module.webbrowser, "open", opened.append)
    world.controller.update_open_page()
    world.controller.updater = FakeUpdater()
    world.controller.update_check()
    world.controller.update_open_page()
    assert opened == ["https://github.com/TyMaxiss/jarvis/releases", RELEASE.page]


# --- Кнопка «Стоп» -----------------------------------------------------------------------


def test_stop_button_stops_task_speech_and_questions(world: SimpleNamespace) -> None:
    controller = world.controller
    hooks: list[str] = []
    world.pipeline.on_stop.append(lambda: hooks.append("речь"))
    answers: list[bool] = []
    asking = threading.Thread(
        target=lambda: answers.append(controller.confirmer.ask("Удалить файл?", use_voice=False))
    )
    asking.start()
    assert wait_until(lambda: world.bus.of("confirm"))
    controller.voice_stop()
    asking.join(2)
    assert answers == [False] and hooks == ["речь"]
    assert world.bus.of("stopped") == [{}]
    assert world.pipeline._cancel.is_set()


def test_typed_stop_is_a_stop_not_a_command(world: SimpleNamespace) -> None:
    world.controller.send_text("Джарвис, стоп")
    assert wait_until(lambda: world.bus.of("result"))
    result = world.bus.of("result")[0]
    assert result["phrase"] == "Джарвис, стоп" and result["reply"] == "Остановлено" and result["ok"]
    assert world.bus.of("stopped") == [{}]


class Talking:
    """Озвучка: говорит, пока speaking."""

    def __init__(self) -> None:
        self.speaking = False


def test_busy_reaches_main_window(world: SimpleNamespace) -> None:
    # Кнопка микрофона становится «Стоп», пока идёт задача или ассистент говорит.
    controller = world.controller
    controller.speech = Talking()
    assert controller.state()["busy"] is False
    thread = threading.Thread(target=controller.pump, args=(0.01,), daemon=True)
    thread.start()
    try:
        with world.pipeline._lock:  # задача из текста ещё выполняется
            assert wait_until(lambda: world.bus.of("busy") == [{"busy": True}])
            assert controller.state()["busy"] is True
        assert wait_until(lambda: world.bus.of("busy")[-1] == {"busy": False})
        controller.speech.speaking = True  # задача кончилась, а ответ ещё звучит
        assert wait_until(lambda: world.bus.of("busy")[-1] == {"busy": True})
        controller.speech.speaking = False
        assert wait_until(lambda: world.bus.of("busy")[-1] == {"busy": False})
    finally:
        controller.stop()
        thread.join(1)
    assert all(to == ("main",) for kind, _, to in world.bus.events if kind == "busy")
