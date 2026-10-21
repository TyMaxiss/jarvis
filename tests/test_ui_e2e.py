"""Интерфейс в браузере с настоящим Python за ним — без pywebview. Включается JARVIS_UI_E2E=1.

Страница из ui/dist открывается в Chromium (Playwright из Node, скрипт ui/e2e/app.e2e.mjs),
а window.pywebview.api подменяется мостом (ui/e2e/bridge.js): вызовы уходят по HTTP в
настоящий UiController, события возвращаются опросом. Так проверяется весь путь, кроме
самого pywebview: настройки нейросети, ключ в .env, config.yaml, проверка связи с
OpenAI-совместимым сервером (локальным), команда из быстрого чата.

Нужны Node.js, зависимости интерфейса (npm ci в папке ui) и Chromium для Playwright
(npx playwright install chromium).
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from assistant.actions.executor import ActionExecutor
from assistant.actions.registry import ActionRegistry
from assistant.commands.loader import CommandStore
from assistant.journal import ActionJournal
from assistant.logging_setup import SecretMasker
from assistant.nlu.matcher import Matcher
from assistant.pipeline import Pipeline
from assistant.safety.confirm import StaticConfirmer
from assistant.safety.policy import SafetyPolicy
from assistant.ui.activity import Activity, UiSpeaker
from assistant.ui.api import Api
from assistant.ui.confirm import UiConfirmer
from assistant.ui.controller import UiController
from assistant.ui.events import EventBus
from assistant.ui.settings import SettingsStore
from assistant.ui.windows import UI_DIST
from tests.conftest import NO_WINDOW, ROOT
from tests.test_llm_http import delta_chunk, sse
from tests.ui_fakes import FakeAgent, FakeVolume

pytestmark = pytest.mark.skipif(
    os.environ.get("JARVIS_UI_E2E") != "1", reason="включается через JARVIS_UI_E2E=1"
)

KEY = "sk-e2e-openai-key-1234567890"
CONTENT_TYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}


class FakeOpenAI:
    """OpenAI-совместимый сервер: на любой запрос отвечает потоком «Готово»."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                server.requests.append(json.loads(self.rfile.read(length)))
                server.headers.append({k.lower(): v for k, v in self.headers.items()})
                body = sse(
                    [delta_chunk({"role": "assistant", "content": "Готово"}), delta_chunk({}, finish="stop")]
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.http.server_address[1]}/v1"
        threading.Thread(target=self.http.serve_forever, daemon=True).start()


class BridgeHost:
    """Окна для контроллера: события копятся в очередях, браузер забирает их опросом."""

    def __init__(self) -> None:
        self.queues: dict[str, queue.Queue[str]] = {
            name: queue.Queue() for name in ("main", "overlay", "chat")
        }
        self.calls: list[tuple[str, ...]] = []

    def sender(self, name: str) -> Any:
        return self.queues[name].put

    def show(self, name: str) -> None:
        self.calls.append(("show", name))

    def hide(self, name: str) -> None:
        self.calls.append(("hide", name))

    def visible(self, name: str) -> bool:
        return name == "main"

    def minimize(self) -> None:
        self.calls.append(("minimize",))

    def toggle_maximize(self) -> None:
        self.calls.append(("maximize",))

    def resize_main(self, width: int, height: int) -> None:
        self.calls.append(("resize",))

    def quit(self) -> None:
        self.calls.append(("quit",))


def serve_bridge(controller: UiController, host: BridgeHost) -> ThreadingHTTPServer:
    """HTTP: файлы ui/dist, вызовы /api/<окно>/<метод> и события /events/<окно>."""
    apis = {name: Api(controller, name) for name in host.queues}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = self.path.split("?")[0]
            if path.startswith("/events/"):
                events = host.queues[path.removeprefix("/events/")]
                scripts = []
                try:
                    scripts.append(events.get(timeout=1))
                    while True:
                        scripts.append(events.get_nowait())
                except queue.Empty:
                    pass
                return self._send(200, "application/json", json.dumps(scripts).encode())
            file = (UI_DIST / path.lstrip("/")).resolve()
            if path == "/" or not file.is_file() or UI_DIST.resolve() not in file.parents:
                return self._send(404, "text/plain", b"not found")
            self._send(200, CONTENT_TYPES.get(file.suffix, "application/octet-stream"), file.read_bytes())

        def do_POST(self) -> None:
            _, _, window, method = self.path.split("/", 3)
            args = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"[]")
            try:
                result = getattr(apis[window], method)(*args)
            except Exception as exc:
                return self._send(500, "application/json", json.dumps({"error": repr(exc)}).encode())
            self._send(200, "application/json", json.dumps({"result": result}, ensure_ascii=False).encode())

        def _send(self, status: int, kind: str, body: bytes) -> None:
            try:
                self.send_response(status)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # браузер закрыли, пока шёл опрос событий

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture
def fake_openai() -> Iterator[FakeOpenAI]:
    server = FakeOpenAI()
    yield server
    server.http.shutdown()


def test_interface_with_real_python(
    tmp_path: Path,
    registry: ActionRegistry,
    matcher: Matcher,
    fake_openai: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not (UI_DIST / "index.html").exists():
        pytest.skip("интерфейс не собран (npm --prefix ui run build)")
    node = shutil.which("node")
    if node is None:
        pytest.skip("нет Node.js")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    commands = tmp_path / "commands"  # редактор пишет в my.yaml — не в папку репозитория
    shutil.copytree(ROOT / "commands", commands)
    store = CommandStore(commands, "my.yaml", registry)
    store.reload()
    matcher.set_commands(store.commands)
    config_path = tmp_path / "config.yaml"
    config_text = (ROOT / "config.example.yaml").read_text(encoding="utf-8")
    # ChatGPT отвечает с локального сервера, а не с api.openai.com.
    config_text = config_text.replace("base_url: https://api.openai.com/v1", f"base_url: {fake_openai.url}")
    config_path.write_text(config_text, encoding="utf-8")
    env_path = tmp_path / ".env"

    bus = EventBus()
    activity = Activity(bus)
    executor = ActionExecutor(
        registry,
        SafetyPolicy(),
        StaticConfirmer(True),
        ActionJournal(tmp_path / "j.jsonl", SecretMasker()),
        dry_run=True,
    )
    settings = SettingsStore(config_path, env_path)
    agent: Any = FakeAgent(settings.load().llm)
    pipeline = Pipeline(matcher, executor, UiSpeaker(activity), lambda: NO_WINDOW, agent=agent)
    controller = UiController(
        config=settings.load(), store=settings, pipeline=pipeline, executor=executor, commands=store,
        make_agent=FakeAgent, speech=None, volume=FakeVolume(), bus=bus, activity=activity,
        confirmer=UiConfirmer(bus, timeout=1), log_file=tmp_path / "a.log", journal_file=tmp_path / "j.jsonl",
        force_dry_run=True,  # как с --dry-run: действия только пишутся в журнал
    )  # fmt: skip
    pipeline.on_command_step = lambda n, total, step: activity.step(n, total, step.action, step.text, step.ok)
    host = BridgeHost()
    controller.host = host
    bridge = serve_bridge(controller, host)
    shots = Path(os.environ.get("JARVIS_UI_E2E_SHOTS") or tmp_path / "shots")
    shots.mkdir(parents=True, exist_ok=True)
    try:
        run = subprocess.run(
            [node, str(ROOT / "ui" / "e2e" / "app.e2e.mjs"), f"http://127.0.0.1:{bridge.server_address[1]}",
             str(shots), KEY],
            capture_output=True, text=True, timeout=120, encoding="utf-8",
        )  # fmt: skip
    finally:
        bridge.shutdown()
        bus.close()
    assert run.returncode == 0, run.stderr
    report = json.loads(run.stdout.strip().splitlines()[-1])
    results = report["results"]
    assert report["errors"] == [], report["errors"]

    # Главная: состояние пришло из Python.
    assert "голос" in results["heading"].lower() and results["llmCard"] == "DeepSeek"
    # Проверка связи ушла на сервер с вставленным ключом, как запрос агента: с инструментом
    # и без рассуждений (иначе модели OpenAI не вызывают инструменты).
    check = fake_openai.requests[0]
    assert fake_openai.headers[0]["authorization"] == f"Bearer {KEY}"
    assert check["model"] == "gpt-6-luna" and check["reasoning_effort"] == "none"
    assert check["tools"][0]["function"]["name"] == "ping"
    assert results["check"].startswith("Работает: ответ за")
    # Сохранение: ключ в .env, профиль в config.yaml, в окне — только хвост ключа.
    assert f"OPENAI_API_KEY='{KEY}'" in env_path.read_text(encoding="utf-8")
    saved = settings.load()
    assert saved.llm.profile == "openai" and saved.ui.theme == "light"
    assert results["saved"] == "Сохранено. Отвечает ChatGPT · gpt-6-luna"
    assert "сейчас" in results["chatgptTile"] and "ключ сохранён" in results["chatgptTile"]
    assert results["keyPlaceholder"].startswith("Сохранён ••••7890")
    assert results["theme"] == "light"
    # Быстрый чат: команда выполнена (dry-run), шаг и итог дошли до окна и до главной.
    assert "Шаг 1" in results["chatSteps"]
    assert controller.activity.recent[0]["phrase"] == "сделай громче"
    assert results["recent"] == "сделай громче"
    # История: фраза из журнала (не из памяти окна) вместе с действием команды.
    assert results["historyDay"] == "Сегодня"
    assert results["historyAction"] == "volume_change(delta=10)"
    # Редактор команд: новая команда проверена настоящим матчером, сохранена в my.yaml и загружена.
    assert results["commandTest"].startswith("Сработает эта команда (оценка 100")
    assert results["commandId"] == "proverka_redaktora"
    saved = '- id: proverka_redaktora\n  phrases: ["проверка редактора"]\n  steps:\n'
    saved += '    - say: {text: "Редактор работает"}\n'
    assert (commands / "my.yaml").read_text(encoding="utf-8").endswith(saved)
    assert any(command.id == "proverka_redaktora" for command in store.commands)
    assert KEY not in json.dumps(controller.settings())
