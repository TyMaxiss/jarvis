"""Агент через настоящий SDK openai и HTTP: локальный сервер отвечает потоком SSE.

Проверяет то, чего не видно на подделках: что SDK отправляет extra_body в теле запроса
(так DeepSeek получает {"thinking": {"type": "disabled"}}), а ответ с вызовом
инструмента, пришедший потоком, доходит до действия и обратно к модели.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from assistant.actions.executor import ActionExecutor
from assistant.actions.registry import ActionRegistry
from assistant.config import LlmConfig
from assistant.journal import ActionJournal
from assistant.llm.agent import Agent
from assistant.logging_setup import SecretMasker
from assistant.safety.confirm import StaticConfirmer
from assistant.safety.policy import SafetyPolicy
from tests.conftest import RecordingSpeaker


def sse(chunks: list[dict[str, Any]]) -> bytes:
    body = "".join(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n" for chunk in chunks)
    return (body + "data: [DONE]\n\n").encode("utf-8")


def delta_chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


class Provider:
    """OpenAI-совместимый сервер: первый ответ — вызов инструмента, второй — текст."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        provider = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                provider.requests.append(json.loads(self.rfile.read(length)))
                provider.headers.append({k.lower(): v for k, v in self.headers.items()})
                if len(provider.requests) == 1:
                    args = json.dumps({"name": "Телеграм"}, ensure_ascii=False)
                    body = sse(
                        [
                            delta_chunk({"role": "assistant", "content": None, "tool_calls": [
                                {"index": 0, "id": "call_1", "type": "function",
                                 "function": {"name": "open_app", "arguments": args[:5]}}]}),
                            delta_chunk({"tool_calls": [{"index": 0, "function": {"arguments": args[5:]}}]}),
                            delta_chunk({}, finish="tool_calls"),
                        ]
                    )  # fmt: skip
                else:
                    body = sse(
                        [
                            delta_chunk({"role": "assistant", "content": "Открыл "}),
                            delta_chunk({"content": "Телеграм."}),
                            delta_chunk({}, finish="stop"),
                        ]
                    )
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


@pytest.fixture
def provider() -> Iterator[Provider]:
    server = Provider()
    yield server
    server.server.shutdown()


def test_agent_over_http(provider: Provider, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_LLM_KEY", "test-secret-key-123")
    opened: list[str] = []
    registry = ActionRegistry()

    @registry.action("open_app", "Открыть программу")
    def open_app(name: str) -> str:
        opened.append(name)
        return f"Открываю {name}"

    executor = ActionExecutor(
        registry, SafetyPolicy(), StaticConfirmer(True), ActionJournal(tmp_path / "j.jsonl", SecretMasker())
    )
    settings = LlmConfig.model_validate(
        {
            "profile": "local",
            "profiles": {
                "local": {
                    "base_url": provider.url,
                    "model": "test-model",
                    "api_key_env": "TEST_LLM_KEY",
                    "extra_body": {"thinking": {"type": "disabled"}},
                }
            },
        }
    )
    speaker = RecordingSpeaker()
    result = Agent(settings, executor, speaker).run("запусти телегу")

    assert result.ok and opened == ["Телеграм"]
    assert speaker.said == ["Открыл Телеграм."]
    first, second = provider.requests
    assert first["model"] == "test-model" and first["stream"] is True
    assert first["thinking"] == {"type": "disabled"}  # extra_body попал в тело запроса
    assert [tool["function"]["name"] for tool in first["tools"]] == ["open_app"]
    assert provider.headers[0]["authorization"] == "Bearer test-secret-key-123"
    assert second["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "Открываю Телеграм",
    }
