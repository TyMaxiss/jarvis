"""Поддельный клиент SDK anthropic: отдаёт заранее заданные потоки ответов Claude."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
from anthropic.types.beta import BetaMessage


def message(*blocks: dict[str, Any], stop_reason: str = "end_turn", **extra: Any) -> BetaMessage:
    """Окончательное сообщение Claude из блоков содержимого."""
    return BetaMessage.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": list(blocks),
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
            **extra,
        }
    )


def thinking(signature: str = "sig-1") -> dict[str, Any]:
    """Блок рассуждений: текст скрыт (display omitted), подпись есть."""
    return {"type": "thinking", "thinking": "", "signature": signature}


def text(value: str) -> dict[str, Any]:
    return {"type": "text", "text": value}


def tool_use(name: str, args: dict[str, Any], call_id: str = "toolu_1") -> dict[str, Any]:
    return {"type": "tool_use", "id": call_id, "name": name, "input": args}


class FakeClaudeStream:
    """Поток событий: кусочки текста, затем окончательное сообщение."""

    def __init__(self, final: BetaMessage, pieces: tuple[str, ...] = ()) -> None:
        self.final = final
        self.pieces = pieces
        self.closed = False
        self.release: threading.Event | None = None  # если задано — «думает», пока не отпустят

    def __enter__(self) -> FakeClaudeStream:
        return self

    def __exit__(self, *exc: object) -> None:
        self.closed = True

    def __iter__(self) -> Any:
        if self.release is not None:
            self.release.wait(timeout=5)
        return iter([SimpleNamespace(type="text", text=piece) for piece in self.pieces])

    def get_final_message(self) -> BetaMessage:
        return self.final


def reply(*blocks: dict[str, Any], pieces: tuple[str, ...] | None = None, **extra: Any) -> FakeClaudeStream:
    """Ответ Claude; текст по умолчанию приходит одним кусочком."""
    final = message(*blocks, **extra)
    if pieces is None:
        pieces = tuple(block["text"] for block in blocks if block["type"] == "text")
    return FakeClaudeStream(final, pieces)


def status_error(cls: type[anthropic.APIStatusError], status: int, text: str) -> anthropic.APIStatusError:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls(text, response=httpx2.Response(status, request=request), body=None)


class FakeAnthropic:
    """Вместо anthropic.Anthropic: на каждый запрос — следующий поток или исключение."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.client_kwargs: dict[str, Any] = {}
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))
        self.models = SimpleNamespace(
            list=lambda: [SimpleNamespace(id="claude-opus-5-5"), SimpleNamespace(id="claude-haiku-5-5")]
        )

    def _stream(self, **request: Any) -> FakeClaudeStream:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("Claude спросили больше раз, чем было ответов")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def factory_for(fake: FakeAnthropic) -> Any:
    def factory(**kwargs: Any) -> FakeAnthropic:
        fake.client_kwargs = kwargs
        return fake

    return factory
