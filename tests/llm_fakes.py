"""Поддельный OpenAI-клиент: отдаёт заранее заданные потоки ответов."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from typing import Any

import httpx2
import openai
from openai.types.chat import ChatCompletionChunk


def chunk(content: str | None = None, tool_calls: list[dict[str, Any]] | None = None, **extra: Any) -> Any:
    """Один кусочек потока chat completions."""
    delta: dict[str, Any] = {"role": "assistant", **extra}
    if content is not None:
        delta["content"] = content
    if tool_calls:
        delta["tool_calls"] = tool_calls
    return ChatCompletionChunk.model_validate(
        {
            "id": "chunk",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "fake",
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
    )


def text_reply(*parts: str) -> list[Any]:
    """Ответ текстом, по кусочкам."""
    return [chunk(part) for part in parts]


def tool_reply(*calls: tuple[str, dict[str, Any]]) -> list[Any]:
    """Ответ вызовами инструментов; аргументы приходят двумя кусочками, как в настоящем потоке."""
    chunks = []
    for index, (name, args) in enumerate(calls):
        raw = json.dumps(args, ensure_ascii=False)
        half = len(raw) // 2
        chunks.append(
            chunk(
                tool_calls=[
                    {
                        "index": index,
                        "id": f"call_{index}",
                        "type": "function",
                        "function": {"name": name, "arguments": raw[:half]},
                    }
                ]
            )
        )
        chunks.append(chunk(tool_calls=[{"index": index, "function": {"arguments": raw[half:]}}]))
    return chunks


def status_error(cls: type[openai.APIStatusError], status: int, message: str) -> openai.APIStatusError:
    """Ошибка провайдера с HTTP-статусом."""
    request = httpx2.Request("POST", "https://llm.example/v1/chat/completions")
    response = httpx2.Response(status, request=request)
    return cls(message, response=response, body={"error": {"message": message}})


def connection_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=httpx2.Request("POST", "https://llm.example/v1"))


class FakeStream:
    """Поток кусочков с отметкой о закрытии."""

    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks
        self.closed = False

    def __iter__(self) -> Any:
        return iter(self._chunks)

    def close(self) -> None:
        self.closed = True


class SilentStream(FakeStream):
    """Нейросеть «думает»: кусочки не приходят, пока не установлен release."""

    def __init__(self, chunks: list[Any]) -> None:
        super().__init__(chunks)
        self.release = threading.Event()

    def __iter__(self) -> Any:
        self.release.wait(timeout=5)
        return iter(self._chunks)


class FakeOpenAI:
    """Вместо openai.OpenAI: на каждый запрос отдаёт следующий ответ из списка.

    Элемент списка — список кусочков, готовый поток (FakeStream) или исключение.
    """

    def __init__(self, responses: list[Any], **client_kwargs: Any) -> None:
        self.responses = list(responses)
        self.client_kwargs = client_kwargs
        self.requests: list[dict[str, Any]] = []
        self.streams: list[FakeStream] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request: Any) -> FakeStream:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("Нейросеть спросили больше раз, чем было ответов")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        stream = item if isinstance(item, FakeStream) else FakeStream(item)
        self.streams.append(stream)
        return stream


def factory_for(fake: FakeOpenAI) -> Any:
    """Фабрика клиента для LlmClient: запоминает параметры и возвращает fake."""

    def factory(**kwargs: Any) -> FakeOpenAI:
        fake.client_kwargs = kwargs
        return fake

    return factory
