"""Тесты реестра действий и генерации JSON-схемы для LLM."""

from __future__ import annotations

from typing import Annotated, Literal

import pytest
from pydantic import Field

from assistant.actions.registry import ActionRegistry, ArgsError, Risk, UnknownActionError


@pytest.fixture
def reg() -> ActionRegistry:
    reg = ActionRegistry()

    @reg.action("volume_set", "Установить громкость")
    def volume_set(percent: Annotated[int, Field(ge=0, le=100, description="Громкость, %")]) -> str:
        return f"Громкость {percent}%"

    @reg.action("media", "Медиа")
    def media(action: Literal["play_pause", "next"]) -> str:
        return action

    @reg.action("mute", "Звук")
    def mute(state: Annotated[bool | None, Field(description="true — выключить")] = None) -> str:
        return str(state)

    @reg.action("shutdown", "Выключить компьютер", risk=Risk.CONFIRM)
    def shutdown() -> str:
        return "ok"

    @reg.action("format_disk", "Отформатировать диск", risk=Risk.FORBIDDEN)
    def format_disk() -> str:
        return "never"

    return reg


def test_schema_is_generated_from_signature(reg: ActionRegistry) -> None:
    schema = reg.get("volume_set").tool_schema()
    assert schema == {
        "type": "function",
        "function": {
            "name": "volume_set",
            "description": "Установить громкость",
            "parameters": {
                "type": "object",
                "properties": {
                    "percent": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100,
                        "description": "Громкость, %",
                    }
                },
                "required": ["percent"],
                "additionalProperties": False,
            },
        },
    }


def test_literal_becomes_enum_and_optional_is_simplified(reg: ActionRegistry) -> None:
    media = reg.get("media").tool_schema()["function"]["parameters"]
    assert media["properties"]["action"]["enum"] == ["play_pause", "next"]
    mute = reg.get("mute").tool_schema()["function"]["parameters"]
    expected = {"type": "boolean", "default": None, "description": "true — выключить"}
    assert mute["properties"]["state"] == expected
    assert "required" not in mute


def test_no_titles_in_schema(reg: ActionRegistry) -> None:
    assert "title" not in repr(reg.tools())


def test_parameter_named_title_is_kept() -> None:
    # title — и ключевое слово схемы, и имя параметра focus_window: удаляться должно только первое.
    reg = ActionRegistry()

    @reg.action("focus", "Переключиться на окно")
    def focus(title: Annotated[str, Field(description="Часть заголовка")]) -> str:
        return title

    params = reg.get("focus").tool_schema()["function"]["parameters"]
    assert params["properties"] == {"title": {"type": "string", "description": "Часть заголовка"}}
    assert params["required"] == ["title"]


def test_every_required_parameter_is_described(registry: ActionRegistry) -> None:
    """Во всех настоящих действиях обязательные параметры описаны в properties."""
    for tool in registry.tools():
        params = tool["function"]["parameters"]
        assert set(params.get("required", [])) <= set(params["properties"]), tool["function"]["name"]


def test_tools_hide_forbidden_actions(reg: ActionRegistry) -> None:
    names = [tool["function"]["name"] for tool in reg.tools()]
    assert "format_disk" not in names
    assert "shutdown" in names
    assert "volume_set" not in [t["function"]["name"] for t in reg.tools(exclude=["volume_set"])]


def test_args_are_validated_and_coerced(reg: ActionRegistry) -> None:
    action = reg.get("volume_set")
    args = action.parse_args({"percent": "50"})
    assert args.percent == 50
    assert action.run(args) == "Громкость 50%"


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({}, "percent — не указан"),
        ({"percent": 101}, "не больше 100"),
        ({"percent": "много"}, "нужно целое число"),
        ({"percent": 5, "x": 1}, "x — лишний параметр"),
    ],
)
def test_bad_args_have_russian_messages(reg: ActionRegistry, raw: dict, message: str) -> None:
    with pytest.raises(ArgsError, match=message):
        reg.get("volume_set").parse_args(raw)


def test_unknown_action(reg: ActionRegistry) -> None:
    with pytest.raises(UnknownActionError):
        reg.get("fly")
    assert "fly" not in reg


def test_duplicate_registration_is_an_error(reg: ActionRegistry) -> None:
    with pytest.raises(ValueError):

        @reg.action("media", "Ещё раз")
        def media_again() -> str:
            return ""


def test_real_registry_has_stage1_actions(registry: ActionRegistry) -> None:
    expected = {
        "open_app", "close_app", "focus_window", "window_action", "get_active_window",
        "open_url", "web_search", "youtube_search", "media", "volume_set", "volume_change",
        "mute", "hotkey", "type_text", "get_time", "wait", "say",
    }  # fmt: skip
    assert expected <= set(registry.names())
    for tool in registry.tools():
        assert tool["function"]["description"]
        assert tool["function"]["parameters"]["type"] == "object"
