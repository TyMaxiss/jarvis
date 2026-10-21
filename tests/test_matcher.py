"""Тесты локального матчера на настоящих YAML-паках из commands/."""

from __future__ import annotations

import logging

import pytest

from assistant.actions.registry import ActionRegistry
from assistant.commands.loader import Command
from assistant.config import MatcherConfig
from assistant.nlu.matcher import Matcher
from assistant.nlu.normalize import Normalizer
from tests.conftest import NO_WINDOW, NOTEPAD_WINDOW, YOUTUBE_WINDOW


def ids(result) -> list[str]:
    return [m.command.id for m in result.matches]


@pytest.mark.parametrize(
    ("phrase", "command_id"),
    [
        ("сделай громче", "volume_up"),
        ("Джарвис, сделай громче!", "volume_up"),
        ("сделай погромче", "volume_up"),
        ("громче сделай", "volume_up"),
        ("сделай тише", "volume_down"),
        ("сверни все окна", "minimize_all"),
        ("покажи рабочий стол", "minimize_all"),
        ("Пауза", "media_play_pause"),
        ("стоп", "media_stop"),
        ("следующий трек", "media_next"),
        ("выключи звук", "mute"),
        ("включи звук", "unmute"),
        ("открой ютуб", "open_youtube"),
        ("открой мне ютуб пожалуйста", "open_youtube"),
        ("открой YouTube", "open_youtube"),
        ("который час", "time"),
        ("закрой вкладку", "close_tab"),
        ("закрой окно", "window_close"),
        ("полный экран", "fullscreen"),
    ],
)
def test_static_phrases(matcher: Matcher, phrase: str, command_id: str) -> None:
    result = matcher.match(phrase, NO_WINDOW)
    assert result.matched, result.reason
    assert ids(result) == [command_id]


@pytest.mark.parametrize(
    ("phrase", "command_id", "slots"),
    [
        ("громкость на пятьдесят", "volume_set", {"percent": 50}),
        ("Громкость на 70%", "volume_set", {"percent": 70}),
        ("поставь громкость сорок пять процентов", "volume_set", {"percent": 45}),
        ("громкость на половину", "volume_set", {"percent": 50}),
        ("убавь громкость до двадцати", "volume_set", {"percent": 20}),
        ("сделай громче на двадцать", "volume_up_by", {"delta": 20}),
        ("тише на 5", "volume_down_by", {"delta": 5}),
        ("открой телегу", "open_app", {"name": "телегу"}),
        ("запусти вижуал студио код", "open_app", {"name": "вижуал студио код"}),
        ("закрой дискорд", "close_app", {"name": "дискорд"}),
        ("переключись на блокнот", "focus_window", {"title": "блокнот"}),
        ("найди рецепт борща", "web_search", {"query": "рецепт борща"}),
    ],
)
def test_templates(matcher: Matcher, phrase: str, command_id: str, slots: dict) -> None:
    result = matcher.match(phrase, NO_WINDOW)
    assert ids(result) == [command_id]
    assert result.matches[0].slots == slots


def test_most_specific_template_wins(matcher: Matcher) -> None:
    # «найди {query}» тоже подходит, но «найди на ютубе {query}» конкретнее.
    result = matcher.match("найди на ютубе котиков", NO_WINDOW)
    assert ids(result) == ["youtube_search"]
    assert result.matches[0].slots == {"query": "котиков"}


def test_static_phrase_beats_generic_template(matcher: Matcher) -> None:
    # Шаблон «открой {name}» тоже совпадает, но точная фраза важнее.
    result = matcher.match("открой диспетчер задач", NO_WINDOW)
    assert ids(result) == ["task_manager"]


def test_raw_text_is_kept_for_trailing_text_slot(matcher: Matcher) -> None:
    result = matcher.match("Джарвис, напечатай Привет, мир!", NO_WINDOW)
    assert ids(result) == ["type_text"]
    assert result.matches[0].slots == {"text": "Привет, мир!"}


def test_raw_text_is_not_used_when_it_has_extra_words(matcher: Matcher) -> None:
    result = matcher.match("напечатай привет пожалуйста", NO_WINDOW)
    assert result.matches[0].slots == {"text": "привет"}


class TestWhenPriority:
    def test_youtube_pause_when_youtube_is_active(self, matcher: Matcher) -> None:
        assert ids(matcher.match("пауза", YOUTUBE_WINDOW)) == ["youtube_pause"]
        assert ids(matcher.match("стоп", YOUTUBE_WINDOW)) == ["youtube_pause"]
        assert ids(matcher.match("полный экран", YOUTUBE_WINDOW)) == ["youtube_fullscreen"]

    def test_when_command_does_not_hijack_fuzzy_phrases(self, matcher: Matcher) -> None:
        # Команды YouTube проходят условие, но не похожи на фразу — побеждает обычная команда.
        assert ids(matcher.match("сделай погромче", YOUTUBE_WINDOW)) == ["volume_up"]
        assert ids(matcher.match("закрой вкладки", YOUTUBE_WINDOW)) == ["close_tab"]
        assert ids(matcher.match("громкость на 30", YOUTUBE_WINDOW)) == ["volume_set"]

    def test_generic_command_in_other_windows(self, matcher: Matcher) -> None:
        assert ids(matcher.match("пауза", NOTEPAD_WINDOW)) == ["media_play_pause"]
        assert ids(matcher.match("стоп", NOTEPAD_WINDOW)) == ["media_stop"]

    def test_process_condition(self, normalizer: Normalizer) -> None:
        commands = [
            make_command("save_notepad", ["сохрани"], when={"active_process": "notepad.exe"}),
            make_command("save_other", ["сохрани"]),
        ]
        matcher = build_matcher(normalizer, commands)
        assert ids(matcher.match("сохрани", NOTEPAD_WINDOW)) == ["save_notepad"]
        assert ids(matcher.match("сохрани", YOUTUBE_WINDOW)) == ["save_other"]

    def test_unsatisfied_when_excludes_command(self, normalizer: Normalizer) -> None:
        matcher = build_matcher(
            normalizer, [make_command("yt_only", ["перемотай"], when={"active_window_contains": "YouTube"})]
        )
        assert matcher.match("перемотай", NO_WINDOW).status == "not_found"
        assert ids(matcher.match("перемотай", YOUTUBE_WINDOW)) == ["yt_only"]


class TestComposite:
    def test_split_into_local_commands(self, matcher: Matcher) -> None:
        result = matcher.match("открой ютуб и сделай громче", NO_WINDOW)
        assert ids(result) == ["open_youtube", "volume_up"]

    @pytest.mark.parametrize("joiner", ["потом", "а затем", "и потом"])
    def test_other_conjunctions(self, matcher: Matcher, joiner: str) -> None:
        result = matcher.match(f"сверни все окна {joiner} поставь громкость на 30", NO_WINDOW)
        assert ids(result) == ["minimize_all", "volume_set"]
        assert result.matches[1].slots == {"percent": 30}

    def test_conjunction_inside_slot(self, matcher: Matcher) -> None:
        # «собаки» само по себе не команда, поэтому фраза берётся целиком.
        result = matcher.match("найди на ютубе кошки и собаки", NO_WINDOW)
        assert ids(result) == ["youtube_search"]
        assert result.matches[0].slots == {"query": "кошки и собаки"}

    def test_unknown_part_sends_whole_phrase_to_llm(self, matcher: Matcher) -> None:
        result = matcher.match("сделай громче и расскажи анекдот", NO_WINDOW)
        assert not result.matched

    def test_exact_phrase_with_conjunction_is_not_split(self, normalizer: Normalizer) -> None:
        matcher = build_matcher(
            normalizer,
            [
                make_command("light", ["свет"]),
                make_command("sound", ["звук"]),
                make_command("both", ["свет и звук"]),
            ],
        )
        assert ids(matcher.match("свет и звук", NO_WINDOW)) == ["both"]


class TestAmbiguityAndThreshold:
    def test_close_candidates_go_to_llm(self, normalizer: Normalizer) -> None:
        matcher = build_matcher(
            normalizer,
            [make_command("mail", ["открой почту"]), make_command("mails", ["открой почты"])],
        )
        result = matcher.match("открой почта", NO_WINDOW)
        assert result.status == "ambiguous"
        assert "mail" in result.reason

    def test_exact_match_is_not_ambiguous(self, matcher: Matcher) -> None:
        # «выключи звук» и «включи звук» похожи на 96%, но совпадение точное.
        assert ids(matcher.match("выключи звук", NO_WINDOW)) == ["mute"]

    def test_same_template_specificity_is_ambiguous(self, normalizer: Normalizer) -> None:
        matcher = build_matcher(
            normalizer,
            [
                make_command("a", ["покажи {x}"], slots={"x": "str"}),
                make_command("b", ["покажи {y}"], slots={"y": "str"}),
            ],
        )
        assert matcher.match("покажи погоду", NO_WINDOW).status == "ambiguous"

    def test_unknown_phrase(self, matcher: Matcher) -> None:
        result = matcher.match("расскажи анекдот про программистов", NO_WINDOW)
        assert result.status == "not_found"
        assert result.candidates  # топ кандидатов есть даже при неудаче

    def test_threshold_from_config(self, normalizer: Normalizer) -> None:
        commands = [make_command("louder", ["сделай громче"])]
        strict = build_matcher(normalizer, commands, MatcherConfig(threshold=99))
        loose = build_matcher(normalizer, commands, MatcherConfig(threshold=80))
        assert not strict.match("сделай погромче", NO_WINDOW).matched
        assert loose.match("сделай погромче", NO_WINDOW).matched

    def test_empty_phrase(self, matcher: Matcher) -> None:
        assert matcher.match("Джарвис", NO_WINDOW).status == "empty"


def test_debug_logs_top_candidates(
    normalizer: Normalizer, registry: ActionRegistry, caplog: pytest.LogCaptureFixture
) -> None:
    matcher = Matcher(normalizer, MatcherConfig(), debug=True)
    matcher.set_commands(
        [make_command("louder", ["сделай громче"]), make_command("quieter", ["сделай тише"])]
    )
    with caplog.at_level(logging.INFO, logger="assistant.nlu.matcher"):
        matcher.match("сделай громче", NO_WINDOW)
    assert "Кандидаты: louder (100" in caplog.text
    assert "quieter" in caplog.text


# --- помощники -------------------------------------------------------------------


def make_command(
    command_id: str, phrases: list[str], when: dict | None = None, slots: dict | None = None
) -> Command:
    return Command.model_validate(
        {
            "id": command_id,
            "phrases": phrases,
            "slots": slots or {},
            "when": when,
            "steps": [{"say": {"text": command_id}}],
        }
    )


def build_matcher(
    normalizer: Normalizer, commands: list[Command], config: MatcherConfig | None = None
) -> Matcher:
    matcher = Matcher(normalizer, config or MatcherConfig())
    matcher.set_commands(commands)
    return matcher
