"""Локальный поиск команды по фразе, без LLM.

Правила выбора:
* фразы-шаблоны со слотами («громкость на {percent}») превращаются в регулярные
  выражения, остальные фразы сравниваются нечётко (rapidfuzz, порог из конфига);
* команды с невыполненным условием when не участвуют, а с выполненным — вытесняют
  команды без when;
* точное совпадение фразы побеждает сразу; при нечётком совпадении, если два лучших
  кандидата ближе ambiguity_gap пунктов, фраза считается неоднозначной (уходит в LLM);
* нечёткие совпадения выше порога важнее шаблонов: «открой ютуб» — это команда
  открытия сайта, а не open_app из шаблона «открой {name}»;
* среди шаблонов выигрывает самый конкретный (с большей долей постоянного текста);
* составная фраза («открой ютуб и сделай громче») делится по союзам; если каждая часть
  нашлась локально, выполняются все части по очереди, иначе фраза пробуется целиком.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from rapidfuzz import fuzz

from assistant.actions.winapi import WindowInfo
from assistant.commands.loader import PLACEHOLDER_RE, Command
from assistant.config import MatcherConfig
from assistant.nlu.names import skeleton
from assistant.nlu.normalize import Normalizer

log = logging.getLogger(__name__)

_SLOT_PATTERNS = {"int": r"\d+", "float": r"\d+(?:\.\d+)?", "str": r".+?"}
_SLOT_CONVERTERS: dict[str, Callable[[str], Any]] = {"int": int, "float": float, "str": str}
_LEADING_SEPARATORS = " \t,.:;—–-"

Status = Literal["matched", "ambiguous", "not_found", "empty"]


@dataclass(frozen=True)
class _Phrase:
    """Подготовленная фраза команды."""

    command: Command
    source: str  # фраза как в YAML
    text: str  # нормализованная фраза (для шаблона — с {слотами})
    skeleton: str  # латинский «скелет» для сравнения ютуб ≈ youtube
    regex: re.Pattern[str] | None = None
    specificity: int = 0  # длина постоянного текста шаблона
    tail_slot: str | None = None  # строковый слот в конце шаблона
    anchor: tuple[str, ...] = ()  # слова шаблона прямо перед хвостовым слотом


@dataclass(frozen=True)
class Candidate:
    """Кандидат на совпадение."""

    command: Command
    score: float
    phrase: str
    kind: Literal["exact", "fuzzy", "template"]
    slots: dict[str, Any] = field(default_factory=dict)
    specificity: int = 0

    def describe(self) -> str:
        """Строка для лога: id, оценка и фраза."""
        extra = f", слоты {self.slots}" if self.slots else ""
        return f"{self.command.id} ({self.score:.0f}, {self.kind}: «{self.phrase}»{extra})"


@dataclass(frozen=True)
class MatchedCommand:
    """Найденная команда со значениями слотов."""

    command: Command
    slots: dict[str, Any]
    score: float
    phrase: str
    text: str  # часть фразы пользователя, которая совпала


@dataclass
class MatchResult:
    """Итог поиска: найденные команды (несколько — для составной фразы) или причина неудачи."""

    status: Status
    normalized: str
    matches: list[MatchedCommand] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    reason: str = ""

    @property
    def matched(self) -> bool:
        """Фраза распознана локально."""
        return self.status == "matched"


@dataclass
class _SegmentResult:
    chosen: Candidate | None
    status: Status
    candidates: list[Candidate]
    reason: str = ""


def _fuzzy_score(text: str, text_skeleton: str, phrase: _Phrase) -> float:
    """Нечёткая оценка 0–100: посимвольно, без учёта порядка слов и по «скелету»."""
    return max(
        fuzz.ratio(text, phrase.text),
        fuzz.token_sort_ratio(text, phrase.text),
        fuzz.ratio(text_skeleton, phrase.skeleton),
    )


def _raw_tail(original: str, anchor: Sequence[str]) -> str | None:
    """Текст исходной фразы после слов-якорей — с регистром и пунктуацией.

    Нужен для строковых слотов вроде «напечатай {text}»: печатать надо «Привет, мир!»,
    а не нормализованное «привет мир».
    """
    lowered = original.lower().replace("ё", "е")
    if not anchor or len(lowered) != len(original):
        return None
    pattern = r"(?<!\w)" + r"\W+".join(re.escape(word) for word in anchor) + r"(?!\w)"
    found = list(re.finditer(pattern, lowered))
    if not found:
        return None
    tail = original[found[-1].end() :].lstrip(_LEADING_SEPARATORS).rstrip()
    return tail or None


class Matcher:
    """Сопоставляет фразы с командами.

    Args:
        normalizer: нормализатор текста (тот же, что для фраз пользователя).
        config: порог, разрыв неоднозначности и союзы для составных фраз.
        debug: писать топ-3 кандидата в лог на уровне INFO.
    """

    def __init__(self, normalizer: Normalizer, config: MatcherConfig, debug: bool = False) -> None:
        self._normalizer = normalizer
        self._bare = Normalizer()  # без удаления слов: для сверки исходного текста слотов
        self._config = config
        self._debug = debug
        self._phrases: list[_Phrase] = []
        self._split_words = sorted(
            (tuple(normalizer.tokens(word)) for word in config.split_words if normalizer.tokens(word)),
            key=len,
            reverse=True,
        )

    def set_commands(self, commands: Iterable[Command]) -> None:
        """Подготавливает фразы команд (вызывается при загрузке и перезагрузке)."""
        phrases = []
        for command in commands:
            for source in command.phrases:
                phrase = self._compile(command, source)
                if phrase is not None:
                    phrases.append(phrase)
        self._phrases = phrases

    def _compile(self, command: Command, source: str) -> _Phrase | None:
        """Нормализует фразу; шаблон со слотами превращает в регулярное выражение."""
        parts = PLACEHOLDER_RE.split(source)
        if len(parts) == 1:
            text = self._normalizer.normalize(source)
            if not text:
                log.warning("Фраза «%s» команды %s пуста после нормализации", source, command.id)
                return None
            return _Phrase(command, source, text, skeleton(text))

        regex_parts: list[str] = []
        display: list[str] = []
        literal: list[str] = []
        tail_slot: str | None = None
        anchor: tuple[str, ...] = ()
        for index, part in enumerate(parts):
            if index % 2 == 0:
                tokens = self._normalizer.tokens(part)
                if tokens:
                    regex_parts.append(re.escape(" ".join(tokens)))
                    display.append(" ".join(tokens))
                    literal.extend(tokens)
                    anchor = tuple(tokens)
                    tail_slot = None
            else:
                slot_type = command.slot_type(part)
                regex_parts.append(f"(?P<{part}>{_SLOT_PATTERNS[slot_type]})")
                display.append(f"{{{part}}}")
                tail_slot = part if slot_type == "str" else None
        text = " ".join(display)
        return _Phrase(
            command=command,
            source=source,
            text=text,
            skeleton=skeleton(text),
            regex=re.compile(" ".join(regex_parts)),
            specificity=len(" ".join(literal)),
            tail_slot=tail_slot,
            anchor=anchor if tail_slot else (),
        )

    # --- Поиск ---------------------------------------------------------------------

    def match(self, text: str, window: WindowInfo) -> MatchResult:
        """Ищет команду (или последовательность команд) для фразы.

        Args:
            text: фраза пользователя как есть.
            window: активное окно — для условий when.
        """
        tokens = self._normalizer.tokens(text)
        normalized = " ".join(tokens)
        if not normalized:
            return MatchResult("empty", normalized, reason="пустая фраза")

        whole = self._match_segment(normalized, window, original=text)
        # Точное совпадение всей фразы важнее деления: команда «свет и звук» не делится.
        if not (whole.chosen and whole.chosen.kind == "exact"):
            parts = self._split(tokens)
            if len(parts) > 1:
                segments = [self._match_segment(part, window) for part in parts]
                chosen = [seg.chosen for seg in segments if seg.chosen]
                if len(chosen) == len(parts):
                    result = MatchResult(
                        "matched",
                        normalized,
                        matches=[self._to_match(c, part) for c, part in zip(chosen, parts, strict=True)],
                        candidates=chosen,
                        reason=f"составная фраза из {len(parts)} частей",
                    )
                    self._log(text, result)
                    return result

        if whole.chosen:
            result = MatchResult(
                "matched",
                normalized,
                matches=[self._to_match(whole.chosen, normalized)],
                candidates=whole.candidates,
            )
        else:
            result = MatchResult(whole.status, normalized, candidates=whole.candidates, reason=whole.reason)
        self._log(text, result)
        return result

    def _split(self, tokens: list[str]) -> list[str]:
        """Делит фразу по союзам; пустые части отбрасываются."""
        parts: list[list[str]] = [[]]
        i = 0
        while i < len(tokens):
            for word in self._split_words:
                if tuple(tokens[i : i + len(word)]) == word:
                    parts.append([])
                    i += len(word)
                    break
            else:
                parts[-1].append(tokens[i])
                i += 1
        return [" ".join(part) for part in parts if part]

    def _match_segment(self, text: str, window: WindowInfo, original: str | None = None) -> _SegmentResult:
        """Ищет одну команду для нормализованного фрагмента."""
        text_skeleton = skeleton(text)
        statics: dict[str, Candidate] = {}
        templates: list[Candidate] = []
        for phrase in self._phrases:
            command = phrase.command
            if command.when is not None and not command.when.matches(window):
                continue
            if phrase.regex is None:
                if phrase.text == text:
                    candidate = Candidate(command, 100.0, phrase.source, "exact")
                else:
                    score = _fuzzy_score(text, text_skeleton, phrase)
                    candidate = Candidate(command, score, phrase.source, "fuzzy")
                best = statics.get(command.id)
                if best is None or (candidate.score, candidate.kind == "exact") > (
                    best.score,
                    best.kind == "exact",
                ):
                    statics[command.id] = candidate
                continue
            found = phrase.regex.fullmatch(text)
            if found:
                slots = self._extract_slots(phrase, found, original)
                templates.append(
                    Candidate(command, 100.0, phrase.source, "template", slots, phrase.specificity)
                )

        ranked = sorted(statics.values(), key=lambda c: -c.score)
        templates.sort(key=lambda c: -c.specificity)
        candidates = sorted(ranked[:3] + templates[:3], key=lambda c: (-c.score, -c.specificity))

        exact = [c for c in ranked if c.kind == "exact"]
        if exact:
            return self._decide(exact, candidates, lambda c: c.score, gap=0.0001)
        if ranked and ranked[0].score >= self._config.threshold:
            viable = [c for c in ranked if c.score >= self._config.threshold]
            prioritized = [c for c in viable if c.command.when is not None]
            group = prioritized or ranked
            return self._decide(group, candidates, lambda c: c.score, gap=self._config.ambiguity_gap)
        if templates:
            return self._decide(templates, candidates, lambda c: c.specificity, gap=0.0001)
        return _SegmentResult(None, "not_found", candidates, "нет подходящей команды")

    @staticmethod
    def _decide(
        group: list[Candidate],
        candidates: list[Candidate],
        key: Callable[[Candidate], float],
        gap: float,
    ) -> _SegmentResult:
        """Выбирает лучшего кандидата группы или признаёт фразу неоднозначной.

        Команды с выполненным условием when вытесняют остальные.
        """
        prioritized = [c for c in group if c.command.when is not None]
        ordered = sorted(prioritized or group, key=key, reverse=True)
        best = ordered[0]
        rivals = [c for c in ordered[1:] if c.command.id != best.command.id]
        if rivals and key(best) - key(rivals[0]) < gap:
            reason = f"неоднозначно: {best.command.id} или {rivals[0].command.id}"
            return _SegmentResult(None, "ambiguous", candidates, reason)
        # Выбранный кандидат — первым в списке (для лога и «Проверить фразу»).
        return _SegmentResult(best, "matched", [best, *(c for c in candidates if c is not best)])

    def _extract_slots(self, phrase: _Phrase, found: re.Match[str], original: str | None) -> dict[str, Any]:
        """Значения слотов с приведением типов; хвостовой текст берётся из исходной фразы."""
        slots = {
            name: _SLOT_CONVERTERS[phrase.command.slot_type(name)](value)
            for name, value in found.groupdict().items()
        }
        if phrase.tail_slot and original:
            raw = _raw_tail(original, phrase.anchor)
            # Исходный текст берётся, только если в нём нет лишних слов (паразитов и т.п.).
            value = str(slots[phrase.tail_slot])
            if raw and self._bare.normalize(raw) == self._bare.normalize(value):
                slots[phrase.tail_slot] = raw
        return slots

    @staticmethod
    def _to_match(candidate: Candidate, text: str) -> MatchedCommand:
        return MatchedCommand(
            candidate.command, dict(candidate.slots), candidate.score, candidate.phrase, text
        )

    def _log(self, text: str, result: MatchResult) -> None:
        """Пишет в лог итог и топ-3 кандидата (в debug-режиме — на уровне INFO)."""
        top = "; ".join(c.describe() for c in result.candidates[:3]) or "нет"
        level = logging.INFO if self._debug else logging.DEBUG
        log.log(level, "Фраза «%s» → «%s»: %s. Кандидаты: %s", text, result.normalized,
                result.reason or result.status, top)  # fmt: skip
