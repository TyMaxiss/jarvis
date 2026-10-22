"""Анимация оверлея: путь окна, кадры появления и ухода, разворот на полпути, связка с окнами."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from assistant.ui import win32
from assistant.ui.slide import HIDE_TIME, SHOW_TIME, Slide, Track, ease_in, ease_out

# Окно 480×64 над панелью задач: на месте его верх в 952, край рабочей области — 1040.
TRACK = Track(x=720, top=952, edge=1040, width=480, height=64, radius=8)
FRAME = 1 / 60


def wait_for(condition: Callable[[], bool], timeout: float = 5) -> bool:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.001)
    return True


class Screen:
    """Окно и экран для Slide: часы идут только кадрами, всё сделанное с окном записано."""

    def __init__(self, track: Track = TRACK) -> None:
        self.track = track
        self.now = 0.0
        self.frames = 0
        self.calls: list[Any] = []  # "appear", y (кадр), "vanish"
        self.at_frame: dict[int, Callable[[], None]] = {}  # что сделать на кадре с номером
        self.fail: set[str] = set()  # какие вызовы окна упадут (по одному разу)

    def _call(self, name: str) -> None:
        if name in self.fail:
            self.fail.discard(name)
            raise RuntimeError(f"окно не смогло: {name}")

    def appear(self) -> Track:
        self._call("appear")
        self.calls.append("appear")
        return self.track

    def place(self, track: Track, y: int) -> None:
        assert track is self.track
        self._call("place")
        self.calls.append(y)

    def vanish(self) -> None:
        self._call("vanish")
        self.calls.append("vanish")

    def wait_frame(self) -> None:
        self.now += FRAME
        self.frames += 1
        action = self.at_frame.pop(self.frames, None)
        if action is not None:
            action()

    def clock(self) -> float:
        return self.now

    @property
    def ys(self) -> list[int]:
        return [call for call in self.calls if isinstance(call, int)]

    def slide(self, animate: bool = True) -> Slide:
        return Slide(self.appear, self.place, self.vanish, lambda: animate, self.wait_frame, clock=self.clock)


def shown(screen: Screen) -> bool:
    return screen.ys[-1:] == [screen.track.top]


def gone(screen: Screen) -> bool:
    return screen.calls[-1:] == ["vanish"]


def test_track_geometry() -> None:
    assert (TRACK.y(0), TRACK.y(0.5), TRACK.y(1)) == (1040, 996, 952)
    assert TRACK.visible(1040) == 0  # верх окна на краю: не видно ничего
    assert TRACK.visible(1000) == 40  # над краем 40 пикселей из 64
    assert TRACK.visible(976) is None  # окно целиком над краем — обрезать нечего
    assert TRACK.visible(952) is None


def test_easing() -> None:
    for ease in (ease_in, ease_out):
        assert ease(0) == 0 and ease(1) == 1
    # Появление быстрое в начале и плавное к концу, уход — наоборот.
    assert ease_out(0.25) > 0.25 > ease_in(0.25)


def test_show_slides_up_from_edge() -> None:
    screen = Screen()
    screen.slide().show()
    assert wait_for(lambda: shown(screen))
    ys = screen.ys
    assert screen.calls[0] == "appear" and len(ys) >= 8
    assert TRACK.top < ys[0] < TRACK.edge
    assert all(a > b for a, b in pairwise(ys)), "только вверх, без одинаковых кадров"
    steps = [a - b for a, b in pairwise([TRACK.edge, *ys])]
    assert steps[0] > 3 * steps[-1], "к концу замедляется"
    time.sleep(0.05)
    # Длительность — по часам, а не по числу кадров; доехав, окно больше не трогают.
    assert abs(screen.frames * FRAME - SHOW_TIME) < FRAME
    assert screen.ys == ys


def test_hide_slides_down_behind_edge_then_hides() -> None:
    screen = Screen()
    slide = screen.slide()
    slide.show()
    assert wait_for(lambda: shown(screen))
    frames, before = screen.frames, len(screen.ys)
    slide.hide()
    assert wait_for(lambda: gone(screen))
    ys = screen.ys[before - 1 :]
    assert ys[0] == TRACK.top and ys[-1] == TRACK.edge
    assert all(a < b for a, b in pairwise(ys)), "только вниз"
    steps = [b - a for a, b in pairwise(ys)]
    assert steps[-1] > 3 * steps[0], "к концу ускоряется"
    assert abs((screen.frames - frames) * FRAME - HIDE_TIME) < FRAME
    assert screen.calls.count("vanish") == 1


def test_hide_while_showing_turns_back_from_current_place() -> None:
    screen = Screen()
    slide = screen.slide()
    screen.at_frame[5] = slide.hide
    slide.show()
    assert wait_for(lambda: gone(screen))
    ys = screen.ys
    turn = ys.index(min(ys))
    assert TRACK.top < ys[turn] < TRACK.edge, "развернулось, не доехав"
    assert all(a > b for a, b in pairwise(ys[: turn + 1]))
    assert all(a < b for a, b in pairwise(ys[turn:]))
    assert ys[turn + 1] - ys[turn] < 10, "без скачка в момент разворота"
    assert ys[-1] == TRACK.edge and screen.calls.count("appear") == 1


def test_show_while_hiding_turns_back_without_hiding() -> None:
    screen = Screen()
    slide = screen.slide()
    slide.show()
    assert wait_for(lambda: shown(screen))
    before = len(screen.ys)
    screen.at_frame[screen.frames + 4] = slide.show
    slide.hide()
    assert wait_for(lambda: len(screen.ys) > before and shown(screen))
    ys = screen.ys[before - 1 :]
    turn = ys.index(max(ys))
    assert TRACK.top < ys[turn] < TRACK.edge
    assert "vanish" not in screen.calls and screen.calls.count("appear") == 1


def test_every_appearance_measures_the_screen_again() -> None:
    screen = Screen()
    slide = screen.slide()
    for _ in range(2):
        slide.show()
        assert wait_for(lambda: shown(screen))
        slide.hide()
        assert wait_for(lambda: gone(screen))
    assert screen.calls.count("appear") == 2 and screen.calls.count("vanish") == 2


def test_without_animation_window_jumps() -> None:
    screen = Screen()
    slide = screen.slide(animate=False)
    slide.show()
    assert wait_for(lambda: shown(screen))
    slide.hide()
    assert wait_for(lambda: gone(screen))
    assert screen.calls == ["appear", TRACK.top, "vanish"] and screen.frames == 0


def test_requests_that_change_nothing_do_nothing() -> None:
    screen = Screen()
    slide = screen.slide()
    slide.hide()  # и так спрятано
    time.sleep(0.05)
    assert screen.calls == []
    slide.show()
    assert wait_for(lambda: shown(screen))
    calls = list(screen.calls)
    slide.show()  # и так на месте
    time.sleep(0.05)
    assert screen.calls == calls


@pytest.mark.parametrize("broken", ["appear", "place", "vanish"])
def test_window_error_hides_overlay_and_next_show_works(
    broken: str, caplog: pytest.LogCaptureFixture
) -> None:
    screen = Screen()
    slide = screen.slide()
    screen.fail.add(broken)
    slide.show()
    if broken == "vanish":
        assert wait_for(lambda: shown(screen))
        slide.hide()  # уход упадёт на последнем шаге
    assert wait_for(lambda: "Не удалось" in caplog.text)
    assert wait_for(lambda: gone(screen)), "после ошибки оверлей прячется"
    appeared = screen.calls.count("appear")
    slide.show()
    assert wait_for(lambda: shown(screen) and screen.calls.count("appear") == appeared + 1)


def test_closed_slide_ignores_requests() -> None:
    screen = Screen()
    slide = screen.slide()
    slide.show()
    assert wait_for(lambda: shown(screen))
    thread = slide._thread
    slide.close()
    assert thread is not None
    thread.join(2)
    assert not thread.is_alive()
    slide.hide()
    time.sleep(0.05)
    assert "vanish" not in screen.calls


# --- Связка с окнами pywebview ------------------------------------------------------


class Handlers(list[Any]):
    def __iadd__(self, handler: Any) -> Handlers:  # type: ignore[override]
        self.append(handler)
        return self


class FakeWindow:
    """Окно pywebview: дескриптор и hide."""

    def __init__(self) -> None:
        self.events = SimpleNamespace(before_show=Handlers(), closing=Handlers())
        self.native = SimpleNamespace(Handle=SimpleNamespace(ToInt64=lambda: 42))
        self.hidden = 0

    def hide(self) -> None:
        self.hidden += 1


@pytest.fixture
def desktop(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ...]]:
    """WinAPI оверлея: монитор 1920×1080 с панелью задач 40 пикселей, масштаб 150 %."""
    calls: list[tuple[Any, ...]] = []

    size = [(720, 96)]  # окно создано плашкой

    def record(name: str) -> Callable[..., None]:
        return lambda *args, **options: calls.append((name, *args, *options.values()))

    def set_bounds(hwnd: int, x: int, y: int, width: int, height: int) -> None:
        calls.append(("bounds", hwnd, x, y, width, height))
        size[0] = (width, height)

    monkeypatch.setattr(win32, "make_overlay", record("make_overlay"))
    monkeypatch.setattr(win32, "remove_backdrop", record("no_backdrop"))
    monkeypatch.setattr(win32, "round_corners", record("corners"))
    monkeypatch.setattr(win32, "set_bounds", set_bounds)
    monkeypatch.setattr(win32, "show_without_focus", record("show"))
    monkeypatch.setattr(win32, "move_window", record("move"))
    monkeypatch.setattr(win32, "clip_window", record("clip"))
    monkeypatch.setattr(win32, "shape_window", record("shape"))
    monkeypatch.setattr(win32, "work_area", lambda: (0, 0, 1920, 1040))
    monkeypatch.setattr(win32, "dpi_scale", lambda hwnd: 1.5)
    monkeypatch.setattr(win32, "window_size", lambda hwnd: size[0])
    monkeypatch.setattr(win32, "rounds_corners", lambda: True)
    monkeypatch.setattr(win32, "wait_frame", lambda: time.sleep(0.004))
    return calls


def make_windows(animate: bool, overlay_style: str = "classic") -> tuple[Any, FakeWindow]:
    from assistant.ui.windows import Windows

    windows: dict[str, FakeWindow] = {}

    def create_window(title: str, url: str, **options: Any) -> FakeWindow:
        windows[url.split("view=")[1].split("&")[0]] = window = FakeWindow()
        return window

    webview = SimpleNamespace(create_window=create_window)
    urls = {name: f"index.html?view={name}&theme=dark" for name in ("main", "overlay", "chat")}
    apis = dict.fromkeys(urls)
    host = Windows(webview, urls, apis, "dark", False, overlay_style, animate=lambda: animate)
    return host, windows["overlay"]


def test_overlay_window_gets_no_backdrop_before_its_frame_changes(desktop: list[tuple[Any, ...]]) -> None:
    # pywebview ставит Mica в тёмной теме Windows уже при создании окна; у оверлея её снимаем до
    # make_overlay — Windows применяет подложку при смене рамки окна.
    host, overlay = make_windows(animate=False)
    for handler in overlay.events.before_show:
        handler(overlay)
    assert desktop == [("corners", 42), ("no_backdrop", 42), ("make_overlay", 42)]
    host.quit()


def test_overlay_appears_behind_taskbar_edge_and_takes_its_place(desktop: list[tuple[Any, ...]]) -> None:
    host, overlay = make_windows(animate=False)
    host.show("overlay")
    assert wait_for(lambda: ("move", 42, 600, 908) in desktop)
    # Отступ 24 и радиус 8 — логические пиксели, при 150 % это 36 и 12.
    assert desktop == [
        ("no_backdrop", 42),  # подложку Mica pywebview мог поставить снова
        ("make_overlay", 42),
        ("clip", 42, 720, 0, 12),  # до показа: не видно ничего
        ("show", 42, 600, 1040),  # верх окна на краю рабочей области
        ("move", 42, 600, 908),
        ("clip", 42, 720, None, 12),  # на месте окно видно целиком
    ]
    assert host.visible("overlay")
    host.hide("overlay")
    assert not host.visible("overlay")
    assert wait_for(lambda: overlay.hidden == 1)
    host.quit()


def test_overlay_slides_in_and_out(desktop: list[tuple[Any, ...]]) -> None:
    host, overlay = make_windows(animate=True)
    host.show("overlay")
    assert wait_for(lambda: ("move", 42, 600, 908) in desktop)
    moves = [call[3] for call in desktop if call[0] == "move"]
    clips = [call[3] for call in desktop if call[0] == "clip"]
    assert len(moves) >= 5 and all(a > b for a, b in pairwise(moves))
    assert clips[0] == 0 and clips[-1] is None and clips.count(None) == 1
    visible = [height for height in clips if height is not None]
    assert all(a < b for a, b in pairwise(visible)) and visible[-1] < 96
    desktop.clear()
    host.hide("overlay")
    assert wait_for(lambda: overlay.hidden == 1)
    moves = [call[3] for call in desktop if call[0] == "move"]
    clips = [call[3] for call in desktop if call[0] == "clip"]
    assert len(moves) >= 4 and all(a < b for a, b in pairwise(moves)) and moves[-1] == 1040
    assert clips[-1] == 0, "уехав за край, окно не видно ещё до того, как спрятано"
    host.quit()


def test_orb_window_is_shaped_like_orb_and_card_and_slides_in(desktop: list[tuple[Any, ...]]) -> None:
    from assistant.ui.windows import ORB_SHAPE, scaled_shape

    shape = scaled_shape(ORB_SHAPE, 1.5)
    host, overlay = make_windows(animate=True, overlay_style="orb")
    host.show("overlay")
    assert wait_for(lambda: ("move", 42, 630, 374) in desktop)
    # Сфера 440×420 при 150 % — 660×630; окно — только круг шара и карточка, выезжает снизу.
    assert desktop[:6] == [
        ("no_backdrop", 42),  # подложку Mica pywebview мог поставить снова
        ("make_overlay", 42),
        ("corners", 42, False),  # прямые углы без рамки: окно и так по форме шара
        ("bounds", 42, 630, 374, 660, 630),
        ("shape", 42, shape, 0),  # до показа: не видно ничего
        ("show", 42, 630, 1040),
    ]
    heights = [call[3] for call in desktop if call[0] == "shape"]
    assert heights[-1] is None and heights.count(None) == 1, "на месте — форма целиком, один раз"
    visible = [height for height in heights if height is not None]
    assert all(a < b for a, b in pairwise(visible)) and visible[-1] < 630
    assert all(call[2] == shape for call in desktop if call[0] == "shape")
    assert not [call for call in desktop if call[0] == "clip"], "у сферы нет прямоугольной обрезки"
    desktop.clear()
    host.hide("overlay")
    assert wait_for(lambda: overlay.hidden == 1)
    heights = [call[3] for call in desktop if call[0] == "shape"]
    assert heights[-1] == 0 and all(a > b for a, b in pairwise(heights)), "уезжая, видна всё меньшая часть"
    host.quit()


def test_classic_after_orb_gets_its_window_back(desktop: list[tuple[Any, ...]]) -> None:
    host, overlay = make_windows(animate=False, overlay_style="orb")
    host.show("overlay")
    assert wait_for(lambda: any(call[0] == "move" for call in desktop))
    host.hide("overlay")
    assert wait_for(lambda: overlay.hidden == 1)
    desktop.clear()
    host.set_overlay_style("classic")
    host.set_overlay_style("neon")  # неизвестное оформление не принимается
    host.show("overlay")
    assert wait_for(lambda: ("move", 42, 600, 908) in desktop)
    assert desktop == [
        ("no_backdrop", 42),  # подложку Mica pywebview мог поставить снова
        ("make_overlay", 42),
        ("corners", 42, True),
        ("bounds", 42, 600, 908, 720, 96),
        ("clip", 42, 720, 0, 12),
        ("show", 42, 600, 1040),
        ("move", 42, 600, 908),
        ("clip", 42, 720, None, 12),
    ]
    host.quit()


def test_orb_shape_scales_by_edges() -> None:
    from assistant.ui.windows import ORB_SHAPE, scaled_shape

    assert scaled_shape(ORB_SHAPE, 1) == ORB_SHAPE
    # Шар 378 в (31, 17) при 200 % — 756 физических пикселей от 62 до 817 включительно.
    assert scaled_shape(ORB_SHAPE, 2)[0] == (62, 34, 817, 789, 378)


def test_orb_window_shape_matches_page_layout() -> None:
    """Форма окна сферы совпадает с раскладкой страницы (ui/src/index.css): шар и карточка в ней."""
    from assistant.ui.windows import ORB_SHAPE, OVERLAY_SIZES

    css = (Path(__file__).resolve().parents[1] / "ui" / "src" / "index.css").read_text(encoding="utf-8")

    def rule(selector: str) -> dict[str, int]:
        body = re.search(re.escape(selector) + r" \{([^}]*)\}", css)
        assert body, selector
        return {key: int(value) for key, value in re.findall(r"([a-z-]+): (-?\d+)px;", body.group(1))}

    stage, orb, card = rule(".orb-stage"), rule(".orb"), rule(".orb-card")
    assert (stage["width"], stage["height"]) == OVERLAY_SIZES["orb"]
    assert orb["width"] == orb["height"] and orb["left"] * 2 + orb["width"] == stage["width"]
    assert card["left"] * 2 + card["width"] == stage["width"], "карточка по центру окна"
    # Точно по краю шара и карточки: правая и нижняя границы — включительно.
    disc = (orb["left"], orb["top"], orb["left"] + orb["width"] - 1, orb["top"] + orb["height"] - 1)
    box = (card["left"], card["top"], card["left"] + card["width"] - 1, card["top"] + card["height"] - 1)
    assert ((*disc, orb["width"] // 2), (*box, card["border-radius"])) == ORB_SHAPE
