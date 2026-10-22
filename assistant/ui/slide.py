"""Плавное появление и уход оверлея: панель выезжает снизу и уезжает вниз, сфера раскрывается
на своём месте.

Окно двигается кадр за кадром, а всё, что ниже края рабочей области (над панелью задач),
обрезается — панель будто выдвигается из-за панели задач. Неподвижный путь (Track.still) — окно
сразу на месте, а кадры только отсчитывают долю появления (сфера раскрывается из центра).
Кадры идут в своём потоке; доля считается по времени, поэтому длительность одна при любой
частоте монитора. Новая команда (показать, пока окно уходит, или спрятать, пока появляется)
разворачивает его с того места, где оно сейчас. Окно двигает вызывающий код (appear, place,
vanish); здесь только ход времени.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

log = logging.getLogger(__name__)

SHOW_TIME = 0.22  # с: появление, к концу замедляется
HIDE_TIME = 0.17  # с: уход, к концу ускоряется


@dataclass(frozen=True)
class Track:
    """Путь окна в физических пикселях экрана.

    Attributes:
        x: левый край окна.
        top: верх окна на месте.
        edge: край, из-за которого окно выезжает (низ рабочей области).
        width: ширина окна.
        height: высота окна.
        radius: скругление верхних углов, пока низ окна обрезан краем.
    """

    x: int
    top: int
    edge: int
    width: int
    height: int
    radius: int = 0

    @property
    def still(self) -> bool:
        """Окно не двигается: появляется на месте (сфера)."""
        return self.edge == self.top

    def y(self, shown: float) -> int:
        """Верх окна, когда оно проехало долю shown пути: 0 — за краем, 1 — на месте."""
        return round(self.edge + (self.top - self.edge) * shown)

    def visible(self, y: int) -> int | None:
        """Сколько пикселей окна над краем, когда его верх в y; None — окно видно целиком."""
        above = self.edge - y
        return None if above >= self.height else max(0, above)


def ease_out(t: float) -> float:
    """Быстро в начале и плавно к концу — так окно появляется."""
    return 1 - (1 - t) ** 3


def ease_in(t: float) -> float:
    """Плавно в начале и быстро к концу — так окно уходит."""
    return t**3


class Slide:
    """Показывает и прячет окно, двигая его в своём потоке.

    show и hide возвращаются сразу; окно трогает только поток анимации, по очереди.

    Args:
        appear: поставить окно в начало пути (его не видно) и показать без фокуса; возвращает путь.
        place: кадр — окно проехало долю пути shown (0 — не видно, 1 — на месте).
        vanish: спрятать окно (его уже не видно).
        animate: включены ли анимации; нет — окно сразу на месте или сразу спрятано.
        wait_frame: дождаться следующего кадра экрана.
        clock: часы в секундах (в тестах — подделка).
        show_time: длительность появления, с.
        hide_time: длительность ухода, с.
    """

    def __init__(
        self,
        appear: Callable[[], Track],
        place: Callable[[Track, float], None],
        vanish: Callable[[], None],
        animate: Callable[[], bool],
        wait_frame: Callable[[], None],
        clock: Callable[[], float] = time.monotonic,
        show_time: float = SHOW_TIME,
        hide_time: float = HIDE_TIME,
    ) -> None:
        self._appear = appear
        self._place = place
        self._vanish = vanish
        self._animate = animate
        self._wait_frame = wait_frame
        self._clock = clock
        self._show_time = show_time
        self._hide_time = hide_time
        self._changed = threading.Condition()
        self._target = 0.0  # куда ехать: 1 — на место, 0 — за край и спрятать
        self._closed = False
        self._thread: threading.Thread | None = None
        # Дальше — только для потока анимации.
        self._track: Track | None = None  # путь показанного окна; None — окно спрятано
        self._shown = 0.0  # какую долю пути окно проехало

    def show(self) -> None:
        """Окно выезжает на место (уезжавшее — разворачивается)."""
        self._request(1.0)

    def hide(self) -> None:
        """Окно уезжает за край и прячется."""
        self._request(0.0)

    def close(self) -> None:
        """Останавливает поток анимации; его не ждём — он может стоять в вызове закрываемого окна."""
        with self._changed:
            self._closed = True
            self._changed.notify()

    def _request(self, target: float) -> None:
        with self._changed:
            if self._closed:
                return
            self._target = target
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="overlay-slide", daemon=True)
                self._thread.start()
            self._changed.notify()

    def _arrived(self, target: float) -> bool:
        """Окно уже там, куда его просят."""
        if target:
            return self._track is not None and self._shown >= 1
        return self._track is None

    def _run(self) -> None:
        while True:
            with self._changed:
                while not self._closed and self._arrived(self._target):
                    self._changed.wait()
                if self._closed:
                    return
                target = self._target
            try:
                self._go(target)
            except Exception:
                log.exception("Не удалось %s оверлей", "показать" if target else "спрятать")
                self._give_up(target)

    def _go(self, target: float) -> None:
        """Едет к target, пока не попросят другого; спрятанное окно сначала появляется за краем."""
        if self._track is None:
            self._track, self._shown = self._appear(), 0.0
        track, start = self._track, self._shown
        showing = target > start
        duration = (self._show_time if showing else self._hide_time) * abs(target - start)
        if duration > 0 and self._animate():
            ease = ease_out if showing else ease_in
            started, progress = self._clock(), 0.0
            while progress < 1:
                self._wait_frame()
                with self._changed:
                    if self._closed or self._target != target:
                        return  # закрыли или развернули: новый путь начнётся с этого места
                progress = min(1.0, (self._clock() - started) / duration)
                self._move(track, target if progress >= 1 else start + (target - start) * ease(progress))
        elif target:
            self._move(track, target)
        if not target:
            self._vanish()
            self._track = None

    def _move(self, track: Track, shown: float) -> None:
        """Ставит окно на долю пути shown; кадр, в котором окно не сдвинулось бы, пропускает
        (неподвижному окну нужен каждый кадр: оно раскрывается на месте)."""
        before, self._shown = self._shown, shown
        if track.still or track.y(shown) != track.y(before):
            self._place(track, shown)

    def _give_up(self, target: float) -> None:
        """После ошибки окно прячется, а просьба считается выполненной — не повторять её по кругу."""
        self._track, self._shown = None, 0.0
        try:
            self._vanish()
        except Exception:
            log.debug("Оверлей не спрятался", exc_info=True)
        with self._changed:
            if self._target == target:
                self._target = 0.0
