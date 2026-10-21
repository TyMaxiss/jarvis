"""Экран: что в активном окне (UI Automation), снимок для нейросети и нажатие элементов.

describe_screen перечисляет видимые элементы активного окна (тип, имя, где они) и делает
снимок экрана, на котором это окно. Если модель принимает картинки, агент прикладывает
снимок к разговору (Screen.take_image). Снимок уменьшается до 1280 px по ширине (JPEG),
а координаты элементов даются в пикселях снимка: click_screen переводит их обратно
в пиксели экрана. click_ui_element нажимает элемент активного окна по имени.

UI Automation работает через COM (comtypes, UIAutomationCore.dll), каждый запрос — в своём
потоке: зависшая программа или модальное окно, которое открыла нажатая кнопка, не
задерживают ассистента дольше таймаута. Координаты — физические пиксели: поток работает
с DPI-осведомлённостью per-monitor, иначе при масштабе 150 % щелчок уходил бы мимо.
Если активно окно самого ассистента (быстрый чат), работа идёт с окном под ним.
"""

from __future__ import annotations

import ctypes
import io
import logging
import re
import threading
import time
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Literal, Protocol

from pydantic import Field

from assistant.actions import winapi
from assistant.actions.registry import ActionError, ActionRegistry
from assistant.nlu.names import name_score

if TYPE_CHECKING:
    from PIL import Image

log = logging.getLogger(__name__)

SHOT_WIDTH = 1280  # снимок для нейросети — не шире, пикселей
SHOT_QUALITY = 80  # качество JPEG
MAX_ELEMENTS = 150  # сколько элементов перечислять в описании
MATCH_THRESHOLD = 75.0  # насколько имя элемента должно походить на запрошенное
_NAME_LIMIT = 80  # длиннее имя элемента обрезается

# Типы элементов UI Automation (UIA_*ControlTypeId) — названия, как в документации Microsoft.
CONTROL_TYPES = {
    50000: "Button", 50001: "Calendar", 50002: "CheckBox", 50003: "ComboBox", 50004: "Edit",
    50005: "Hyperlink", 50006: "Image", 50007: "ListItem", 50008: "List", 50009: "Menu",
    50010: "MenuBar", 50011: "MenuItem", 50012: "ProgressBar", 50013: "RadioButton",
    50014: "ScrollBar", 50015: "Slider", 50016: "Spinner", 50017: "StatusBar", 50018: "Tab",
    50019: "TabItem", 50020: "Text", 50021: "ToolBar", 50022: "ToolTip", 50023: "Tree",
    50024: "TreeItem", 50025: "Custom", 50026: "Group", 50027: "Thumb", 50028: "DataGrid",
    50029: "DataItem", 50030: "Document", 50031: "SplitButton", 50032: "Window", 50033: "Pane",
    50034: "Header", 50035: "HeaderItem", 50036: "Table", 50037: "TitleBar", 50038: "Separator",
    50039: "SemanticZoom", 50040: "AppBar",
}  # fmt: skip
# Элементы, которые нажимают: их показываем и без имени и предпочитаем при поиске.
INTERACTIVE = frozenset({
    "Button", "SplitButton", "Hyperlink", "MenuItem", "TabItem", "CheckBox", "RadioButton",
    "ListItem", "TreeItem", "DataItem", "ComboBox", "Edit", "HeaderItem", "Slider", "Spinner",
})  # fmt: skip
# Как назвать элемент в ответе: «Нажал кнопку «Сохранить»».
_SPOKEN = {
    "Button": "кнопку", "SplitButton": "кнопку", "Hyperlink": "ссылку", "MenuItem": "пункт меню",
    "TabItem": "вкладку", "CheckBox": "флажок", "RadioButton": "переключатель",
    "ListItem": "пункт списка", "TreeItem": "пункт", "DataItem": "строку", "Edit": "поле",
    "ComboBox": "список", "HeaderItem": "заголовок", "Image": "картинку", "Text": "текст",
}  # fmt: skip
# Чем закончилось нажатие (способ → глагол).
_DONE = {"invoke": "Нажал", "toggle": "Переключил", "select": "Выбрал", "expand": "Раскрыл",
         "collapse": "Свернул", "click": "Нажал"}  # fmt: skip


@dataclass(frozen=True)
class Rect:
    """Прямоугольник в пикселях (правая и нижняя границы не входят)."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def center(self) -> tuple[int, int]:
        return (self.left + self.right) // 2, (self.top + self.bottom) // 2

    @property
    def empty(self) -> bool:
        return self.width <= 0 or self.height <= 0


@dataclass(frozen=True)
class Element:
    """Видимый элемент окна по данным UI Automation."""

    kind: str  # тип: Button, Edit…
    name: str
    rect: Rect  # физические пиксели экрана
    enabled: bool = True

    def spoken(self) -> str:
        """«кнопку «Сохранить»» — для ответа пользователю."""
        what = _SPOKEN.get(self.kind, "элемент")
        return f"{what} «{self.name}»" if self.name else what


@dataclass(frozen=True)
class Shot:
    """Снимок экрана для нейросети: какую часть экрана он показывает и как уменьшен."""

    monitor: Rect  # снятый монитор, физические пиксели
    width: int  # размер снимка
    height: int

    @property
    def scale(self) -> float:
        """Сколько пикселей экрана в пикселе снимка (≥ 1)."""
        return self.monitor.width / self.width

    def to_shot(self, x: int, y: int) -> tuple[int, int]:
        """Точка экрана → точка снимка (в его пределах)."""
        shot_x = round((x - self.monitor.left) / self.scale)
        shot_y = round((y - self.monitor.top) / self.scale)
        return min(max(shot_x, 0), self.width - 1), min(max(shot_y, 0), self.height - 1)

    def to_screen(self, x: int, y: int) -> tuple[int, int]:
        """Точка снимка → точка экрана (обратный пересчёт для щелчка, в пределах монитора)."""
        screen_x = self.monitor.left + round(x * self.scale)
        screen_y = self.monitor.top + round(y * self.scale)
        return min(screen_x, self.monitor.right - 1), min(screen_y, self.monitor.bottom - 1)


def fit_width(width: int, height: int, limit: int = SHOT_WIDTH) -> tuple[int, int]:
    """Размер снимка: не шире limit, пропорции сохраняются (меньше — не увеличивается)."""
    if width <= limit:
        return width, height
    return limit, max(1, round(height * limit / width))


def clean_name(name: str) -> str:
    """Имя элемента в одну строку и не длиннее _NAME_LIMIT символов."""
    text = " ".join(str(name or "").split())
    return text if len(text) <= _NAME_LIMIT else text[: _NAME_LIMIT - 1] + "…"


def worth_listing(element: Element) -> bool:
    """Стоит ли показывать элемент модели: с именем или тот, что нажимают (поле без подписи).

    Контейнеры и картинки без имени ничего не говорят модели.
    """
    return not element.rect.empty and (bool(element.name) or element.kind in INTERACTIVE)


def screen_report(
    window: winapi.WindowInfo, shot: Shot, elements: list[Element], limit: int = MAX_ELEMENTS
) -> str:
    """Текст для модели: окно, масштаб снимка и элементы с координатами на снимке."""
    lines = []
    if window.hwnd:
        lines.append(f"Окно «{clean_name(window.title)}» ({window.process or 'программа неизвестна'}).")
    else:
        lines.append("Активного окна нет: виден рабочий стол.")
    lines.append(
        f"Координаты ниже — в пикселях снимка экрана {shot.width}×{shot.height} "
        f"(сам экран {shot.monitor.width}×{shot.monitor.height}): центр элемента и размер. "
        "Нажать элемент по имени — click_ui_element, точку снимка — click_screen."
    )
    shown = [e for e in elements if worth_listing(e)]
    if not shown:
        lines.append("Видимых элементов UI Automation в окне нет (программа их не сообщает).")
        return "\n".join(lines)
    lines.append("Элементы окна:")
    for element in shown[:limit]:
        x, y = shot.to_shot(*element.rect.center)
        width, height = round(element.rect.width / shot.scale), round(element.rect.height / shot.scale)
        name = f" «{element.name}»" if element.name else ""
        state = " (недоступен)" if not element.enabled else ""
        lines.append(f"- {element.kind}{name} ({x}, {y}) {width}×{height}{state}")
    if len(shown) > limit:
        lines.append(f"…и ещё {len(shown) - limit} (показаны первые {limit}).")
    return "\n".join(lines)


def _plain(text: str) -> str:
    """Имя для сравнения: без регистра, «ё», «&» (мнемоник) и многоточия."""
    text = text.lower().replace("ё", "е").replace("&", "")
    return " ".join(re.sub(r"[.…:]+$", "", text).split())


def element_score(query: str, element: Element) -> float:
    """Насколько элемент подходит под название (0–100)."""
    q, name = _plain(query), _plain(element.name)
    if not q or not name:
        return 0.0
    if q == name:
        return 100.0
    if name.startswith(q) or re.search(rf"\b{re.escape(q)}\b", name):
        return 90.0 - min(9.0, (len(name) - len(q)) / 4)  # «Сохранить» важнее «Сохранить как…»
    return name_score(q, name) * 0.9


# Слова «кнопку», «по ссылке» в начале названия: тип элемента, а не часть его имени.
_TYPE_WORDS = (
    ("кнопк", ("Button", "SplitButton")),
    ("ссылк", ("Hyperlink",)),
    ("вкладк", ("TabItem",)),
    ("пункт", ("MenuItem", "ListItem", "TreeItem")),
    ("меню", ("MenuItem",)),
    ("флажок", ("CheckBox",)),
    ("галочк", ("CheckBox",)),
    ("переключател", ("RadioButton",)),
    ("поле", ("Edit", "ComboBox")),
    ("значок", ("ListItem", "Button", "Image")),
    ("иконк", ("ListItem", "Button", "Image")),
)


def split_type_words(text: str) -> tuple[set[str], str]:
    """«ссылке войти» → ({Hyperlink}, «войти»); «пункт меню файл» → ({MenuItem}, «файл»)."""
    words = text.split()
    kinds: set[str] = set()
    while len(words) > 1:
        kind = next((k for prefix, k in _TYPE_WORDS if words[0].lower().startswith(prefix)), None)
        if kind is None:
            break
        kinds = set(kind) if not kinds else (kinds & set(kind)) or set(kind)
        words = words[1:]
    return kinds, " ".join(words)


def choose_element(elements: list[Element], text: str, kind: str = "", number: int = 1) -> Element:
    """Элемент по названию (и типу, если он указан); number — какой по счёту из равных.

    Если название начинается со слова-типа («кнопку сохранить», «по ссылке войти»), а целиком
    его в окне нет, ищется остаток среди элементов этого типа.

    Raises:
        ActionError: подходящего элемента нет (с похожими названиями, если они есть).
    """
    kinds = {kind.strip().lower()} if kind.strip() else set()
    attempts = [(text, kinds)]
    hinted, rest = split_type_words(text)
    if hinted:
        attempts.append((rest, kinds or {k.lower() for k in hinted}))
    similar: list[str] = []  # похожие названия — из поиска по названию целиком
    for query, allowed in attempts:
        candidates = [
            e for e in elements if e.name and not e.rect.empty and (not allowed or e.kind.lower() in allowed)
        ]
        scored = sorted(
            ((element_score(query, e), index, e) for index, e in enumerate(candidates)),
            # Лучшее совпадение; из равных — доступный и тот, что нажимают; дальше — по порядку в окне.
            key=lambda item: (-item[0], not item[2].enabled, item[2].kind not in INTERACTIVE, item[1]),
        )
        good = [item for item in scored if item[0] >= MATCH_THRESHOLD]
        if not good:
            similar = similar or [f"«{e.name}»" for score, _, e in scored[:3] if score >= 50]
            continue
        best = [item for item in good if item[0] == good[0][0]]
        if number > len(best):
            raise ActionError(f"Элементов «{query}» в окне {len(best)}, а нужен {number}-й")
        element = best[number - 1][2]
        if not element.enabled:
            raise ActionError(f"Не могу нажать {element.spoken()} — элемент сейчас неактивен")
        return element
    hint = f". Похожие: {', '.join(similar)}" if similar else ""
    raise ActionError(f"Не нашёл «{text}» в окне{hint}")


class Desktop(Protocol):
    """Откуда берутся окна, элементы и снимок (в тестах — подделка)."""

    def target_window(self) -> winapi.WindowInfo: ...

    def look(self, hwnd: int) -> tuple[Rect, Image.Image, list[Element]]: ...

    def press(self, hwnd: int, choose: Callable[[list[Element]], Element]) -> tuple[Element, str]: ...

    def click(self, x: int, y: int, button: str, double: bool) -> None: ...


class Screen:
    """Описание экрана для нейросети и нажатия; последний снимок ждёт, пока его заберёт агент.

    Args:
        desktop: настоящий экран Windows или подделка в тестах.
    """

    def __init__(self, desktop: Desktop | None = None) -> None:
        self._desktop = desktop or WindowsDesktop()
        self._lock = threading.Lock()
        self._shot: Shot | None = None
        self._image: bytes | None = None

    def describe(self) -> str:
        """Окно, элементы и снимок (снимок забирает агент через take_image)."""
        window = self._desktop.target_window()
        monitor, image, elements = self._desktop.look(window.hwnd)
        width, height = fit_width(image.width, image.height)
        if (width, height) != image.size:
            from PIL import Image

            image = image.resize((width, height), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, "JPEG", quality=SHOT_QUALITY, optimize=True)
        shot = Shot(monitor, width, height)
        with self._lock:
            self._shot, self._image = shot, buffer.getvalue()
        log.info("Экран: %s, элементов %d, снимок %d×%d (%d КБ)", window.title or "рабочий стол",
                 len(elements), width, height, buffer.tell() // 1024)  # fmt: skip
        return screen_report(window, shot, elements)

    def take_image(self) -> bytes | None:
        """Снимок последнего describe_screen (JPEG) — один раз; дальше None."""
        with self._lock:
            image, self._image = self._image, None
        return image

    def click_element(self, text: str, kind: str = "", number: int = 1) -> str:
        """Нажимает элемент активного окна по названию."""
        window = self._desktop.target_window()
        if not window.hwnd:
            raise ActionError("Нет активного окна")
        element, how = self._desktop.press(
            window.hwnd, lambda found: choose_element(found, text, kind, number)
        )
        return f"{_DONE.get(how, 'Нажал')} {element.spoken()}"

    def click_point(self, x: int, y: int, button: str = "left", double: bool = False) -> str:
        """Щелчок в точке последнего снимка (координаты пересчитываются в экранные)."""
        with self._lock:
            shot = self._shot
        if shot is None:
            raise ActionError("Сначала опишите экран (describe_screen): координаты берутся с его снимка")
        if not (0 <= x < shot.width and 0 <= y < shot.height):
            raise ActionError(f"Точка {x}, {y} за пределами снимка {shot.width}×{shot.height}")
        screen_x, screen_y = shot.to_screen(x, y)
        self._desktop.click(screen_x, screen_y, button, double)
        kind = "Двойной щелчок" if double else "Щелчок"
        return f"{kind} в точке {x}, {y} снимка (на экране {screen_x}, {screen_y})"


def register(registry: ActionRegistry, screen: Screen) -> None:
    """Регистрирует действия с экраном.

    Args:
        registry: реестр действий.
        screen: экран (общий с агентом: он забирает снимок для модели).
    """

    @registry.action(
        "describe_screen",
        "Посмотреть, что на экране: активное окно и его видимые элементы (тип, имя, координаты "
        "на снимке экрана); модели, которые понимают картинки, получают и сам снимок",
    )
    def describe_screen() -> str:
        return screen.describe()

    @registry.action(
        "click_ui_element",
        "Нажать кнопку, ссылку, пункт меню, вкладку или флажок в активном окне по названию "
        "(через UI Automation, без координат)",
    )
    def click_ui_element(
        text: Annotated[
            str, Field(min_length=1, max_length=200, description="Название элемента, как в окне")
        ],
        kind: Annotated[
            str, Field(max_length=40, description="Тип из describe_screen (Button, MenuItem…); пусто — любой")
        ] = "",
        number: Annotated[
            int, Field(ge=1, le=50, description="Какой по счёту, если одинаковых несколько")
        ] = 1,
    ) -> str:
        return screen.click_element(text, kind, number)

    @registry.action(
        "click_screen",
        "Щёлкнуть мышью точку на снимке из последнего describe_screen (координаты снимка); "
        "для элемента с названием лучше click_ui_element",
    )
    def click_screen(
        x: Annotated[int, Field(ge=0, le=10000, description="X на снимке, пиксели")],
        y: Annotated[int, Field(ge=0, le=10000, description="Y на снимке, пиксели")],
        button: Annotated[Literal["left", "right", "middle"], Field(description="Кнопка мыши")] = "left",
        double: Annotated[bool, Field(description="Двойной щелчок")] = False,
    ) -> str:
        return screen.click_point(x, y, button, double)


# --- Windows -------------------------------------------------------------------------

CLSID_CUIAUTOMATION = "{ff48dba4-60ef-4201-aa87-54103eef594e}"
CLSID_CUIAUTOMATION8 = "{e22ad333-b25f-460c-83d0-0581107395c9}"  # Windows 8+: с таймаутами
TREE_SCOPE_DESCENDANTS = 4
UIA_BOUNDING_RECTANGLE = 30001
UIA_CONTROL_TYPE = 30003
UIA_NAME = 30005
UIA_IS_ENABLED = 30010
UIA_IS_OFFSCREEN = 30022
UIA_INVOKE_PATTERN = 10000
UIA_EXPAND_COLLAPSE_PATTERN = 10005
UIA_SELECTION_ITEM_PATTERN = 10010
UIA_TOGGLE_PATTERN = 10015
EXPAND_STATE_COLLAPSED = 0
# Родной способ «нажать» элемент этого типа (остальные — Invoke).
_PREFERRED_PATTERN = {
    "CheckBox": UIA_TOGGLE_PATTERN,
    "RadioButton": UIA_SELECTION_ITEM_PATTERN,
    "TabItem": UIA_SELECTION_ITEM_PATTERN,
}
MONITOR_DEFAULTTONEAREST = 2
MONITOR_DEFAULTTOPRIMARY = 1
DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
_PRESS_GRACE = 3.0  # модальное окно после нажатия: столько ждём, потом считаем нажатым


class MONITORINFO(ctypes.Structure):
    _fields_ = (
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", wintypes.RECT),
        ("rcWork", wintypes.RECT),
        ("dwFlags", wintypes.DWORD),
    )


class WindowsDesktop:
    """Экран Windows: UI Automation (comtypes), снимок (Pillow), мышь (SendInput).

    Args:
        timeout: сколько ждать ответа окна, секунды.
        skip_own: окна ассистента и его консоли — не цель (в тестах цель — их собственные окна).
    """

    def __init__(self, timeout: float = 10.0, skip_own: bool = True) -> None:
        self._timeout = timeout
        self._skip_own = skip_own
        self._module_lock = threading.Lock()  # обёртку UIAutomationCore comtypes создаёт один раз

    def target_window(self) -> winapi.WindowInfo:
        """Активное окно; если активно окно ассистента (быстрый чат, консоль) — окно под ним."""
        own = winapi.own_process_tree() if self._skip_own else set()
        window = winapi.foreground_window()
        if window.hwnd and window.pid not in own:
            return window
        return next((w for w in winapi.list_windows() if w.pid not in own), winapi.EMPTY_WINDOW)

    def look(self, hwnd: int) -> tuple[Rect, Image.Image, list[Element]]:
        """Монитор окна, его снимок и видимые элементы окна."""

        def work(uia: _Uia) -> tuple[Rect, Image.Image, list[Element]]:
            from PIL import ImageGrab

            monitor = _monitor_of(hwnd)
            try:
                image = ImageGrab.grab(
                    bbox=(monitor.left, monitor.top, monitor.right, monitor.bottom), all_screens=True
                )
            except OSError as exc:  # например, экран заблокирован
                raise ActionError(f"Не удалось сделать снимок экрана: {exc}") from None
            elements = [element for element, _ in uia.elements(hwnd)] if hwnd else []
            return monitor, image, elements

        return self._call(work, "Окно не ответило вовремя: не удалось прочитать его элементы")

    def press(self, hwnd: int, choose: Callable[[list[Element]], Element]) -> tuple[Element, str]:
        """Находит элемент (choose выбирает из видимых) и нажимает его."""
        chosen: list[Element] = []
        pressing = threading.Event()

        def work(uia: _Uia) -> tuple[Element, str]:
            pairs = uia.elements(hwnd)
            element = choose([e for e, _ in pairs])
            chosen.append(element)
            native = next(item for e, item in pairs if e is element)
            return element, uia.press(native, element, hwnd, pressing)

        try:
            return self._call(work, "Окно не ответило вовремя", started=pressing)
        except _Late:
            # Кнопка открыла модальное окно: метод нажатия вернётся, только когда его закроют.
            if pressing.is_set() and chosen:
                log.info("Нажатие %s ещё не вернулось — считаю, что оно выполнено", chosen[0])
                return chosen[0], "invoke"
            raise

    def click(self, x: int, y: int, button: str, double: bool) -> None:
        """Щелчок мышью в точке экрана (физические пиксели)."""

        def work(uia: _Uia) -> None:
            _mouse_click(x, y, button, double)

        self._call(work, "Щелчок не выполнен вовремя", with_uia=False)

    def _call[T](
        self,
        work: Callable[[_Uia], T],
        late: str,
        with_uia: bool = True,
        started: threading.Event | None = None,
    ) -> T:
        """Выполняет work в своём потоке (COM, DPI per-monitor) и ждёт не дольше таймаута.

        started — нажатие начато: после него ждём не дольше _PRESS_GRACE (модальное окно).

        Raises:
            ActionError: ошибка UI Automation; _Late — поток не успел.
        """
        outcome: list[tuple[bool, Any]] = []

        def target() -> None:
            import comtypes

            _per_monitor_dpi()
            comtypes.CoInitialize()
            try:
                uia = _Uia(self._module_lock) if with_uia else _Uia.none()
                try:
                    outcome.append((True, work(uia)))
                finally:
                    uia.close()
            except ActionError as exc:
                outcome.append((False, ActionError(str(exc))))
            except Exception as exc:
                # Здесь же: объекты COM из исключения не должны пережить CoUninitialize.
                log.warning("UI Automation: %s", exc, exc_info=True)
                outcome.append(
                    (False, ActionError(f"UI Automation не смогла прочитать окно ({type(exc).__name__})"))
                )
            finally:
                comtypes.CoUninitialize()

        if not winapi.IS_WINDOWS:
            raise ActionError("Это действие работает только в Windows")
        thread = threading.Thread(target=target, name="uia", daemon=True)
        thread.start()
        deadline = time.monotonic() + self._timeout
        while thread.is_alive() and time.monotonic() < deadline:
            if started is not None and started.is_set():
                deadline = min(deadline, time.monotonic() + _PRESS_GRACE)
                started = None
            thread.join(0.05)
        if not outcome:
            raise _Late(late)
        ok, value = outcome[0]
        if not ok:
            raise value
        return value


class _Late(ActionError):
    """Окно не ответило за отведённое время."""


class _Uia:
    """UI Automation в текущем потоке (COM уже инициализирован)."""

    def __init__(self, module_lock: threading.Lock | None) -> None:
        self.module: Any = None
        self.automation: Any = None
        if module_lock is None:
            return
        import comtypes
        import comtypes.client

        with module_lock:
            self.module = comtypes.client.GetModule("UIAutomationCore.dll")
        try:
            self.automation = comtypes.client.CreateObject(
                CLSID_CUIAUTOMATION8, interface=self.module.IUIAutomation2
            )
            # Зависшая программа не держит запрос дольше нескольких секунд.
            self.automation.ConnectionTimeout = 2000
            self.automation.TransactionTimeout = 5000
        except (OSError, AttributeError, comtypes.COMError):
            self.automation = comtypes.client.CreateObject(
                CLSID_CUIAUTOMATION, interface=self.module.IUIAutomation
            )

    @classmethod
    def none(cls) -> _Uia:
        """Без UI Automation (для щелчка мышью)."""
        return cls(None)

    def close(self) -> None:
        """Отпускает объекты COM до CoUninitialize."""
        self.automation = None
        self.module = None

    def elements(self, hwnd: int) -> list[tuple[Element, Any]]:
        """Видимые элементы окна (представление «элементы управления») с их объектами COM."""
        automation = self.automation
        root = automation.ElementFromHandle(hwnd)
        cache = automation.CreateCacheRequest()
        for prop in (UIA_NAME, UIA_CONTROL_TYPE, UIA_BOUNDING_RECTANGLE, UIA_IS_ENABLED, UIA_IS_OFFSCREEN):
            cache.AddProperty(prop)
        condition = automation.CreateAndCondition(
            automation.ControlViewCondition, automation.CreatePropertyCondition(UIA_IS_OFFSCREEN, False)
        )
        found = root.FindAllBuildCache(TREE_SCOPE_DESCENDANTS, condition, cache)
        result = []
        for index in range(found.Length):
            item = found.GetElement(index)
            box = item.CachedBoundingRectangle
            element = Element(
                kind=CONTROL_TYPES.get(item.CachedControlType, "Custom"),
                name=clean_name(item.CachedName or ""),
                rect=Rect(box.left, box.top, box.right, box.bottom),
                enabled=bool(item.CachedIsEnabled),
            )
            result.append((element, item))
        return result

    def press(self, native: Any, element: Element, hwnd: int, pressing: threading.Event) -> str:
        """Нажимает элемент подходящим способом; pressing — нажатие начато.

        Обычно — Invoke; флажок переключается (Toggle), вкладка и переключатель выбираются
        (Select): их «нажатие» тоже работает, но так точнее и ответ правдивее.
        """
        module = self.module
        interfaces = {
            UIA_INVOKE_PATTERN: module.IUIAutomationInvokePattern,
            UIA_TOGGLE_PATTERN: module.IUIAutomationTogglePattern,
            UIA_SELECTION_ITEM_PATTERN: module.IUIAutomationSelectionItemPattern,
            UIA_EXPAND_COLLAPSE_PATTERN: module.IUIAutomationExpandCollapsePattern,
        }
        order = [
            UIA_INVOKE_PATTERN,
            UIA_TOGGLE_PATTERN,
            UIA_SELECTION_ITEM_PATTERN,
            UIA_EXPAND_COLLAPSE_PATTERN,
        ]
        preferred = _PREFERRED_PATTERN.get(element.kind)
        if preferred is not None:
            order.remove(preferred)
            order.insert(0, preferred)
        for pattern_id in order:
            interface = interfaces[pattern_id]
            unknown = native.GetCurrentPattern(pattern_id)
            if not unknown:
                continue
            pattern = unknown.QueryInterface(interface)
            pressing.set()
            if pattern_id == UIA_INVOKE_PATTERN:
                pattern.Invoke()
                return "invoke"
            if pattern_id == UIA_TOGGLE_PATTERN:
                pattern.Toggle()
                return "toggle"
            if pattern_id == UIA_SELECTION_ITEM_PATTERN:
                pattern.Select()
                return "select"
            if pattern.CurrentExpandCollapseState == EXPAND_STATE_COLLAPSED:
                pattern.Expand()
                return "expand"
            pattern.Collapse()
            return "collapse"
        # Ни одного способа UI Automation: щелчок мышью по середине элемента.
        if winapi.api().GetForegroundWindow() != hwnd:
            winapi.activate_window(hwnd)
        pressing.set()
        _mouse_click(*element.rect.center, "left", False)
        return "click"


def _per_monitor_dpi() -> None:
    """Поток видит физические пиксели (Windows 10 1703+; раньше — как есть)."""
    try:
        user32 = ctypes.WinDLL("user32")  # type: ignore[attr-defined]
        function = user32.SetThreadDpiAwarenessContext
    except (OSError, AttributeError):
        return
    function.restype = ctypes.c_void_p
    function.argtypes = (ctypes.c_void_p,)
    function(ctypes.c_void_p(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2))


def _monitor_of(hwnd: int) -> Rect:
    """Монитор, на котором окно (без окна — основной)."""
    user32 = ctypes.WinDLL("user32", use_last_error=True)  # type: ignore[attr-defined]
    user32.MonitorFromWindow.restype = wintypes.HMONITOR
    user32.MonitorFromWindow.argtypes = (wintypes.HWND, wintypes.DWORD)
    user32.GetMonitorInfoW.restype = wintypes.BOOL
    user32.GetMonitorInfoW.argtypes = (wintypes.HMONITOR, ctypes.POINTER(MONITORINFO))
    monitor = user32.MonitorFromWindow(
        hwnd or None, MONITOR_DEFAULTTONEAREST if hwnd else MONITOR_DEFAULTTOPRIMARY
    )
    info = MONITORINFO(cbSize=ctypes.sizeof(MONITORINFO))
    if not monitor or not user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
        raise ActionError("Не удалось узнать, на каком мониторе окно")
    box = info.rcMonitor
    return Rect(box.left, box.top, box.right, box.bottom)


def _mouse_click(x: int, y: int, button: str, double: bool) -> None:
    winapi.wait_modifiers_released()
    if not winapi.api().SetCursorPos(x, y):
        raise ActionError(f"Не удалось переместить курсор в точку {x}, {y}")
    winapi.send_inputs(winapi.mouse_button_events(button, double=double))
