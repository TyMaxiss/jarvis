"""Окна интерфейса на настоящей Windows: pywebview + WebView2, оверлей без фокуса, быстрый чат.

Запускаются только при JARVIS_DESKTOP_TESTS=1 (так в CI): создают окна на рабочем столе.
Нужен собранный интерфейс (ui/dist) и Microsoft Edge WebView2 Runtime.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import shutil
import sys
import threading
import time
from collections.abc import Iterator
from ctypes import wintypes
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
from assistant.ui.controller import WINDOWS, UiController
from assistant.ui.events import EventBus
from assistant.ui.settings import SettingsStore
from tests.conftest import NO_WINDOW, ROOT
from tests.ui_fakes import FakeAgent, FakeVolume

pytestmark = [
    pytest.mark.windows,
    pytest.mark.desktop,
    pytest.mark.skipif(sys.platform != "win32", reason="нужна Windows"),
    pytest.mark.skipif(
        os.environ.get("JARVIS_DESKTOP_TESTS") != "1", reason="включаются через JARVIS_DESKTOP_TESTS=1"
    ),
]


def wait(condition: Any, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


def follow(hwnd: int, seconds: float) -> list[tuple[int, int]]:
    """Путь видимого окна: верх окна и тип его региона (0 — региона нет, 1 — пустой, 2–3 — обрезан)."""
    from assistant.actions import winapi
    from assistant.ui import win32

    user32 = ctypes.WinDLL("user32")
    user32.GetWindowRgn.restype = ctypes.c_int
    user32.GetWindowRgn.argtypes = (wintypes.HWND, wintypes.HRGN)
    api = win32._api()
    assert api is not None
    path: list[tuple[int, int]] = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if winapi.api().IsWindowVisible(hwnd):
            region = api.CreateRectRgn(0, 0, 0, 0)
            sample = (win32.window_rect(hwnd)[1], user32.GetWindowRgn(hwnd, region))
            api.DeleteObject(region)
            if not path or path[-1] != sample:
                path.append(sample)
        time.sleep(0.002)
    return path


WS_POPUP, SWP_SHOWWINDOW, PM_REMOVE = 0x80000000, 0x40, 0x1
WS_EX_TOOLWINDOW, WS_EX_NOACTIVATE = 0x80, 0x08000000
MAGENTA = (255, 0, 255)


class Backdrop:
    """Пурпурное окно в своём потоке: где у сферы нет окна (или окно прозрачно), виден этот цвет.

    Такого цвета нет ни в оформлении, ни в подложках Windows — его не спутать с фоном окна.
    """

    def __init__(self, x: int, y: int, width: int, height: int) -> None:
        self._stop = threading.Event()
        ready = threading.Event()
        threading.Thread(target=self._run, args=(x, y, width, height, ready), daemon=True).start()
        assert ready.wait(5), "пурпурное окно не создалось"

    def _run(self, x: int, y: int, width: int, height: int, ready: threading.Event) -> None:
        from assistant.ui import win32

        procedure_type = ctypes.WINFUNCTYPE(  # type: ignore[attr-defined]
            ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
        )

        class WNDCLASSEXW(ctypes.Structure):
            _fields_ = (
                ("cbSize", wintypes.UINT), ("style", wintypes.UINT), ("lpfnWndProc", procedure_type),
                ("cbClsExtra", ctypes.c_int), ("cbWndExtra", ctypes.c_int), ("hInstance", wintypes.HINSTANCE),
                ("hIcon", wintypes.HICON), ("hCursor", wintypes.HANDLE), ("hbrBackground", wintypes.HBRUSH),
                ("lpszMenuName", wintypes.LPCWSTR), ("lpszClassName", wintypes.LPCWSTR),
                ("hIconSm", wintypes.HICON),
            )  # fmt: skip

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        gdi32, kernel32 = ctypes.WinDLL("gdi32"), ctypes.WinDLL("kernel32")
        user32.DefWindowProcW.restype = ctypes.c_ssize_t
        user32.DefWindowProcW.argtypes = (wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
        user32.RegisterClassExW.argtypes = (ctypes.POINTER(WNDCLASSEXW),)
        user32.CreateWindowExW.restype = wintypes.HWND
        user32.CreateWindowExW.argtypes = (
            wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID,
        )  # fmt: skip
        gdi32.CreateSolidBrush.restype = wintypes.HBRUSH
        kernel32.GetModuleHandleW.restype = wintypes.HMODULE
        # Окно рисует стандартная процедура: фон — кистью класса, Python-обработчик не нужен.
        procedure = procedure_type(ctypes.cast(user32.DefWindowProcW, ctypes.c_void_p).value)
        instance = kernel32.GetModuleHandleW(None)
        red, green, blue = MAGENTA
        brush = gdi32.CreateSolidBrush(red | green << 8 | blue << 16)
        name = "JarvisTestBackdrop"
        window_class = WNDCLASSEXW(
            ctypes.sizeof(WNDCLASSEXW), 0, procedure, 0, 0, instance, None, None, brush, None, name, None
        )
        user32.RegisterClassExW(ctypes.byref(window_class))
        hwnd = user32.CreateWindowExW(
            WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE, name, "", WS_POPUP, x, y, width, height,
            None, None, instance, None,
        )  # fmt: skip
        api = win32._api()
        assert api is not None and hwnd, f"CreateWindowExW: ошибка {ctypes.get_last_error()}"
        # Поверх обычных окон (но под оверлеем, он поверх всех) и без активации.
        api.SetWindowPos(hwnd, 0, x, y, width, height, win32.SWP_NOACTIVATE | SWP_SHOWWINDOW)
        ready.set()
        message = wintypes.MSG()
        while not self._stop.is_set():
            while user32.PeekMessageW(ctypes.byref(message), None, 0, 0, PM_REMOVE):
                user32.TranslateMessage(ctypes.byref(message))
                user32.DispatchMessageW(ctypes.byref(message))
            time.sleep(0.01)
        user32.DestroyWindow(hwnd)

    def close(self) -> None:
        self._stop.set()


def orb_pixels(
    rect: tuple[int, int, int, int], scale: float, name: str = "overlay-orb"
) -> dict[str, tuple[int, ...]]:
    """Снимок окна сферы с экрана (в артефакт CI, если задан JARVIS_SCREENSHOTS) и цвета в его точках."""
    from PIL import ImageGrab

    x, y, width, height = rect
    shot = ImageGrab.grab(bbox=(x, y, x + width, y + height), include_layered_windows=True)
    folder = os.environ.get("JARVIS_SCREENSHOTS")
    if folder:
        Path(folder).mkdir(parents=True, exist_ok=True)
        shot.save(Path(folder) / f"{name}.png")

    def at(left: int, top: int) -> tuple[int, ...]:
        return tuple(shot.getpixel((round(left * scale), round(top * scale))))[:3]

    # Логические координаты в окне 440×420 (ui/src/index.css): шар 378 в (31, 17), карточка
    # 358×126 в (41, 266), кружок микрофона — центр в (70, 293), радиус 13.
    return {
        "corner": at(6, 6),  # мимо формы окна
        "beside": at(14, 206),  # слева от круга
        "ring": at(30, 206),  # в пикселе от края шара — уже мимо формы окна
        "disc": at(40, 206),  # внешний диск шара
        "orb": at(220, 150),  # середина шара
        "badge": at(61, 293),  # кружок микрофона слева от значка
        "card": at(60, 372),  # карточка ниже круга шара
    }


def near(color: tuple[int, ...], expected: tuple[int, int, int], tolerance: int = 14) -> bool:
    return all(abs(a - b) <= tolerance for a, b in zip(color, expected, strict=True))


def blend(color: tuple[int, int, int], alpha: float) -> tuple[int, int, int]:
    """Цвет с непрозрачностью alpha поверх пурпурного окна (как его смешает Windows)."""
    r, g, b = (round(alpha * top + (1 - alpha) * under) for top, under in zip(color, MAGENTA, strict=True))
    return r, g, b


# Полупрозрачные кольцо и карточка сферы поверх пурпурного (ui/src/index.css, .orb-glass).
GLASS_DISC = blend((26, 50, 104), 0.6)
GLASS_CARD = blend((22, 24, 28), 0.8)


DWMWA_SYSTEMBACKDROP_TYPE, DWMSBT_NONE, DWMSBT_MAINWINDOW = 38, 1, 2  # подложка окна: нет, Mica
HWND_BROADCAST, WM_SETTINGCHANGE, SMTO_ABORTIFHUNG = 0xFFFF, 0x001A, 0x0002


def backdrop_type(hwnd: int) -> int | None:
    """Какая у окна подложка Windows (DWMWA_SYSTEMBACKDROP_TYPE); None — Windows не сказала."""
    value = ctypes.c_int()
    result = ctypes.WinDLL("dwmapi").DwmGetWindowAttribute(
        wintypes.HWND(hwnd), DWMWA_SYSTEMBACKDROP_TYPE, ctypes.byref(value), ctypes.sizeof(value)
    )
    return value.value if result == 0 else None


def set_backdrop_type(hwnd: int, kind: int) -> None:
    value = ctypes.c_int(kind)
    ctypes.WinDLL("dwmapi").DwmSetWindowAttribute(
        wintypes.HWND(hwnd), DWMWA_SYSTEMBACKDROP_TYPE, ctypes.byref(value), ctypes.sizeof(value)
    )


def broadcast_settings_change() -> None:
    """Как при смене настроек Windows: WM_SETTINGCHANGE всем окнам (SystemEvents.UserPreferenceChanged)."""
    user32 = ctypes.WinDLL("user32")
    user32.SendMessageTimeoutW.argtypes = (
        wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPCWSTR, wintypes.UINT, wintypes.UINT,
        ctypes.POINTER(ctypes.c_size_t),
    )  # fmt: skip
    answer = ctypes.c_size_t()
    user32.SendMessageTimeoutW(
        HWND_BROADCAST, WM_SETTINGCHANGE, 0, "ImmersiveColorSet", SMTO_ABORTIFHUNG, 3000, ctypes.byref(answer)
    )


class Personalize:
    """Оформление Windows (HKCU\\…\\Themes\\Personalize) на время теста; restore возвращает прежнее."""

    def __init__(self) -> None:
        self._before: dict[str, int | None] = {}

    def set(self, name: str, value: int) -> None:
        import winreg

        from assistant.ui.win32 import PERSONALIZE_KEY

        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, PERSONALIZE_KEY) as key:
            if name not in self._before:
                try:
                    self._before[name] = winreg.QueryValueEx(key, name)[0]
                except FileNotFoundError:
                    self._before[name] = None
            winreg.SetValueEx(key, name, 0, winreg.REG_DWORD, value)

    def restore(self) -> None:
        import winreg

        from assistant.ui.win32 import PERSONALIZE_KEY

        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, PERSONALIZE_KEY) as key:
            for name, before in self._before.items():
                if before is None:
                    with contextlib.suppress(FileNotFoundError):
                        winreg.DeleteValue(key, name)
                else:
                    winreg.SetValueEx(key, name, 0, winreg.REG_DWORD, before)


@pytest.fixture
def personalize() -> Iterator[Personalize]:
    """Как у пользователя: тёмная тема приложений (pywebview тогда ставит окнам подложку Mica)
    и включённые «Эффекты прозрачности» (сфера полупрозрачная). Тест может их менять."""
    look = Personalize()
    try:
        look.set("AppsUseLightTheme", 0)
        look.set("EnableTransparency", 1)
        yield look
    finally:
        look.restore()


def test_windows_load_interface_and_talk_to_python(
    tmp_path: Path,
    registry: ActionRegistry,
    store: CommandStore,
    matcher: Matcher,
    capsys: pytest.CaptureFixture[str],
    personalize: Personalize,
) -> None:
    import webview

    from assistant.actions import winapi
    from assistant.ui import win32
    from assistant.ui.windows import UI_DIST, Windows, ui_urls

    if not (UI_DIST / "index.html").exists():
        pytest.fail("интерфейс не собран: нет ui/dist/index.html")
    version = win32.webview2_version()
    assert version, "нет Microsoft Edge WebView2 Runtime"

    config_path = tmp_path / "config.yaml"
    shutil.copyfile(ROOT / "config.example.yaml", config_path)
    settings = SettingsStore(config_path, tmp_path / ".env")
    bus = EventBus()
    activity = Activity(bus)
    executor = ActionExecutor(
        registry,
        SafetyPolicy(),
        StaticConfirmer(True),
        ActionJournal(tmp_path / "j.jsonl", SecretMasker()),
        dry_run=True,
    )
    agent: Any = FakeAgent(settings.load().llm)
    pipeline = Pipeline(matcher, executor, UiSpeaker(activity), lambda: NO_WINDOW, agent=agent)
    controller = UiController(
        config=settings.load(), store=settings, pipeline=pipeline, executor=executor, commands=store,
        make_agent=FakeAgent, speech=None, volume=FakeVolume(), bus=bus, activity=activity,
        confirmer=UiConfirmer(bus, timeout=1), log_file=tmp_path / "a.log", journal_file=tmp_path / "j.jsonl",
        force_dry_run=True,
    )  # fmt: skip
    webview.settings["DRAG_REGION_DIRECT_TARGET_ONLY"] = True
    apis = {name: Api(controller, name) for name in WINDOWS}
    # Оверлей двигается, даже если на сервере CI анимации Windows выключены.
    windows = Windows(
        webview, ui_urls("dark", "#7c6cff"), apis, "dark", start_hidden=False, animate=lambda: True
    )
    controller.host = windows
    results: dict[str, Any] = {}

    def scenario() -> None:
        try:
            # Все три страницы загрузились и спросили состояние у Python.
            results["ready"] = wait(lambda: all(bus.attached(name) for name in WINDOWS), 60)
            main = windows._windows["main"]
            results["heading"] = main.evaluate_js("document.querySelector('h1')?.textContent")
            results["renderer"] = getattr(webview, "renderer", None)

            # Оверлей выезжает снизу из-за края рабочей области и не забирает фокус.
            before = winapi.api().GetForegroundWindow()
            controller.overlay.voice_state("listening")
            overlay = windows._handle("overlay")
            results["overlay_rise"] = follow(overlay, 0.6)
            results["overlay_visible"] = bool(winapi.api().IsWindowVisible(overlay))
            results["focus_kept"] = winapi.api().GetForegroundWindow() == before
            style = win32._api().GetWindowLongPtrW(overlay, win32.GWL_EXSTYLE)
            results["overlay_no_activate"] = bool(style & win32.WS_EX_NOACTIVATE)
            x, y, width, height = win32.window_rect(overlay)
            left, top, right, bottom = win32.work_area()
            results["overlay_bottom"] = (
                abs(left + (right - left - width) // 2 - x) <= 2 and bottom - (y + height) > 0
            )
            results["overlay_place"] = (y, bottom)
            controller.overlay.hide_now()
            results["overlay_fall"] = follow(overlay, 0.6)
            results["overlay_hidden"] = wait(lambda: not winapi.api().IsWindowVisible(overlay), 5)

            # Сфера: окно 440×420 — только круг шара и карточка; под ним пурпурное окно.
            controller.save_settings({"ui": {"overlay_style": "orb"}})
            scale = win32.dpi_scale(overlay)
            size = (int(440 * scale), int(420 * scale))
            ox, oy = win32.bottom_center(size, win32.work_area(), round(24 * scale))
            under = Backdrop(ox - 20, oy - 20, size[0] + 40, size[1] + 40)
            try:
                before = winapi.api().GetForegroundWindow()
                controller.overlay.voice_state("listening")
                results["orb_path"] = follow(overlay, 0.8)
                results["orb_rise"] = [y for y, _ in results["orb_path"]]
                time.sleep(0.8)  # внутренние слои распускаются
                results["orb_rect"] = win32.window_rect(overlay)
                results["orb_focus_kept"] = winapi.api().GetForegroundWindow() == before
                results["orb_pixels"] = orb_pixels(results["orb_rect"], scale)
                main_hwnd = windows._handle("main")
                results["backdrops"] = (backdrop_type(overlay), backdrop_type(main_hwnd))
                # Для сведения — так было у пользователя: подложка Mica, которую ставит pywebview.
                set_backdrop_type(overlay, DWMSBT_MAINWINDOW)
                time.sleep(0.4)
                results["mica_pixels"] = orb_pixels(results["orb_rect"], scale, "overlay-orb-mica")
                # Настройки Windows изменились: pywebview снова ставит Mica всем окнам (главное — снова
                # с Mica), а оверлей сразу после этого её снимает.
                set_backdrop_type(main_hwnd, DWMSBT_NONE)
                broadcast_settings_change()
                wait(lambda: backdrop_type(main_hwnd) == DWMSBT_MAINWINDOW, 5)
                time.sleep(0.4)
                results["backdrops_after_change"] = (backdrop_type(overlay), backdrop_type(main_hwnd))
                results["orb_pixels_after"] = orb_pixels(results["orb_rect"], scale, "overlay-orb-after")
                controller.overlay.hide_now()
                results["orb_hidden"] = wait(lambda: not winapi.api().IsWindowVisible(overlay), 5)
                # «Эффекты прозрачности» выключили: при следующем показе сфера непрозрачная.
                personalize.set("EnableTransparency", 0)
                time.sleep(0.6)  # уход доигран
                controller.overlay.voice_state("listening")
                wait(lambda: bool(winapi.api().IsWindowVisible(overlay)), 5)
                time.sleep(1.2)  # выехала, слои распустились
                results["opaque_pixels"] = orb_pixels(win32.window_rect(overlay), scale, "overlay-orb-opaque")
                controller.overlay.hide_now()
                wait(lambda: not winapi.api().IsWindowVisible(overlay), 5)
            finally:
                under.close()
            results["orb_size"] = size

            # Быстрый чат: окно показывается, команда из него проходит через js_api в конвейер.
            controller.toggle_chat()
            chat = windows._handle("chat")
            results["chat_visible"] = wait(lambda: bool(winapi.api().IsWindowVisible(chat)), 5)
            results["chat_foreground"] = winapi.api().GetForegroundWindow() == chat
            windows._windows["chat"].evaluate_js("window.pywebview.api.send_text('сделай громче')")
            results["chat_command"] = wait(lambda: bool(activity.recent), 10) and activity.recent[0]["phrase"]
            controller.toggle_chat()
            results["chat_hidden"] = wait(lambda: not winapi.api().IsWindowVisible(chat), 5)

            # Размер главного окна меняется уголком, «развернуть» — на рабочую область.
            controller.window_resize(1000, 700, 1)
            hwnd = windows._handle("main")
            results["resized"] = (
                wait(lambda: win32.window_rect(hwnd)[2] > 0, 2) and win32.window_rect(hwnd)[2:]
            )
            results["scale"] = win32.dpi_scale(hwnd)
            controller.window_action("main", "maximize")
            results["maximized"] = win32.window_rect(hwnd)[2:] == (right - left, bottom - top)
            controller.window_action("main", "maximize")
        except Exception as exc:
            results["error"] = repr(exc)
        finally:
            controller.quit()

    started = time.monotonic()
    finished = threading.Event()

    def watchdog() -> None:
        # Если окна так и не закрылись (WebView2 не запустился), тест не должен висеть до конца CI.
        if not finished.wait(180):
            print(f"Окна не закрылись за 180 с; результаты: {results}", flush=True)
            personalize.restore()  # os._exit не даст фикстуре вернуть тему
            os._exit(3)

    threading.Thread(target=watchdog, daemon=True).start()
    webview.start(scenario, gui="edgechromium")
    finished.set()
    bus.close()
    print(f"WebView2 {version}; результаты: {results}; {time.monotonic() - started:.1f} с")
    assert "error" not in results, results
    assert results["ready"], "страницы окон не загрузились"
    assert results["renderer"] in (None, "edgechromium")
    assert results["heading"] == "Голос выключен"
    assert results["overlay_visible"] and results["overlay_no_activate"] and results["overlay_bottom"]
    assert results["focus_kept"], "оверлей забрал фокус"
    assert results["overlay_hidden"]
    # Появление: только вверх, сначала низ обрезан краем, на месте региона нет.
    place, edge = results["overlay_place"]
    rise = [y for y, _ in results["overlay_rise"]]
    assert len(set(rise)) >= 3 and rise == sorted(rise, reverse=True), results["overlay_rise"]
    assert rise[0] > place and rise[-1] == place, results["overlay_rise"]
    assert any(kind for _, kind in results["overlay_rise"]), "низ оверлея не обрезался"
    assert results["overlay_rise"][-1][1] == 0, "на месте оверлей должен быть виден целиком"
    # Уход: только вниз до края рабочей области, низ обрезан.
    fall = [y for y, _ in results["overlay_fall"]]
    assert len(set(fall)) >= 3 and fall == sorted(fall), results["overlay_fall"]
    assert place <= fall[0] and fall[-1] <= edge, results["overlay_fall"]
    assert any(kind for _, kind in results["overlay_fall"]), "уходя, оверлей не обрезался"
    # Сфера: выехала снизу на место своего размера, без фокуса; вне формы окна — пурпурное окно под
    # ним, в форме — диск, синий шар, кружок микрофона и тёмная карточка там, где им положено быть.
    # С «Эффектами прозрачности» внешнее кольцо и карточка полупрозрачные: сквозь них — пурпурный.
    pixels = results["orb_pixels"]
    with capsys.disabled():
        print(f"\nЦвета сферы: {pixels}; путь: {results['orb_rise']}")
        print(f"Подложки (оверлей, главное): {results['backdrops']}, после смены настроек Windows: "
              f"{results['backdrops_after_change']}; с Mica у оверлея: {results['mica_pixels']}")  # fmt: skip
    # Тёмная тема: pywebview поставил главному окну Mica, у оверлея её нет — и после смены настроек.
    expected = (DWMSBT_NONE, DWMSBT_MAINWINDOW)
    assert results["backdrops"] == expected and results["backdrops_after_change"] == expected, results
    after = results["orb_pixels_after"]
    assert all(near(after[point], MAGENTA) for point in ("corner", "beside", "ring")), after
    assert near(after["disc"], GLASS_DISC), f"после смены настроек Windows кольцо не прозрачное: {after}"
    assert results["orb_hidden"] and results["orb_focus_kept"]
    assert tuple(results["orb_rect"][2:]) == results["orb_size"], results["orb_rect"]
    rise = results["orb_rise"]
    assert len(set(rise)) >= 3 and rise == sorted(rise, reverse=True), rise
    assert rise[-1] == results["orb_rect"][1], rise
    assert results["orb_path"][-1][1] == 3, f"на месте у окна сферы нет формы: {results['orb_path']}"
    assert all(near(pixels[point], MAGENTA) for point in ("corner", "beside", "ring")), (
        f"вокруг шара есть фон: {pixels}"
    )
    assert near(pixels["disc"], GLASS_DISC), f"внешнее кольцо шара не полупрозрачное: {pixels}"
    red, _, blue = pixels["orb"]
    assert blue >= 200 and blue - red >= 80, f"нет синего шара: {pixels}"
    assert near(pixels["badge"], (68, 134, 255), 30), f"карточка не на месте (нет кружка микрофона): {pixels}"
    assert near(pixels["card"], GLASS_CARD), f"карточка не полупрозрачная: {pixels}"
    # Без «Эффектов прозрачности» — как в 0.7.3: окно по форме, но кольцо и карточка непрозрачные.
    opaque = results["opaque_pixels"]
    with capsys.disabled():
        print(f"Без «Эффектов прозрачности»: {opaque}")
    assert all(near(opaque[point], MAGENTA) for point in ("corner", "beside", "ring")), opaque
    assert near(opaque["disc"], (30, 47, 81)), f"нет непрозрачного тёмно-синего диска: {opaque}"
    assert near(opaque["card"], (22, 24, 28), 8), f"нет непрозрачной тёмной карточки: {opaque}"
    assert results["chat_visible"] and results["chat_hidden"]
    assert results["chat_command"] == "сделай громче"
    width, height = results["resized"]
    assert (
        abs(width - round(1000 * results["scale"])) <= 2 and abs(height - round(700 * results["scale"])) <= 2
    )
    assert results["maximized"]


def test_system_hotkey_fires(tmp_path: Path) -> None:
    from assistant.actions import winapi
    from assistant.actions.input import parse_hotkey
    from assistant.ui.win32 import SystemHotkey

    pressed = threading.Event()
    hotkey = SystemHotkey("ctrl+alt+f11", pressed.set)
    hotkey.start()
    try:
        codes = parse_hotkey("ctrl+alt+f11")
        winapi.send_inputs([winapi.key_event(vk, up=False) for vk in codes])
        winapi.send_inputs([winapi.key_event(vk, up=True) for vk in reversed(codes)])
        assert pressed.wait(3), "RegisterHotKey не получил нажатие"
    finally:
        hotkey.close()
    # Сочетание снято — его можно зарегистрировать снова.
    again = SystemHotkey("ctrl+alt+f11", lambda: None)
    again.start()
    again.close()


def test_tray_icon_starts_and_stops() -> None:
    from assistant.ui.tray import Tray, TrayActions

    noop = lambda: None  # noqa: E731
    actions = TrayActions(noop, noop, noop, lambda: False, lambda: True, noop, noop, noop, noop)
    tray = Tray("Джарвис: проверка", "#7c6cff", actions)
    tray.start()
    time.sleep(1)
    tray.update("Джарвис: на паузе")
    tray.stop()
