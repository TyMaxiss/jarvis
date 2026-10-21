# Сборка Jarvis (PyInstaller, onedir) — в Windows, из папки проекта:
#   uv sync --group build
#   uv run --no-sync pyinstaller --noconfirm --clean packaging/jarvis.spec
# Результат — dist/Jarvis: Jarvis.exe (окна, без консоли) и JarvisConsole.exe (консольный режим,
# --download-models) с общей папкой _internal; рядом — то, что правит пользователь.
import shutil
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = Path(SPECPATH).parent
APP = "Jarvis"
sys.path.insert(0, str(ROOT))

# Значок exe — тот же, что в трее (рисует Pillow).
from assistant.ui.tray import tray_image  # noqa: E402

ICON = Path(workpath) / "jarvis.ico"
ICON.parent.mkdir(parents=True, exist_ok=True)
tray_image("#7c6cff", 256).save(ICON, format="ICO", sizes=[(16, 16), (32, 32), (48, 48), (256, 256)])

# Обёртки COM для UI Automation и SAPI создаются сейчас и входят в сборку: иначе собранная
# программа создавала бы их при каждом запуске в %TEMP% (папка _internal не для записи).
if sys.platform == "win32":
    import comtypes.client

    comtypes.client.GetModule("UIAutomationCore.dll")
    comtypes.client.CreateObject("SAPI.SpVoice")

datas = [(str(ROOT / "ui" / "dist"), "ui/dist")]
datas += collect_data_files("faster_whisper")  # assets/silero_vad*.onnx
# Из Piper — только словари espeak-ng: модели иврита (hebrew) и арабского (tashkeel) не нужны.
datas += collect_data_files("piper", includes=["espeak-ng-data/**/*"])
# vosk и ctranslate2 загружают свои DLL сами (через cffi и при импорте), анализ их не видит.
binaries = collect_dynamic_libs("vosk") + collect_dynamic_libs("ctranslate2")
hiddenimports = collect_submodules("comtypes.gen") if sys.platform == "win32" else []

a = Analysis(
    [str(ROOT / "packaging" / "jarvis_entry.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=[
        "tkinter", "IPython", "matplotlib", "pytest", "PyQt5", "PyQt6", "PySide2", "PySide6", "qtpy",
        "gi", "cefpython3", "jnius", "torch", "transformers", "piper.train",
        # Xet для huggingface_hub не нужен: без метаданных пакета он в сборке и так не работает.
        "hf_xet",
    ],
)
pyz = PYZ(a.pure)
# Режим UTF-8 (как PYTHONUTF8=1): переменную окружения собранная программа не читает.
options = [("X utf8", None, "OPTION")]
common = dict(exclude_binaries=True, upx=False, strip=False, contents_directory="_internal", icon=str(ICON))
gui = EXE(pyz, a.scripts, options, name=APP, console=False, **common)
cli = EXE(pyz, a.scripts, options, name=f"{APP}Console", console=True, **common)
coll = COLLECT(gui, cli, a.binaries, a.datas, upx=False, strip=False, name=APP)

# Рядом с exe — файлы, которые правит пользователь. Своих данных пользователя (config.yaml,
# .env, commands/my.yaml, models/, logs/) в сборке нет: распакованная поверх новая версия их не затрёт.
out = Path(DISTPATH) / APP
for name in ("config.example.yaml", ".env.example", "README.md"):
    shutil.copy2(ROOT / name, out / name)
commands = out / "commands"
shutil.copytree(ROOT / "commands", commands, dirs_exist_ok=True, ignore=shutil.ignore_patterns("my.yaml"))
shutil.copy2(ROOT / "commands" / "my.yaml", commands / "my.yaml.example")
for exe in (APP, f"{APP}Console"):
    shutil.copy2(ROOT / "packaging" / "jarvis.exe.config", out / f"{exe}.exe.config")
