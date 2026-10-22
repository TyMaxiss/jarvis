"""Скачивание того, что нужно голосу: модель Vosk (обращение по имени), голос Piper, модель
Whisper, библиотеки CUDA для Whisper на видеокарте NVIDIA и PyTorch с моделью для голоса Silero.

Недостающее скачивается само при запуске голоса; всё сразу — --download-models.
Скачанное повторно не загружается, а прерванная загрузка продолжается с того же места.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from assistant.config import AppConfig
from assistant.frozen import FROZEN, cuda_dir, silero_dir
from assistant.progress import Progress, megabytes
from assistant.stt.whisper_stt import (
    cuda_available,
    cuda_libraries_needed,
    missing_cuda_library,
    register_cuda_dlls,
    resolve_settings,
    whisper_download_progress,
)

log = logging.getLogger(__name__)

VOSK_BASE_URL = "https://alphacephei.com/vosk/models/"
PYPI_FILES = "https://files.pythonhosted.org/packages/"
INSTALLED = "wheels.txt"  # в папке cuda и silero/lib: какие колёса туда уже разложены
SILERO_MODEL_URL = "https://models.silero.ai/models/tts/ru/"
# Модели Silero, которые программа умеет скачать: имя файла → sha256 ('' — без проверки).
SILERO_MODELS = {"v5_6_ru.pt": ""}
_CHUNK = 1 << 20
_TIMEOUT = 60  # секунд ожидания ответа сервера


@dataclass(frozen=True)
class Wheel:
    """Колесо с PyPI, из которого берутся DLL: версия, sha256 и размер — как в uv.lock."""

    package: str
    version: str
    path: str  # адрес после PYPI_FILES
    sha256: str
    size: int

    @property
    def url(self) -> str:
        return PYPI_FILES + self.path

    @property
    def filename(self) -> str:
        return self.path.rsplit("/", 1)[-1]


# DLL, без которых CTranslate2 не запустит Whisper на видеокарте: cuBLAS 12 и cuDNN 9 — те же
# колёса, что ставит uv sync --extra cuda (тест сверяет их с uv.lock); NVRTC — зависимость cuBLAS.
CUDA_WHEELS = (
    Wheel(
        "nvidia-cublas-cu12",
        "12.9.2.10",
        "20/e2/fc9a0e985249d873150276d5afb02e39a66817fedbf1a385724393e505ed/"
        "nvidia_cublas_cu12-12.9.2.10-py3-none-win_amd64.whl",
        "623f43027d40d44ceadf0043f002bd25cf353e8f13ce90b9a87057019f560661",
        553_162_896,
    ),
    Wheel(
        "nvidia-cuda-nvrtc-cu12",
        "12.9.86",
        "52/de/823919be3b9d0ccbf1f784035423c5f18f4267fb0123558d58b813c6ec86/"
        "nvidia_cuda_nvrtc_cu12-12.9.86-py3-none-win_amd64.whl",
        "72972ebdcf504d69462d3bcd67e7b81edd25d0fb85a2c46d3ea3517666636349",
        76_408_187,
    ),
    Wheel(
        "nvidia-cudnn-cu12",
        "9.27.0.42",
        "aa/38/f856579877f7c1c5066e61182e7de7bc27bf35a78c8d1b0fa592e6985bc4/"
        "nvidia_cudnn_cu12-9.27.0.42-py3-none-win_amd64.whl",
        "06e9b0026f3bad97d2b58666330fabec04fe1672f776661ecb0ce0029c27f142",
        743_068_852,
    ),
)


# PyTorch для процесса синтеза Silero (CPU, только Windows) со всеми зависимостями — те же колёса,
# что ставит uv sync --extra silero (тест сверяет их с uv.lock).
SILERO_WHEELS = (
    Wheel(
        "torch",
        "2.14.1",
        "ad/1a/b9648b0d228e513f69f8453ec3f90a8d436d78ff6d85a2773e539f0c07cf/"
        "torch-2.14.1-cp312-cp312-win_amd64.whl",
        "38bee9f2a2ccfc6898172a5075e4ba52522fbe143eeb099674fc67ab390e858d",
        124_113_906,
    ),
    Wheel(
        "sympy",
        "1.14.0",
        "a2/09/77d55d46fd61b4a135c444fc97158ef34a095e5681d0a6c10b75bf356191/sympy-1.14.0-py3-none-any.whl",
        "e091cc3e99d2141a0ba2847328f5479b05d94a6635cb96148ccb3f34671bd8f5",
        6_299_353,
    ),
    Wheel(
        "mpmath",
        "1.3.0",
        "43/e3/7d92a15f894aa0c9c4b49b8ee9ac9850d6e63b03c9c32c0367a13ae62209/mpmath-1.3.0-py3-none-any.whl",
        "a0b2b9fe80bbcd81a6647ff13108738cfb482d481d826cc0e02f5b35e5c88d2c",
        536_198,
    ),
    Wheel(
        "networkx",
        "3.7",
        "7e/cd/fe58041e9011f307c490e3e17dd48cc516448f7c698a3f2d9d9d65d7e6a8/networkx-3.7-py3-none-any.whl",
        "e3fd2c13a7814cee3746340d8d7f8598a67f16a58bf47fb7f8793fab6efca1b0",
        2_142_205,
    ),
    Wheel(
        "jinja2",
        "3.1.6",
        "62/a1/3d680cbfd5f4b8f15abc1d571870c5fc3e594bb582bc3b64ea099db13e56/jinja2-3.1.6-py3-none-any.whl",
        "85ece4451f492d0c13c5dd7c13a64681a86afae63a5f347908daf103ce6d2f67",
        134_899,
    ),
    Wheel(
        "markupsafe",
        "3.0.4",
        "3f/bf/62495e180b7000aaf30000fff849e933f74264638057176cf46852500adc/"
        "markupsafe-3.0.4-cp312-cp312-win_amd64.whl",
        "11935df9bf455ed0c04eb87bcd720f02b1fe5e02128a9430f23aed6f93336fc7",
        14_303,
    ),
    Wheel(
        "setuptools",
        "84.2.1",
        "ef/55/6dc6d6bb953bdfc52519a0059592c52b6e3b8ad51cf8256dffceb55b0610/"
        "setuptools-84.2.1-py3-none-any.whl",
        "ca140b45d5325a9478f880e351484f62fef15f336462e449ca0c03c873d589c2",
        876_263,
    ),
    Wheel(
        "fsspec",
        "2026.9.0",
        "6c/c0/a98505f18594f1bce828bb159cec0fcf9860562f1a2c85913409fc8f3d9e/fsspec-2026.9.0-py3-none-any.whl",
        "8dd6e646e99ea382bd85f97a45e6b526a442d79423a7dc673f1e2756d05fcb5f",
        221_738,
    ),
    Wheel(
        "filelock",
        "4.2.4",
        "e0/1b/5e74289a1878e666bb4b65fe5805a04cafcac2aa5f519ef1ead67ef91c56/filelock-4.2.4-py3-none-any.whl",
        "301241438c57936ef93f4243a94f2559022268bfcf9b375a9a72eb0f8db8358e",
        162_475,
    ),
    Wheel(
        "typing-extensions",
        "4.16.0",
        "49/d3/b8441a820a491ddfc024b0b0cf0393375b75ea13866d9c66727e54c2fc80/"
        "typing_extensions-4.16.0-py3-none-any.whl",
        "481caa481374e813c1b176ada14e97f1f67a4539ce9cfeb3f350d78d6370c2e8",
        45_571,
    ),
)


def download_file(url: str, destination: Path, progress: Progress | None = None, *, sha256: str = "") -> None:
    """Скачивает файл: с докачкой, ходом загрузки и проверкой sha256 (если он задан).

    Пока файл качается, он называется <имя>.part: прерванная загрузка в следующий раз
    продолжится с того же места, а под своим именем файл появляется только целиком.

    Raises:
        OSError: нет сети, сервер ответил ошибкой, загрузка оборвалась или файл не совпал с sha256.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(f"{destination.name}.part")
    offset = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
    expected = 0
    try:
        response = urllib.request.urlopen(request, timeout=_TIMEOUT)
    except urllib.error.HTTPError as exc:
        if exc.code != 416 or not offset:
            raise
        # 416 на докачку: докачивать нечего — .part целиком, если он того же размера, что файл.
        size = exc.headers.get("Content-Range", "").rpartition("/")[2]
        exc.close()
        if size.isdigit() and int(size) != offset:
            part.unlink()
            raise OSError(f"{destination.name}: недокачанный файл испорчен — скачаю его заново") from exc
        if progress is not None:
            progress.advance(offset)
    else:
        with response:
            if response.status != 206:
                offset = 0  # сервер не умеет докачку — сначала
            elif not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise OSError(f"{destination.name}: сервер прислал не тот кусок файла")
            length = int(response.headers.get("Content-Length") or 0)
            expected = offset + length if length else 0
            if progress is not None:
                progress.total = progress.total or expected
                progress.advance(offset)
            with part.open("ab" if offset else "wb") as file:
                while chunk := response.read(_CHUNK):
                    file.write(chunk)
                    if progress is not None:
                        progress.advance(len(chunk))
    size = part.stat().st_size
    if expected and size != expected:
        raise OSError(
            f"{destination.name}: загрузка оборвалась на {megabytes(size)} из {megabytes(expected)}"
        )
    if sha256 and file_sha256(part) != sha256:
        part.unlink()
        raise OSError(f"{destination.name} скачался с ошибкой: не совпала контрольная сумма")
    os.replace(part, destination)


def file_sha256(path: Path) -> str:
    """sha256 файла в шестнадцатеричном виде."""
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _safe_extract(archive: zipfile.ZipFile, target: Path) -> None:
    """Распаковывает архив, не выпуская файлы за пределы target."""
    root = target.resolve()
    for member in archive.infolist():
        path = (root / member.filename).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Подозрительный путь в архиве: {member.filename}")
    archive.extractall(root)


def ensure_vosk(model_dir: Path, out: Any = sys.stdout) -> None:
    """Скачивает модель Vosk: папка называется так же, как архив на alphacephei.com.

    Архив распаковывается рядом и переносится на место целиком: оборванная распаковка
    не оставит полупустую модель.
    """
    if model_dir.is_dir():
        out.write(f"Vosk: уже есть ({model_dir})\n")
        return
    url = f"{VOSK_BASE_URL}{model_dir.name}.zip"
    log.info("Модель Vosk: %s", url)
    out.write("Скачиваю модель Vosk для обращения по имени (один раз)\n")
    archive_path = model_dir.with_name(f"{model_dir.name}.zip")
    download_file(url, archive_path, Progress(out, "Скачиваю модель Vosk"))
    unpacked = model_dir.with_name(f"{model_dir.name}.unpack")
    shutil.rmtree(unpacked, ignore_errors=True)
    with zipfile.ZipFile(archive_path) as archive:
        _safe_extract(archive, unpacked)
    if not (unpacked / model_dir.name).is_dir():
        raise RuntimeError(f"В архиве нет папки {model_dir.name}")
    os.replace(unpacked / model_dir.name, model_dir)
    shutil.rmtree(unpacked, ignore_errors=True)
    archive_path.unlink()
    out.write("Vosk: готово\n")


def piper_voice_ready(voice_path: Path) -> bool:
    """Голос Piper скачан: есть и .onnx, и .onnx.json рядом."""
    return voice_path.is_file() and Path(f"{voice_path}.json").is_file()


def piper_voice_urls(voice_path: Path) -> tuple[str, str]:
    """Откуда скачать голос Piper (.onnx и .onnx.json) — адреса как у piper.download_voices.

    Raises:
        ValueError: имя файла не похоже на голос Piper (ru_RU-dmitri-medium.onnx).
    """
    from piper.download_voices import URL_FORMAT, VOICE_PATTERN

    match = VOICE_PATTERN.match(voice_path.stem)
    if match is None:
        raise ValueError(f"имя голоса Piper должно быть вида ru_RU-dmitri-medium.onnx: {voice_path.name}")
    family = match.group("lang_family")
    parts = {
        "lang_family": family,
        "lang_code": f"{family}_{match.group('lang_region')}",
        "voice_name": match.group("voice_name"),
        "voice_quality": match.group("voice_quality"),
    }
    return URL_FORMAT.format(extension=".onnx", **parts), URL_FORMAT.format(extension=".onnx.json", **parts)


def ensure_piper(voice_path: Path, out: Any = sys.stdout) -> None:
    """Скачивает голос Piper (.onnx и .onnx.json) с Hugging Face."""
    if piper_voice_ready(voice_path):
        out.write(f"Piper: голос уже есть ({voice_path.name})\n")
        return
    model_url, config_url = piper_voice_urls(voice_path)
    out.write(f"Скачиваю голос Piper {voice_path.stem} (один раз)\n")
    config_path = Path(f"{voice_path}.json")
    if not config_path.is_file():
        download_file(config_url, config_path)
    if not voice_path.is_file():
        download_file(model_url, voice_path, Progress(out, "Скачиваю голос Piper"))
    out.write("Piper: готово\n")


def ensure_whisper(config: AppConfig, out: Any = sys.stdout) -> None:
    """Скачивает модель Whisper, которая будет использоваться на этом компьютере."""
    from faster_whisper.utils import download_model

    model, device, _ = resolve_settings(config.stt, cuda_available())
    download_dir = config.resolve(config.stt.download_dir)
    out.write(f"Whisper: модель {model} для {'видеокарты' if device == 'cuda' else 'процессора'}\n")
    try:
        download_model(model, cache_dir=str(download_dir), local_files_only=True)
    except Exception:
        with whisper_download_progress(model, download_dir, out):
            download_model(model, cache_dir=str(download_dir))
        out.write("Whisper: готово\n")
    else:
        out.write("Whisper: уже есть\n")


def extract_dlls(wheel: Path, folder: Path) -> list[str]:
    """Достаёт из колеса nvidia-* его библиотеки (nvidia/<пакет>/bin/*.dll) прямо в folder.

    Raises:
        OSError: в колесе нет DLL.
    """
    folder.mkdir(parents=True, exist_ok=True)
    names = []
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            parts = PurePosixPath(member.filename).parts
            if len(parts) != 4 or parts[0] != "nvidia" or parts[2] != "bin" or not parts[3].endswith(".dll"):
                continue
            with archive.open(member) as source, (folder / parts[3]).open("wb") as target:
                shutil.copyfileobj(source, target, _CHUNK)
            names.append(parts[3])
    if not names:
        raise OSError(f"В {wheel.name} нет DLL")
    return names


def extract_wheel(wheel: Path, folder: Path) -> None:
    """Раскладывает колесо в folder, как pip в site-packages: пакеты и *.data/purelib, platlib.

    Raises:
        ValueError: путь в колесе выходит за пределы folder.
    """
    root = folder.resolve()
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            parts = PurePosixPath(member.filename).parts
            if parts and parts[0].endswith(".data"):
                if len(parts) < 3 or parts[1] not in ("purelib", "platlib"):
                    continue  # scripts, headers, data программе не нужны
                parts = parts[2:]
            if not parts or member.is_dir():
                continue
            path = root.joinpath(*parts).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"Подозрительный путь в колесе: {member.filename}")
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, path.open("wb") as target:
                shutil.copyfileobj(source, target, _CHUNK)


def installed_wheels(target: Path) -> set[str]:
    """Колёса, которые уже разложены в папку target (файл INSTALLED)."""
    try:
        return set((target / INSTALLED).read_text(encoding="utf-8").split())
    except FileNotFoundError:
        return set()


def install_wheels(
    wheels: Sequence[Wheel],
    target: Path,
    out: Any,
    *,
    label: str,
    unpack: Callable[[Path, Path], None],
    space_factor: float,
) -> None:
    """Скачивает колёса (sha256 из uv.lock) и раскладывает их в target функцией unpack(колесо, target).

    Колесо качается в папку <target>.download рядом: прерванная загрузка продолжается с того
    же места, а колёса, которые уже разложены (файл INSTALLED), не скачиваются.

    Args:
        label: что скачивается — для сообщений («библиотеки CUDA для видеокарты»).
        space_factor: во сколько раз колесо вместе с распакованным больше самого колеса.

    Raises:
        OSError: нет сети, мало места на диске или файл скачался с ошибкой.
    """
    staging = target.with_name(f"{target.name}.download")
    title = label[:1].upper() + label[1:]
    done = installed_wheels(target)
    pending = [wheel for wheel in wheels if wheel.filename not in done]
    if not pending:
        shutil.rmtree(staging, ignore_errors=True)
        out.write(f"{title}: уже есть ({target})\n")
        return
    staging.mkdir(parents=True, exist_ok=True)
    left = sum(wheel.size for wheel in pending)
    need = int(space_factor * left) + (200 << 20)  # с запасом
    free = shutil.disk_usage(staging).free
    if free < need:
        raise OSError(f"мало места на диске: нужно {megabytes(need)}, свободно {megabytes(free)}")
    total = sum(wheel.size for wheel in wheels)
    out.write(f"Скачиваю {label} ({megabytes(left)}, один раз)\n")
    progress = Progress(out, f"Скачиваю {label}", total, done=total - left)
    target.mkdir(parents=True, exist_ok=True)
    for wheel in pending:
        path = staging / wheel.filename
        if path.exists():  # скачано и проверено, но не распаковано
            progress.advance(wheel.size)
        else:
            download_file(wheel.url, path, progress, sha256=wheel.sha256)
        out.write(f"Распаковываю {wheel.package}\n")
        unpack(path, target)
        with (target / INSTALLED).open("a", encoding="utf-8") as file:
            file.write(f"{wheel.filename}\n")
        path.unlink()
    shutil.rmtree(staging, ignore_errors=True)
    out.write(f"{title}: готово ({target})\n")


def _unpack_cuda(wheel: Path, target: Path) -> None:
    """DLL колеса — прямо в target; файлы, которые там уже есть, не заменяются."""
    unpacked = wheel.parent / "bin"
    for name in extract_dlls(wheel, unpacked):
        if (target / name).exists():
            (unpacked / name).unlink()
        else:
            os.replace(unpacked / name, target / name)


def ensure_cuda(target: Path, out: Any = sys.stdout) -> None:
    """Скачивает DLL cuBLAS 12 и cuDNN 9 для Whisper на видеокарте в папку target.

    Колёса nvidia-* с PyPI (около 1,3 ГБ) проверяются по sha256 из uv.lock, из них берутся
    только DLL (около 2 ГБ). Файлы, которые в target уже есть, не заменяются.

    Raises:
        OSError: нет сети, мало места на диске или файл скачался с ошибкой.
    """
    install_wheels(
        CUDA_WHEELS, target, out, label="библиотеки CUDA для видеокарты", unpack=_unpack_cuda, space_factor=2
    )


def download_cuda(out: Any = sys.stdout) -> bool:
    """--download-cuda: скачивает библиотеки CUDA в папку cuda и проверяет, что они загружаются."""
    try:
        ensure_cuda(cuda_dir(), out)
    except Exception as exc:
        log.exception("Не удалось скачать библиотеки CUDA")
        out.write(f"CUDA: ошибка — {exc}\n")
        return False
    register_cuda_dlls()
    if missing := missing_cuda_library():
        out.write(f"CUDA: библиотека {missing} не загружается\n")
        return False
    out.write("CUDA: библиотеки на месте и загружаются\n")
    return True


def silero_lib() -> Path:
    """Папка, куда раскладывается PyTorch для процесса синтеза Silero."""
    return silero_dir() / "lib"


def silero_runtime() -> tuple[bool, Path | None]:
    """Есть ли PyTorch для Silero и откуда его брать процессу синтеза.

    Возвращает (готов, папка): папка — скачанный PyTorch; None — torch из окружения Python
    (из исходников его ставит uv sync --extra silero).
    """
    lib = silero_lib()
    if {wheel.filename for wheel in SILERO_WHEELS} <= installed_wheels(lib):
        return True, lib
    if not FROZEN and importlib.util.find_spec("torch") is not None:
        return True, None
    return False, None


def ensure_silero(model: Path, out: Any = sys.stdout) -> None:
    """Голос Silero: PyTorch (если его ещё нет, ~130 МБ) и модель.

    Raises:
        OSError: нет сети, мало места на диске, файл скачался с ошибкой или модель неизвестна.
    """
    if not silero_runtime()[0]:
        install_wheels(
            SILERO_WHEELS, silero_lib(), out, label="PyTorch для голоса Silero", unpack=extract_wheel,
            space_factor=5,
        )  # fmt: skip
    if model.is_file():
        return
    if model.name not in SILERO_MODELS:
        raise OSError(f"неизвестная модель Silero {model.name}: скачать можно {', '.join(SILERO_MODELS)}")
    out.write(f"Скачиваю модель Silero {model.stem} (один раз)\n")
    download_file(
        SILERO_MODEL_URL + model.name, model, Progress(out, "Скачиваю модель Silero"),
        sha256=SILERO_MODELS[model.name],
    )  # fmt: skip
    out.write("Модель Silero: готово\n")


def download_silero(config: AppConfig, out: Any = sys.stdout) -> bool:
    """--download-silero: скачивает PyTorch и модель Silero и проверяет голос пробной фразой."""
    from assistant.tts.base import TTSError
    from assistant.tts.silero_tts import silero_engine

    model = config.resolve(config.tts.silero_model)
    try:
        ensure_silero(model, out)
    except Exception as exc:
        log.exception("Не удалось скачать Silero")
        out.write(f"Silero: ошибка — {exc}\n")
        return False
    out.write(f"Модель Silero {model.name}: sha256 {file_sha256(model)}\n")
    engine = silero_engine(config)
    started = time.perf_counter()
    try:
        audio, rate = engine.synthesize("Проверка голоса: сейчас 14:35, 21 октября, громкость 50%.")
    except TTSError as exc:
        out.write(f"Silero: голос не работает — {exc}\n")
        return False
    finally:
        engine.close()
    seconds = len(audio) / rate
    out.write(
        f"Silero: {seconds:.1f} с речи за {time.perf_counter() - started:.1f} с (с запуском процесса)\n"
    )
    return seconds > 1


def download_all(config: AppConfig, out: Any = sys.stdout) -> bool:
    """Скачивает всё для голоса. Возвращает False, если что-то не удалось.

    Библиотеки CUDA — только собранному exe и только для видеокарты NVIDIA: из исходников
    их ставит uv sync --extra cuda. Silero — если он выбран голосом ассистента.
    """
    steps = [
        ("Vosk", lambda: ensure_vosk(config.resolve(config.voice.vosk_model), out)),
        ("Piper", lambda: ensure_piper(config.resolve(config.tts.piper_voice), out)),
    ]
    if config.tts.engine == "silero":
        steps.append(("Silero", lambda: ensure_silero(config.resolve(config.tts.silero_model), out)))
    if FROZEN and cuda_libraries_needed(config.stt):
        steps.append(("CUDA", lambda: ensure_cuda(cuda_dir(), out)))
    steps.append(("Whisper", lambda: ensure_whisper(config, out)))
    ok = True
    for name, step in steps:
        try:
            step()
        except Exception as exc:
            log.exception("Не удалось скачать %s", name)
            out.write(f"{name}: ошибка — {exc}\n")
            ok = False
    if shutil.which("nvidia-smi") and not cuda_available():
        out.write("Видеокарта NVIDIA есть, но CUDA для Whisper недоступна — см. README.\n")
    return ok
