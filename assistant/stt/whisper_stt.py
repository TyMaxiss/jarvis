"""Распознавание речи через faster-whisper.

Модель загружается один раз при старте. Видеокарта NVIDIA (CUDA) определяется
автоматически; если CUDA или её библиотеки недоступны, используется процессор.
Собранный exe недостающие библиотеки CUDA скачивает сам (fetch_cuda).
Скачанная модель берётся из локального кэша без обращения к интернету.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import numpy as np

from assistant.config import SttConfig
from assistant.frozen import cuda_dir, keep_dll_search
from assistant.progress import Progress, megabytes, watch_folder

log = logging.getLogger(__name__)

# На тишине и шуме Whisper иногда «слышит» фразы из субтитров обучающих видео.
HALLUCINATION_MARKERS = (
    "субтитр",  # «Субтитры сделал DimaTorzok», «Редактор субтитров …»
    "dimatorzok",
    "продолжение следует",
    "спасибо за просмотр",
    "подписывайтесь на канал",
    "подпишись на канал",
)
_MAX_PROMPT_CHARS = 600
# Без этих DLL CTranslate2 не запустит Whisper на видеокарте. Версии CUDA и cuDNN — те же,
# что у пакетов nvidia-*-cu12 в pyproject.toml (uv sync --extra cuda). cudnn64_9.dll не годится
# для проверки: свой маленький загрузчик с этим именем лежит в самом пакете ctranslate2 и
# загружается при его импорте; cudnn_ops64_9.dll есть только в настоящем cuDNN 9.
CUDA_LIBRARIES = ("cublas64_12.dll", "cudnn_ops64_9.dll")
# DLL, которые CTranslate2 и cuBLAS подгружают по имени (через PATH) уже во время работы на
# видеокарте: заранее загруженные по полному пути, они находятся и без PATH.
PRELOADED_CUDA = ("cublasLt64_*.dll", "cublas64_*.dll", "nvrtc64_*.dll", "nvrtc-builtins64_*.dll")
_LOAD_WITH_ALTERED_SEARCH_PATH = 0x8  # зависимости DLL — из её же папки
# Какие файлы модели скачивает faster_whisper.utils.download_model (allow_patterns).
WHISPER_FILES = ("config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*")
# Тишина по краям фразы: без неё Whisper иногда «съедает» первый звук («Делай» вместо «Сделай»).
_PAD_SECONDS = 0.3
# Файл модели есть, но не открылся (его держит другая программа, например антивирус): через
# столько секунд — ещё одна попытка.
OPEN_RETRY_PAUSE = 3.0


def is_hallucination(text: str) -> bool:
    """Похоже ли распознанное на известную «галлюцинацию» Whisper."""
    lowered = text.lower()
    return any(marker in lowered for marker in HALLUCINATION_MARKERS)


def build_prompt(name: str, command_phrases: Iterable[str], app_names: Iterable[str], max_apps: int) -> str:
    """Подсказка для Whisper: имя ассистента, частые слова команд и названия программ.

    Подсказка помогает правильно писать «Telegram», «Discord» и слова из команд.
    """
    counter: Counter[str] = Counter()
    for phrase in command_phrases:
        counter.update(w for w in phrase.lower().split() if "{" not in w and len(w) > 2)
    words = [w for w, _ in counter.most_common(25)]
    apps = list(dict.fromkeys(app_names))[:max_apps] if max_apps else []
    prompt = f"{name.capitalize()}, {', '.join(words)}."
    if apps:
        prompt += " " + ", ".join(apps) + "."
    return prompt[:_MAX_PROMPT_CHARS]


@dataclass(frozen=True)
class Transcript:
    """Результат распознавания."""

    text: str
    duration_ms: float
    no_speech_prob: float = 0.0


def cuda_dirs() -> list[Path]:
    """Папки с DLL CUDA: из pip-пакетов nvidia-* (uv sync --extra cuda) и папка cuda рядом с программой."""
    dirs = []
    try:
        import nvidia  # type: ignore[import-not-found]
    except ImportError:
        pass
    else:
        for base in getattr(nvidia, "__path__", []):
            dirs += sorted(Path(base).glob("*/bin"))
    own = cuda_dir()  # в собранный exe CUDA не входит: он скачивает DLL сюда сам
    if own.is_dir():
        dirs.append(own)
    return dirs


def register_cuda_dlls() -> None:
    """Делает видимыми DLL CUDA (cuda_dirs); уже добавленные папки второй раз не добавляются.

    CTranslate2 загружает cuBLAS и cuDNN при первом распознавании на видеокарте.
    Пакеты nvidia-cublas-cu12 и nvidia-cudnn-cu12 кладут DLL в site-packages/nvidia/*/bin,
    куда Windows сама не заглядывает.
    """
    if sys.platform != "win32":
        return
    with keep_dll_search():  # иначе external_launch, вернув свой PATH, сотрёт добавленную папку
        known = {os.path.normcase(part) for part in os.environ.get("PATH", "").split(os.pathsep)}
        for bin_dir in cuda_dirs():
            if os.path.normcase(str(bin_dir)) in known:
                continue
            os.add_dll_directory(str(bin_dir))  # type: ignore[attr-defined]
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
            log.debug("DLL CUDA: %s", bin_dir)


def missing_cuda_library() -> str:
    """Какой DLL CUDA не находится ('' — все на месте или проверять нечего: не Windows).

    Ищет так же, как CTranslate2 (LoadLibrary по имени: папка Python, системные папки, PATH),
    поэтому после register_cuda_dlls находит и DLL из пакетов nvidia-*.
    """
    if sys.platform != "win32":
        return ""
    import ctypes

    with keep_dll_search():
        # Загруженные здесь DLL остаются в процессе: дальше видеокарте не нужен PATH, который
        # external_launch на миг меняет, — ни при создании модели, ни при распознавании.
        for folder in cuda_dirs():
            for pattern in PRELOADED_CUDA:
                for path in sorted(folder.glob(pattern)):
                    try:
                        ctypes.WinDLL(str(path), winmode=_LOAD_WITH_ALTERED_SEARCH_PATH)
                    except OSError as exc:
                        log.info("DLL CUDA %s не загружается: %s", path, exc)
        for name in CUDA_LIBRARIES:
            try:
                ctypes.WinDLL(name, winmode=0)
            except OSError as exc:
                log.info("DLL CUDA %s не загружается: %s", name, exc)
                return name
    return ""


def default_cpu_threads() -> int:
    """Потоков для Whisper на процессоре: по числу физических ядер, но не меньше 4.

    4 — значение faster-whisper по умолчанию; на процессорах с большим числом ядер
    берём все физические ядра.
    """
    import psutil

    return max(4, psutil.cpu_count(logical=False) or 0)


def cuda_available() -> bool:
    """Есть ли видеокарта, которую видит CTranslate2."""
    try:
        with keep_dll_search():
            import ctranslate2

            return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        log.warning("CTranslate2 не смог проверить видеокарту", exc_info=True)
        return False


def nvidia_driver_status() -> str:
    """Что говорит о видеокарте сам драйвер NVIDIA ('' — драйвера нет): для лога, когда CTranslate2
    видеокарту не видит."""
    if sys.platform != "win32":
        return ""
    import ctypes

    try:
        driver = ctypes.WinDLL("nvcuda.dll")
    except OSError:
        return ""
    count, version = ctypes.c_int(), ctypes.c_int()
    if error := driver.cuInit(0):
        return f"драйвер NVIDIA не запустил CUDA (cuInit: ошибка {error})"
    driver.cuDeviceGetCount(ctypes.byref(count))
    driver.cuDriverGetVersion(ctypes.byref(version))
    cuda = f"{version.value // 1000}.{version.value % 1000 // 10}"
    return f"драйвер NVIDIA видит видеокарт: {count.value}, CUDA {cuda}"


def resolve_settings(config: SttConfig, has_cuda: bool) -> tuple[str, str, str]:
    """Модель, устройство и тип вычислений с учётом «auto»."""
    device = config.device if config.device != "auto" else ("cuda" if has_cuda else "cpu")
    model = config.model if config.model != "auto" else ("large-v3-turbo" if device == "cuda" else "small")
    compute = config.compute_type
    if compute == "auto":
        compute = "float16" if device == "cuda" else "int8"
    return model, device, compute


def cuda_libraries_needed(config: SttConfig) -> bool:
    """Whisper пойдёт на видеокарту, а её DLL не находятся — их надо скачать."""
    if resolve_settings(config, cuda_available())[1] != "cuda":
        return False
    register_cuda_dlls()
    return bool(missing_cuda_library())


def whisper_repo(model: str) -> str:
    """Репозиторий модели на Hugging Face — как в faster_whisper.utils.download_model ('' — неизвестно)."""
    if "/" in model:
        return model
    from faster_whisper import utils

    return getattr(utils, "_MODELS", {}).get(model, "")


def whisper_download_size(repo: str) -> int:
    """Сколько байт скачивать для модели: её файлы на Hugging Face (0 — узнать не удалось)."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo, files_metadata=True, timeout=10)
    except Exception:
        log.debug("Размер модели %s неизвестен", repo, exc_info=True)
        return 0
    files = info.siblings or []
    return sum(f.size or 0 for f in files if any(fnmatch(f.rfilename, pattern) for pattern in WHISPER_FILES))


def whisper_download_progress(
    model: str, download_dir: Path, out: Any
) -> contextlib.AbstractContextManager[None]:
    """Ход скачивания модели (на время блока): сколько байт в её папке кэша Hugging Face."""
    repo = whisper_repo(model)
    if out is None or not repo:
        return contextlib.nullcontext()
    out.write(f"Скачиваю модель распознавания речи Whisper {model} (один раз)\n")
    progress = Progress(out, f"Скачиваю модель Whisper {model}", whisper_download_size(repo))
    # Так кэш Hugging Face называет папку модели: models--Systran--faster-whisper-small.
    return watch_folder(download_dir / f"models--{repo.replace('/', '--')}", progress)


class ModelDownloadError(OSError):
    """Модели Whisper на диске нет, а скачать её не удалось (нет сети и т. п.)."""


def cached_model(model: str, download_dir: Path) -> str | None:
    """Папка модели Whisper на диске: в кэше download_dir, а если model — папка, то она сама.

    None — модели на диске нет или она скачана не до конца: её надо скачать.
    """
    if os.path.isdir(model):
        return model
    from faster_whisper.utils import download_model
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        path = str(download_model(model, cache_dir=str(download_dir), local_files_only=True))
    except LocalEntryNotFoundError:
        return None
    # Скачанное snapshot_download сверяет со списком файлов, только если тот сохранился: без него
    # папка оборванной загрузки (без model.bin) сошла бы за готовую. Файлы не удаляются: недостающее
    # докачает fetch_model.
    if not os.path.isfile(os.path.join(path, "model.bin")):
        log.warning("В модели Whisper %s на диске нет model.bin (%s) — её надо докачать", model, path)
        return None
    return path


def model_files(folder: str) -> str:
    """Что лежит в папке модели (для лога): файлы, размеры, ссылки и открывается ли model.bin."""
    parts = []
    for path in sorted(Path(folder).iterdir()):
        try:
            size = megabytes(path.stat().st_size)
        except OSError as exc:
            size = f"не прочитать ({exc})"
        link = f" → {os.readlink(path)}" if path.is_symlink() else ""
        parts.append(f"{path.name}: {size}{link}")
    return f"{'; '.join(parts)}; model.bin {open_problem(folder) or 'открывается'}"


def open_problem(folder: str) -> str:
    """Почему model.bin не открывается на чтение ('' — открывается)."""
    try:
        with open(os.path.join(folder, "model.bin"), "rb") as file:
            file.read(1)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 32:  # ERROR_SHARING_VIOLATION
            return f"занят другой программой — часто это антивирус ({exc})"
        return f"не открывается ({exc})"
    return ""


def fetch_model(model: str, download_dir: Path, out: Any) -> str:
    """Скачивает модель Whisper в кэш download_dir (с ходом загрузки для out) и возвращает её папку.

    Raises:
        ModelDownloadError: скачать не удалось.
    """
    from faster_whisper.utils import download_model

    try:
        with whisper_download_progress(model, download_dir, out):
            path = str(download_model(model, cache_dir=str(download_dir)))
    except Exception as exc:
        raise ModelDownloadError(f"не удалось скачать модель Whisper {model} ({exc})") from exc
    # Без связи с Hugging Face snapshot_download не падает, а отдаёт то, что есть на диске.
    if not os.path.isfile(os.path.join(path, "model.bin")):
        raise ModelDownloadError(f"не удалось скачать модель Whisper {model} (model.bin не скачался)")
    return path


def fetch_cuda_libraries(fetch: Callable[[], None]) -> str:
    """Скачивает DLL CUDA и делает их видимыми: '' — скачались, иначе — почему нет."""
    try:
        fetch()
    except Exception as exc:
        log.warning("Не удалось скачать библиотеки CUDA", exc_info=True)
        return str(exc) or type(exc).__name__
    finally:
        register_cuda_dlls()  # папка cuda могла появиться, даже если скачалось не всё
    return ""


def cuda_message(missing: str, fetch_error: str | None) -> str:
    """Почему Whisper не на видеокарте, хотя она есть: нет DLL CUDA.

    fetch_error: None — скачивать DLL нечем (запуск из исходников), '' — скачались, но не
    загружаются, иначе — почему не скачались.
    """
    if fetch_error is None:
        how = "Включить видеокарту: uv sync --extra cuda."
        reason = f"нет библиотек CUDA ({missing})"
    elif fetch_error:
        how = "Скачать их попробую при следующем запуске."
        reason = f"не удалось скачать библиотеки CUDA ({fetch_error})"
    else:
        how = f"Удалите папку {cuda_dir()} — при следующем запуске библиотеки скачаются заново."
        reason = f"библиотека CUDA {missing} не загружается"
    return (
        f"Видеокарта NVIDIA не используется: {reason}, распознаю на процессоре — это медленнее. {how} "
        "Чтобы не видеть это сообщение, выберите в настройках распознавания «Процессор»."
    )


class WhisperSTT:
    """Обёртка над faster_whisper.WhisperModel.

    Args:
        config: настройки распознавания.
        download_dir: куда скачивать модели.
        prompt: функция, возвращающая initial_prompt (список программ строится в фоне).
        out: куда сообщать о скачивании модели (консоль или окна); None — никуда.
        fetch_cuda: скачивает DLL CUDA, если их нет (собранный exe); None — не скачивать.
    """

    def __init__(
        self,
        config: SttConfig,
        download_dir: Path,
        prompt: Callable[[], str] | None = None,
        *,
        out: Any = None,
        fetch_cuda: Callable[[], None] | None = None,
    ) -> None:
        self._config = config
        self._download_dir = download_dir
        self._prompt = prompt or (lambda: "")
        self._out = out
        self._fetch_cuda = fetch_cuda
        self._model: Any = None
        self.model_name = ""
        self.device = ""
        self.cuda_problem = ""  # почему не на видеокарте, хотя она есть ('' — всё в порядке)
        self.gpu_error = ""  # то же коротко — для сообщения, если не загрузился и процессор

    def load(self) -> None:
        """Загружает модель: видеокарта, а если она не запускается — процессор (cuda_problem — почему).

        На процессоре — сначала своя модель, а если она не загрузится, то уже скачанная модель
        для видеокарты: голос заработает без перезапуска.
        """
        register_cuda_dlls()
        model, device, compute = resolve_settings(self._config, cuda_available())
        self.cuda_problem = self.gpu_error = ""
        gpu_model = resolve_settings(self._config.model_copy(update={"device": "cuda"}), True)[0]
        if device == "cpu":
            chosen = self._config.device == "cpu"
            if not chosen and (driver := nvidia_driver_status()):
                log.warning("CTranslate2 не видит видеокарту (%s) — распознаю на процессоре", driver)
            self._load_on_cpu(model, gpu_model, compute, prefer_on_disk=not chosen)
            return
        self.gpu_error = self._gpu_problem(model, compute)
        if not self.gpu_error:
            return
        cpu_model = resolve_settings(self._config.model_copy(update={"device": "cpu"}), False)[0]
        self._load_on_cpu(cpu_model, model, "int8", prefer_on_disk=True)

    def _gpu_problem(self, model: str, compute: str) -> str:
        """Загружает Whisper на видеокарту: '' — получилось, иначе — почему нет (cuda_problem — для людей)."""
        missing = missing_cuda_library()
        fetch_error: str | None = None
        if missing and self._fetch_cuda is not None:
            fetch_error = fetch_cuda_libraries(self._fetch_cuda)
            missing = missing_cuda_library()
        if missing:
            # Модель для видеокарты даже не загружаем: без DLL она не заработает.
            self.cuda_problem = cuda_message(missing, fetch_error)
            log.warning("%s", self.cuda_problem)
            return f"не загружается {missing}"
        try:
            self._load(model, "cuda", compute)
            self._warm_up()  # часть ошибок CUDA видна только при распознавании
        except Exception as exc:
            self._model = None
            log.warning("Whisper на видеокарте не запустился", exc_info=True)
            reason = str(exc) or type(exc).__name__  # пустая причина значила бы «получилось»
            self.cuda_problem = (
                f"Whisper на видеокарте не запустился ({reason}) — распознаю на процессоре. "
                "При следующем запуске снова попробую видеокарту."
            )
            return reason
        return ""

    def _load_on_cpu(self, model: str, spare: str, compute: str, *, prefer_on_disk: bool) -> None:
        """Whisper на процессоре: model, а если она не загрузится — spare, если та уже скачана.

        prefer_on_disk: процессор выбран не пользователем — если model ещё не скачана, а spare
        на диске есть, сначала spare: ради процессора не стоит качать ещё полгигабайта.
        """
        candidates = [model]
        if spare != model and cached_model(spare, self._download_dir):
            if prefer_on_disk and not cached_model(model, self._download_dir):
                candidates.insert(0, spare)
            else:
                candidates.append(spare)
        for index, name in enumerate(candidates):
            try:
                self._load(name, "cpu", compute)
                return
            except Exception as exc:
                if index == len(candidates) - 1:
                    raise
                log.warning(
                    "Whisper %s на процессоре не загрузился (%s) — пробую %s",
                    name, exc, candidates[index + 1], exc_info=True,
                )  # fmt: skip

    def _load(self, model: str, device: str, compute: str) -> None:
        from faster_whisper import WhisperModel

        started = time.perf_counter()
        self._download_dir.mkdir(parents=True, exist_ok=True)
        # Скачивается, только если модели на диске нет: скачанную модель, которая не загрузилась,
        # повторное скачивание не починит, а «Скачиваю…» в окне только собьёт с толку.
        path = cached_model(model, self._download_dir)
        if path is None:
            log.info("Модели Whisper %s нет на диске — скачиваю (один раз)", model)
            path = fetch_model(model, self._download_dir, self._out)
        options: dict[str, Any] = {"device": device, "compute_type": compute}
        if device == "cpu":
            options["cpu_threads"] = self._config.cpu_threads or default_cpu_threads()
        try:
            self._model = WhisperModel(path, **options)
        except RuntimeError as exc:
            if "Unable to open file" not in str(exc):
                raise
            # Файл модели есть, но не открылся — обычно его в этот миг держит другая программа
            # (антивирус проверяет): ещё одна попытка чуть позже, а в лог — что лежит в папке.
            log.warning(
                "Whisper %s: %s — пробую ещё раз через %.0f с. Папка модели: %s",
                model, exc, OPEN_RETRY_PAUSE, model_files(path),
            )  # fmt: skip
            time.sleep(OPEN_RETRY_PAUSE)
            try:
                self._model = WhisperModel(path, **options)
            except RuntimeError as again:
                problem = open_problem(path)
                raise RuntimeError(f"{again}; model.bin {problem}" if problem else str(again)) from again
        self.model_name, self.device = model, device
        log.info(
            "Whisper %s (%s, %s) загружен за %.1f с", model, device, compute, time.perf_counter() - started
        )

    def _warm_up(self) -> None:
        segments, _ = self._model.transcribe(np.zeros(16_000, dtype=np.float32), language="ru", beam_size=1)
        list(segments)

    def transcribe(self, audio: np.ndarray, beam_size: int | None = None) -> Transcript:
        """Распознаёт фразу (int16 или float32, 16 кГц).

        Возвращает пустой текст, если Whisper не услышал речь или выдал «галлюцинацию».

        Args:
            audio: звук фразы.
            beam_size: ширина поиска; по умолчанию — из настроек.
        """
        if self._model is None:
            raise RuntimeError("Модель Whisper не загружена")
        samples = audio.astype(np.float32)
        if audio.dtype == np.int16:
            samples /= 32768.0
        pad = np.zeros(int(16_000 * _PAD_SECONDS), dtype=np.float32)
        samples = np.concatenate([pad, samples, pad])
        started = time.perf_counter()
        segments, _ = self._model.transcribe(
            samples,
            language="ru",
            beam_size=beam_size or self._config.beam_size,
            initial_prompt=self._prompt() or None,
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=False,  # фраза уже вырезана нашим VAD
        )
        parts, no_speech = [], 0.0
        for segment in segments:
            no_speech = max(no_speech, segment.no_speech_prob)
            if segment.no_speech_prob < 0.8:
                parts.append(segment.text.strip())
        text = " ".join(p for p in parts if p).strip()
        duration_ms = (time.perf_counter() - started) * 1000
        if text and is_hallucination(text):
            log.info("Отброшена «галлюцинация» Whisper: %s", text)
            text = ""
        return Transcript(text, duration_ms, no_speech)
