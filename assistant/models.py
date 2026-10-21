"""Скачивание моделей: Vosk (слово активации), голос Piper и Whisper.

Запуск: uv run python -m assistant --download-models
Уже скачанное повторно не загружается.
"""

from __future__ import annotations

import logging
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import TextIO

from assistant.config import AppConfig
from assistant.stt.whisper_stt import cuda_available, resolve_settings

log = logging.getLogger(__name__)

VOSK_BASE_URL = "https://alphacephei.com/vosk/models/"
_CHUNK = 1 << 20


def download_file(url: str, destination: Path, out: TextIO = sys.stdout) -> None:
    """Скачивает файл с выводом прогресса."""
    with urllib.request.urlopen(url, timeout=60) as response, destination.open("wb") as file:
        total = int(response.headers.get("Content-Length") or 0)
        done, last_percent = 0, -10
        while chunk := response.read(_CHUNK):
            file.write(chunk)
            done += len(chunk)
            if total:
                percent = done * 100 // total
                if percent >= last_percent + 10:
                    out.write(f"  {percent}% ({done >> 20} из {total >> 20} МБ)\n")
                    out.flush()
                    last_percent = percent


def _safe_extract(archive: zipfile.ZipFile, target: Path) -> None:
    """Распаковывает архив, не выпуская файлы за пределы target."""
    root = target.resolve()
    for member in archive.infolist():
        path = (root / member.filename).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Подозрительный путь в архиве: {member.filename}")
    archive.extractall(root)


def ensure_vosk(model_dir: Path, out: TextIO = sys.stdout) -> None:
    """Скачивает модель Vosk (папка называется так же, как архив на alphacephei.com)."""
    if model_dir.is_dir():
        out.write(f"Vosk: уже есть ({model_dir})\n")
        return
    url = f"{VOSK_BASE_URL}{model_dir.name}.zip"
    out.write(f"Vosk: скачиваю {url}\n")
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive_path = Path(tmp) / "model.zip"
        download_file(url, archive_path, out)
        with zipfile.ZipFile(archive_path) as archive:
            _safe_extract(archive, model_dir.parent)
    if not model_dir.is_dir():
        raise RuntimeError(f"В архиве нет папки {model_dir.name}")
    out.write("Vosk: готово\n")


def ensure_piper(voice_path: Path, out: TextIO = sys.stdout) -> None:
    """Скачивает голос Piper (.onnx и .onnx.json) с Hugging Face."""
    if voice_path.is_file() and Path(f"{voice_path}.json").is_file():
        out.write(f"Piper: голос уже есть ({voice_path.name})\n")
        return
    from piper.download_voices import download_voice

    out.write(f"Piper: скачиваю голос {voice_path.stem}\n")
    voice_path.parent.mkdir(parents=True, exist_ok=True)
    download_voice(voice_path.stem, voice_path.parent)
    out.write("Piper: готово\n")


def ensure_whisper(config: AppConfig, out: TextIO = sys.stdout) -> None:
    """Скачивает модель Whisper, которая будет использоваться на этом компьютере."""
    from faster_whisper.utils import download_model

    model, device, _ = resolve_settings(config.stt, cuda_available())
    out.write(f"Whisper: модель {model} для {'видеокарты' if device == 'cuda' else 'процессора'}\n")
    download_model(model, cache_dir=str(config.resolve(config.stt.download_dir)))
    out.write("Whisper: готово\n")


def download_all(config: AppConfig, out: TextIO = sys.stdout) -> bool:
    """Скачивает все модели. Возвращает False, если что-то не удалось."""
    steps = [
        ("Vosk", lambda: ensure_vosk(config.resolve(config.voice.vosk_model), out)),
        ("Piper", lambda: ensure_piper(config.resolve(config.tts.piper_voice), out)),
        ("Whisper", lambda: ensure_whisper(config, out)),
    ]
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
