"""Скачивание при первом запуске: докачка, проверка sha256, модели и библиотеки CUDA.

Файлы раздаёт локальный HTTP-сервер (с поддержкой Range, как PyPI и Hugging Face),
поэтому интернет тестам не нужен.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import threading
import tomllib
import zipfile
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from assistant import models
from assistant.config import AppConfig
from assistant.models import Wheel, download_file, ensure_cuda, ensure_piper, ensure_vosk, extract_dlls
from assistant.progress import Progress, folder_size, megabytes, watch_folder
from tests.conftest import ROOT, wait_until

MB = 1 << 20


class Files:
    """Локальный сервер файлов: Range, обрыв соединения и «сервер без докачки» — по желанию."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.requests: list[tuple[str, str]] = []  # (путь, Range)
        self.ranges = True
        self.cut_after: int | None = None  # оборвать ответ после стольких байт
        files = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                path = self.path.split("?")[0]
                files.requests.append((path, self.headers.get("Range", "")))
                data = files.files.get(path)
                if data is None:
                    self.send_error(404)
                    return
                start = 0
                wanted = self.headers.get("Range", "")
                if wanted and files.ranges:
                    start = int(wanted.removeprefix("bytes=").rstrip("-"))
                    if start >= len(data):
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{len(data)}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                else:
                    self.send_response(200)
                body = data[start:]
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body if files.cut_after is None else body[: files.cut_after])

            def log_message(self, *args: Any) -> None:
                """Без вывода в консоль."""

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def add(self, path: str, data: bytes) -> str:
        self.files[path] = data
        return self.url + path


@pytest.fixture
def server() -> Iterator[Files]:
    files = Files()
    yield files
    files.server.shutdown()


class Window(io.StringIO):
    """«Вывод» окон: строки и ход скачивания (метод progress, как у UiOut)."""

    def __init__(self) -> None:
        super().__init__()
        self.steps: list[tuple[str, float | None]] = []

    def progress(self, text: str, fraction: float | None) -> None:
        self.steps.append((text, fraction))


def payload(size: int) -> bytes:
    return bytes(range(256)) * (size // 256) + bytes(size % 256)


# --- Ход загрузки ---------------------------------------------------------------------


def test_megabytes() -> None:
    assert [megabytes(n) for n in (0, 400_000, 45 * MB, 1400 * MB)] == ["0 МБ", "0 МБ", "45 МБ", "1,4 ГБ"]


def test_progress_lines_every_ten_percent() -> None:
    out = io.StringIO()
    progress = Progress(out, "Скачиваю модель Vosk", 100 * MB)
    for _ in range(100):
        progress.advance(MB)
    lines = out.getvalue().splitlines()
    assert len(lines) == 11  # начало и затем каждые 10%
    assert lines[0] == "  Скачиваю модель Vosk: 1% (1 МБ из 100 МБ)"
    assert lines[-1] == "  Скачиваю модель Vosk: 100% (100 МБ из 100 МБ)"


def test_progress_for_windows_has_fraction_and_always_ends_with_100() -> None:
    window = Window()
    progress = Progress(window, "Скачиваю голос Piper", 50 * MB, interval=60)
    for _ in range(50):
        progress.advance(MB)
    # Не чаще interval — но последний шаг (100%) показывается всегда.
    assert window.steps == [
        ("Скачиваю голос Piper: 2% (1 МБ из 50 МБ)", 0.02),
        ("Скачиваю голос Piper: 100% (50 МБ из 50 МБ)", 1.0),
    ]
    assert window.getvalue() == ""  # в строки (лог) ход загрузки окон не попадает


def test_progress_without_size_counts_megabytes() -> None:
    out, window = io.StringIO(), Window()
    for target in (out, window):
        progress = Progress(target, "Скачиваю модель Whisper small", interval=0)
        for done in (MB // 2, 3 * MB, 150 * MB, 151 * MB):
            progress.update(done)
    assert out.getvalue().splitlines() == [
        "  Скачиваю модель Whisper small: 0 МБ",
        "  Скачиваю модель Whisper small: 150 МБ",
    ]
    assert [step for step, _ in window.steps][-1] == "Скачиваю модель Whisper small: 151 МБ"
    assert {fraction for _, fraction in window.steps} == {None}


def test_watch_folder_reports_growing_size(tmp_path: Path) -> None:
    window = Window()
    progress = Progress(window, "Скачиваю", 4 * MB, interval=0)
    (tmp_path / "blobs").mkdir()
    with watch_folder(tmp_path, progress, interval=0.01):
        (tmp_path / "blobs" / "model.bin.incomplete").write_bytes(bytes(2 * MB))
        assert wait_until(lambda: progress.done == 2 * MB)
    assert folder_size(tmp_path) == 2 * MB
    assert window.steps[-1] == ("Скачиваю: 50% (2 МБ из 4 МБ)", 0.5)


# --- download_file ---------------------------------------------------------------------


def test_download_with_progress_and_checksum(server: Files, tmp_path: Path) -> None:
    data = payload(3 * MB + 100)
    url = server.add("/file.bin", data)
    out = io.StringIO()
    target = tmp_path / "sub" / "file.bin"
    download_file(url, target, Progress(out, "Файл"), sha256=hashlib.sha256(data).hexdigest())
    assert target.read_bytes() == data
    assert not (tmp_path / "sub" / "file.bin.part").exists()
    assert out.getvalue().splitlines()[-1] == "  Файл: 100% (3 МБ из 3 МБ)"


def test_download_resumes_part(server: Files, tmp_path: Path) -> None:
    data = payload(2 * MB)
    url = server.add("/file.bin", data)
    target = tmp_path / "file.bin"
    (tmp_path / "file.bin.part").write_bytes(data[:700_000])
    progress = Progress(io.StringIO(), "Файл")
    download_file(url, target, progress, sha256=hashlib.sha256(data).hexdigest())
    assert target.read_bytes() == data
    assert server.requests == [("/file.bin", "bytes=700000-")]
    assert (progress.done, progress.total) == (len(data), len(data))


def test_download_starts_over_when_server_cannot_resume(server: Files, tmp_path: Path) -> None:
    data = payload(MB)
    url = server.add("/file.bin", data)
    server.ranges = False
    (tmp_path / "file.bin.part").write_bytes(b"garbage")
    download_file(url, tmp_path / "file.bin")
    assert (tmp_path / "file.bin").read_bytes() == data


def test_download_of_complete_part_only_checks_it(server: Files, tmp_path: Path) -> None:
    data = payload(MB)
    url = server.add("/file.bin", data)
    (tmp_path / "file.bin.part").write_bytes(data)
    download_file(url, tmp_path / "file.bin", sha256=hashlib.sha256(data).hexdigest())  # сервер: 416
    assert (tmp_path / "file.bin").read_bytes() == data


def test_part_longer_than_file_is_downloaded_again(server: Files, tmp_path: Path) -> None:
    url = server.add("/file.bin", payload(MB))
    (tmp_path / "file.bin.part").write_bytes(payload(MB) + b"extra")  # сервер: 416, а файл короче
    with pytest.raises(OSError, match="недокачанный файл испорчен"):
        download_file(url, tmp_path / "file.bin")
    assert list(tmp_path.iterdir()) == []
    download_file(url, tmp_path / "file.bin")
    assert (tmp_path / "file.bin").read_bytes() == payload(MB)


def test_broken_download_keeps_part_for_next_time(server: Files, tmp_path: Path) -> None:
    data = payload(2 * MB)
    url = server.add("/file.bin", data)
    server.cut_after = MB
    with pytest.raises(OSError, match="загрузка оборвалась на 1 МБ из 2 МБ"):
        download_file(url, tmp_path / "file.bin")
    assert not (tmp_path / "file.bin").exists()
    assert (tmp_path / "file.bin.part").stat().st_size == MB
    server.cut_after = None
    download_file(url, tmp_path / "file.bin")
    assert (tmp_path / "file.bin").read_bytes() == data
    assert server.requests[-1] == ("/file.bin", f"bytes={MB}-")


def test_wrong_checksum_deletes_file(server: Files, tmp_path: Path) -> None:
    url = server.add("/file.bin", payload(MB))
    with pytest.raises(OSError, match="контрольная сумма"):
        download_file(url, tmp_path / "file.bin", sha256="0" * 64)
    assert list(tmp_path.iterdir()) == []


def test_missing_file_is_an_error(server: Files, tmp_path: Path) -> None:
    with pytest.raises(OSError, match="404"):
        download_file(server.url + "/nothing", tmp_path / "file.bin")


# --- Vosk и Piper ---------------------------------------------------------------------


def zip_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def test_ensure_vosk_unpacks_whole_model(
    server: Files, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.add("/vosk-model-test.zip", zip_bytes({"vosk-model-test/am/final.mdl": b"model"}))
    monkeypatch.setattr(models, "VOSK_BASE_URL", server.url + "/")
    model_dir = tmp_path / "models" / "vosk-model-test"
    out = io.StringIO()
    ensure_vosk(model_dir, out)
    assert (model_dir / "am" / "final.mdl").read_bytes() == b"model"
    assert sorted(p.name for p in model_dir.parent.iterdir()) == ["vosk-model-test"]  # ни архива, ни .unpack
    assert "Скачиваю модель Vosk для обращения по имени (один раз)" in out.getvalue()
    ensure_vosk(model_dir, out)
    assert len(server.requests) == 1


def test_ensure_vosk_without_model_folder_in_archive(
    server: Files, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.add("/vosk-model-test.zip", zip_bytes({"other/readme.txt": b"x"}))
    monkeypatch.setattr(models, "VOSK_BASE_URL", server.url + "/")
    with pytest.raises(RuntimeError, match="нет папки vosk-model-test"):
        ensure_vosk(tmp_path / "vosk-model-test", io.StringIO())
    assert not (tmp_path / "vosk-model-test").exists()


def test_piper_voice_urls_like_piper_download_voices() -> None:
    model_url, config_url = models.piper_voice_urls(Path("models/piper/ru_RU-dmitri-medium.onnx"))
    assert model_url == (
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/dmitri/medium/"
        "ru_RU-dmitri-medium.onnx?download=true"
    )
    assert config_url == model_url.replace(".onnx?", ".onnx.json?")
    with pytest.raises(ValueError, match="ru_RU-dmitri-medium.onnx"):
        models.piper_voice_urls(Path("voice.onnx"))


def test_ensure_piper_downloads_both_files(
    server: Files, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    urls = (server.add("/voice.onnx", payload(MB)), server.add("/voice.onnx.json", b"{}"))
    monkeypatch.setattr(models, "piper_voice_urls", lambda path: urls)
    voice = tmp_path / "piper" / "ru_RU-dmitri-medium.onnx"
    window = Window()
    ensure_piper(voice, window)
    assert models.piper_voice_ready(voice) and voice.read_bytes() == payload(MB)
    assert window.steps[-1][0] == "Скачиваю голос Piper: 100% (1 МБ из 1 МБ)"
    assert "Скачиваю голос Piper ru_RU-dmitri-medium (один раз)" in window.getvalue()


# --- Библиотеки CUDA ------------------------------------------------------------------


def lock_packages() -> dict[str, dict[str, Any]]:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    return {package["name"]: package for package in lock["package"]}


def test_cuda_wheels_match_uv_lock() -> None:
    packages = lock_packages()
    # Все пакеты extra cuda с зависимостями — те же колёса, что ставит uv sync --extra cuda.
    wanted = [dep["name"] for dep in packages["jarvis-assistant"]["optional-dependencies"]["cuda"]]
    closure: set[str] = set()
    while wanted:
        name = wanted.pop()
        if name not in closure:
            closure.add(name)
            wanted += [dep["name"] for dep in packages[name].get("dependencies", [])]
    assert {wheel.package for wheel in models.CUDA_WHEELS} == closure
    for wheel in models.CUDA_WHEELS:
        package = packages[wheel.package]
        assert package["version"] == wheel.version
        locked = next(w for w in package["wheels"] if w["url"].endswith("win_amd64.whl"))
        assert (locked["url"], locked["hash"], locked["size"]) == (
            wheel.url,
            f"sha256:{wheel.sha256}",
            wheel.size,
        )


def wheel_bytes(package: str, dlls: dict[str, bytes]) -> bytes:
    files = {f"nvidia/{package}/bin/{name}": data for name, data in dlls.items()}
    files[f"nvidia/{package}/include/{package}.h"] = b"// header"
    files[f"nvidia_{package}-1.0.dist-info/RECORD"] = b""
    return zip_bytes(files)


@pytest.fixture
def fake_wheels(server: Files, monkeypatch: pytest.MonkeyPatch) -> tuple[Wheel, ...]:
    wheels = []
    for package, dlls in (
        ("cublas", {"cublas64_12.dll": payload(MB), "cublasLt64_12.dll": payload(2 * MB)}),
        ("cudnn", {"cudnn64_9.dll": b"loader", "cudnn_ops64_9.dll": payload(MB)}),
    ):
        data = wheel_bytes(package, dlls)
        path = f"aa/bb/nvidia_{package}_cu12-1.0-py3-none-win_amd64.whl"
        server.add(f"/packages/{path}", data)
        wheels.append(
            Wheel(f"nvidia-{package}-cu12", "1.0", path, hashlib.sha256(data).hexdigest(), len(data))
        )
    monkeypatch.setattr(models, "PYPI_FILES", f"{server.url}/packages/")
    monkeypatch.setattr(models, "CUDA_WHEELS", tuple(wheels))
    return tuple(wheels)


def test_extract_dlls_takes_only_bin_dlls(tmp_path: Path) -> None:
    wheel = tmp_path / "x.whl"
    wheel.write_bytes(wheel_bytes("cudnn", {"cudnn64_9.dll": b"loader"}))
    assert extract_dlls(wheel, tmp_path / "bin") == ["cudnn64_9.dll"]
    assert [p.name for p in (tmp_path / "bin").iterdir()] == ["cudnn64_9.dll"]
    empty = tmp_path / "empty.whl"
    empty.write_bytes(zip_bytes({"nvidia/x/include/x.h": b""}))
    with pytest.raises(OSError, match="нет DLL"):
        extract_dlls(empty, tmp_path / "bin")


def test_ensure_cuda_puts_dlls_next_to_program(
    fake_wheels: tuple[Wheel, ...], server: Files, tmp_path: Path
) -> None:
    target = tmp_path / "cuda"
    window = Window()
    ensure_cuda(target, window)
    assert sorted(p.name for p in target.iterdir()) == [
        "cublas64_12.dll",
        "cublasLt64_12.dll",
        "cudnn64_9.dll",
        "cudnn_ops64_9.dll",
        "wheels.txt",
    ]
    assert (target / "cublasLt64_12.dll").read_bytes() == payload(2 * MB)
    assert (target / "wheels.txt").read_text().split() == [w.filename for w in fake_wheels]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cuda"]  # cuda.download убрана
    assert window.steps[-1][1] == 1.0 and window.steps[-1][0].startswith("Скачиваю библиотеки CUDA: 100%")
    # Второй раз ничего не скачивается.
    requests = len(server.requests)
    ensure_cuda(target, window)
    assert len(server.requests) == requests
    assert "CUDA: библиотеки уже есть" in window.getvalue()


def test_ensure_cuda_continues_where_it_stopped(
    fake_wheels: tuple[Wheel, ...], server: Files, tmp_path: Path
) -> None:
    first, second = fake_wheels
    target, staging = tmp_path / "cuda", tmp_path / "cuda.download"
    target.mkdir()
    (target / "wheels.txt").write_text(f"{first.filename}\n")
    (target / "cudnn64_9.dll").write_bytes(b"user's own")  # уже лежит — не заменяется
    staging.mkdir()
    data = server.files[f"/packages/{second.path}"]
    (staging / f"{second.filename}.part").write_bytes(data[:1000])
    out = io.StringIO()
    ensure_cuda(target, out)
    assert server.requests == [(f"/packages/{second.path}", "bytes=1000-")]
    assert (target / "cudnn64_9.dll").read_bytes() == b"user's own"
    assert (target / "cudnn_ops64_9.dll").read_bytes() == payload(MB)
    assert not (target / "cublas64_12.dll").exists()  # первое колесо уже было «установлено»
    assert (target / "wheels.txt").read_text().split() == [first.filename, second.filename]
    assert not staging.exists()
    assert f"Скачиваю библиотеки CUDA для видеокарты ({megabytes(second.size)}, один раз)" in out.getvalue()


def test_ensure_cuda_checks_disk_space_and_checksum(
    fake_wheels: tuple[Wheel, ...], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "cuda"
    real = models.shutil.disk_usage
    monkeypatch.setattr(models.shutil, "disk_usage", lambda path: real(path)._replace(free=100 * MB))
    with pytest.raises(OSError, match="мало места на диске: нужно 2.. МБ, свободно 100 МБ"):
        ensure_cuda(target, io.StringIO())
    monkeypatch.setattr(models.shutil, "disk_usage", real)

    broken = (fake_wheels[0], dataclasses.replace(fake_wheels[1], sha256="0" * 64))
    monkeypatch.setattr(models, "CUDA_WHEELS", broken)
    with pytest.raises(OSError, match="контрольная сумма"):
        ensure_cuda(target, io.StringIO())
    # Первое колесо осталось распакованным, второе скачается заново.
    assert (target / "wheels.txt").read_text().split() == [fake_wheels[0].filename]
    assert not [p for p in (tmp_path / "cuda.download").iterdir() if p.is_file()]


def test_download_cuda_checks_that_dlls_load(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models, "ensure_cuda", lambda target, out: None)
    monkeypatch.setattr(models, "missing_cuda_library", lambda: "cudnn_ops64_9.dll")
    out = io.StringIO()
    assert models.download_cuda(out) is False
    assert "CUDA: библиотека cudnn_ops64_9.dll не загружается" in out.getvalue()
    monkeypatch.setattr(models, "missing_cuda_library", lambda: "")
    assert models.download_cuda(out) is True

    def offline(target: Path, out: Any) -> None:
        raise OSError("нет сети")

    monkeypatch.setattr(models, "ensure_cuda", offline)
    assert models.download_cuda(out) is False
    assert "CUDA: ошибка — нет сети" in out.getvalue()


@pytest.mark.parametrize(("frozen", "needed", "steps"), [
    (True, True, ["Vosk", "Piper", "CUDA", "Whisper"]),
    (True, False, ["Vosk", "Piper", "Whisper"]),
    (False, True, ["Vosk", "Piper", "Whisper"]),  # из исходников — uv sync --extra cuda
])  # fmt: skip
def test_download_all_adds_cuda_only_to_exe(
    monkeypatch: pytest.MonkeyPatch, frozen: bool, needed: bool, steps: list[str]
) -> None:
    done: list[str] = []
    monkeypatch.setattr(models, "FROZEN", frozen)
    monkeypatch.setattr(models, "cuda_libraries_needed", lambda config: needed)
    monkeypatch.setattr(models, "ensure_vosk", lambda *a: done.append("Vosk"))
    monkeypatch.setattr(models, "ensure_piper", lambda *a: done.append("Piper"))
    monkeypatch.setattr(models, "ensure_cuda", lambda *a: done.append("CUDA"))
    monkeypatch.setattr(models, "ensure_whisper", lambda *a: done.append("Whisper"))
    assert models.download_all(AppConfig(), io.StringIO()) is True
    assert done == steps
