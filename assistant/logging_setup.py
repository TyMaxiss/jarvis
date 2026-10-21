"""Настройка логирования. Ключи API маскируются во всех записях лога и журнала."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable, Mapping
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Переменные окружения с такими именами считаются секретами.
_SECRET_ENV_RE = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD)", re.IGNORECASE)
# Похожие на ключи строки маскируются, даже если их нет в окружении.
_KEY_LIKE_RE = re.compile(r"\b(sk|pk|rk)-[A-Za-z0-9_\-]{16,}\b")
_MIN_SECRET_LEN = 6
_MASK = "***"


class SecretMasker:
    """Заменяет известные секреты на «***» в любом тексте."""

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        self._secrets: set[str] = set()
        for secret in secrets:
            self.add(secret)

    @classmethod
    def from_environ(cls, environ: Mapping[str, str] | None = None) -> SecretMasker:
        """Собирает секреты из переменных окружения (*_KEY, *_TOKEN, *_SECRET, *PASSWORD*)."""
        environ = os.environ if environ is None else environ
        return cls(value for name, value in environ.items() if _SECRET_ENV_RE.search(name))

    def add(self, secret: str) -> None:
        """Добавляет секрет; слишком короткие строки игнорируются, чтобы не портить текст."""
        if secret and len(secret) >= _MIN_SECRET_LEN:
            self._secrets.add(secret)

    def mask(self, text: str) -> str:
        """Возвращает текст с замаскированными секретами."""
        for secret in self._secrets:
            if secret in text:
                text = text.replace(secret, _MASK)
        return _KEY_LIKE_RE.sub(_MASK, text)


class MaskingFormatter(logging.Formatter):
    """Форматтер, маскирующий секреты в итоговой строке, включая traceback."""

    def __init__(self, fmt: str, masker: SecretMasker) -> None:
        super().__init__(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")
        self._masker = masker

    def format(self, record: logging.LogRecord) -> str:
        return self._masker.mask(super().format(record))


def setup_logging(log_file: Path, level: str, masker: SecretMasker, console_level: str = "WARNING") -> None:
    """Настраивает корневой логгер: файл с ротацией и вывод в консоль.

    Args:
        log_file: путь к файлу лога (папка создаётся).
        level: уровень для файла.
        masker: маскировщик секретов.
        console_level: уровень для консоли (в консольном режиме лишний шум мешает).
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    file_handler = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=5, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(MaskingFormatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", masker))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(MaskingFormatter("%(levelname)s %(name)s: %(message)s", masker))

    levels = logging.getLevelNamesMapping()
    root.setLevel(min(levels[level], levels[console_level]))
    root.addHandler(file_handler)
    root.addHandler(console_handler)
    # Сторонние библиотеки слишком многословны на уровне DEBUG.
    for noisy in ("comtypes", "urllib3", "httpx", "httpcore", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
