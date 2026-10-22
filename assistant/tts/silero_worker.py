"""Процесс синтеза Silero (запуск с --silero-worker): PyTorch и модель — только здесь.

PyTorch и CTranslate2 (Whisper) приносят каждый свою библиотеку OpenMP (libiomp5md.dll),
и в одном процессе они мешают друг другу. Поэтому torch загружается в отдельном процессе,
а этот модуль импортирует только стандартную библиотеку — до torch процесс не трогает
ничего из ассистента.

Протокол (stdin/stdout, строки JSON в UTF-8):
1. ассистент пишет настройки: {"model": путь .pt, "lib": папка с torch или "", "threads": N};
2. процесс отвечает {"ok": true, "torch": версия} — или {"ok": false, "error": …} и выходит;
3. дальше на каждый запрос {"text", "speaker", "rate", "sample_rate"} — ответ
   {"ok": true, "samples": N} и сразу N чисел float32 (little-endian) или {"ok": false, "error"}.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any, BinaryIO

MAX_CHARS = 800  # длиннее модель за один раз не читает — фраза делится по знакам препинания
_PIECE_RE = re.compile(rf".{{1,{MAX_CHARS}}}(?:[.,!?;:]\s|$)|.{{1,{MAX_CHARS}}}(?:\s|$)", re.DOTALL)


def pieces(text: str) -> list[str]:
    """Текст кусками не длиннее MAX_CHARS, по знакам препинания или пробелам."""
    return [piece.strip() for piece in _PIECE_RE.findall(text) if piece.strip()]


def _send(stdout: BinaryIO, header: dict[str, Any], data: bytes = b"") -> None:
    stdout.write(json.dumps(header, ensure_ascii=False).encode("utf-8") + b"\n" + data)
    stdout.flush()


def _synthesize(model: Any, request: dict[str, Any]) -> bytes:
    import torch

    speaker, sample_rate = request["speaker"], int(request["sample_rate"])
    rate = request.get("rate", "medium")
    parts = []
    for piece in pieces(request["text"]):
        if rate != "medium":
            ssml = f'<speak><prosody rate="{rate}">{piece}</prosody></speak>'
            audio = model.apply_tts(ssml_text=ssml, speaker=speaker, sample_rate=sample_rate)
        else:
            audio = model.apply_tts(text=piece, speaker=speaker, sample_rate=sample_rate)
        parts.append(audio.reshape(-1).to(torch.float32))
    if not parts:
        return b""
    return torch.cat(parts).numpy().astype("<f4").tobytes()


def run_worker(stdin: BinaryIO, stdout: BinaryIO) -> int:
    """Загружает модель и синтезирует фразы, пока ассистент не закроет stdin."""
    try:
        setup = json.loads(stdin.readline() or b"{}")
        if setup.get("lib"):
            sys.path.insert(0, setup["lib"])
        import torch
        from torch.package import PackageImporter

        torch.set_num_threads(max(1, int(setup.get("threads") or 4)))
        model = PackageImporter(setup["model"]).load_pickle("tts_models", "model")
        model.to(torch.device("cpu"))
    except Exception as exc:
        _send(stdout, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1
    _send(stdout, {"ok": True, "torch": torch.__version__})
    for line in stdin:
        try:
            data = _synthesize(model, json.loads(line))
        except Exception as exc:
            _send(stdout, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            continue
        _send(stdout, {"ok": True, "samples": len(data) // 4}, data)
    return 0
