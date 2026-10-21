"""Короткие звуковые сигналы."""

from __future__ import annotations

import numpy as np

SOUND_RATE = 22_050


def _tone(frequency: float, duration: float, volume: float) -> np.ndarray:
    t = np.arange(int(SOUND_RATE * duration)) / SOUND_RATE
    tone = np.sin(2 * np.pi * frequency * t) * volume
    fade = min(len(tone) // 4, int(SOUND_RATE * 0.01))
    envelope = np.ones_like(tone)
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)
    return (tone * envelope).astype(np.float32)


def listen_chime() -> np.ndarray:
    """Сигнал «слушаю»: два коротких восходящих тона (~150 мс)."""
    return np.concatenate([_tone(660, 0.07, 0.25), _tone(880, 0.08, 0.25)])
