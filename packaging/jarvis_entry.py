"""Точка входа собранного Jarvis.exe и JarvisConsole.exe (PyInstaller)."""

import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()  # библиотеки могут запускать процессы через multiprocessing
    if "--silero-worker" in sys.argv[1:]:
        # Процесс синтеза Silero (JarvisConsole.exe --silero-worker): ассистент целиком не нужен.
        from assistant.tts.silero_worker import run_worker

        sys.exit(run_worker(sys.stdin.buffer, sys.stdout.buffer))
    from assistant.main import main

    sys.exit(main())
