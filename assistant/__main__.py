"""Запуск: python -m assistant"""

import sys

if "--silero-worker" in sys.argv[1:]:
    # Процесс синтеза Silero: до torch — только стандартная библиотека, без всего ассистента.
    from assistant.tts.silero_worker import run_worker

    raise SystemExit(run_worker(sys.stdin.buffer, sys.stdout.buffer))

from assistant.main import main

raise SystemExit(main())
