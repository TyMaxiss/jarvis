"""Точка входа собранного Jarvis.exe и JarvisConsole.exe (PyInstaller)."""

import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()  # библиотеки могут запускать процессы через multiprocessing
    from assistant.main import main

    sys.exit(main())
