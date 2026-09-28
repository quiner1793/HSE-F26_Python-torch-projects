"""Запуск базового задания на 5 пород: python main.py."""

from importlib import import_module


def main():
    # Поддерживаем python main.py и python -m lecture_02.main.
    package = f"{__package__}.experiments" if __package__ else "experiments"
    if not __package__:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    return import_module(f"{package}.01_baseline_5").main()


if __name__ == "__main__":
    main()
