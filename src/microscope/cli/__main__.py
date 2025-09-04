from .main import main as _main

if __name__ == "__main__":
    raise SystemExit(_main())

"""CLI entry point for running as module."""

from .main import main

if __name__ == "__main__":
    import sys

    sys.exit(main())
