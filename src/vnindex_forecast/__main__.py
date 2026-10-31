"""Module entry point: ``python -m vnindex_forecast <command>``."""

from __future__ import annotations

import sys

from vnindex_forecast.cli import main

if __name__ == "__main__":
    sys.exit(main())
