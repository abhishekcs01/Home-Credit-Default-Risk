#!/usr/bin/env python3
"""Regenerate all reports from saved metrics without retraining."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reporting.generate_reports import regenerate_reports_from_artifacts
from src.utils import init_logging


def main() -> None:
    init_logging()
    paths = regenerate_reports_from_artifacts()
    print(f"Regenerated {len(paths)} report artifacts.")


if __name__ == "__main__":
    main()
