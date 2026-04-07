"""Module entrypoint for `python -m smart_hospital_orchestration.inference`."""

from __future__ import annotations

import os
import runpy


def main() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    launcher = os.path.join(project_root, "inference.py")
    runpy.run_path(launcher, run_name="__main__")


if __name__ == "__main__":
    main()