"""
Root-level launcher so `python inference.py` works from workspace root.
"""

from __future__ import annotations

import os
import runpy


def main() -> None:
    project_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "smart_hospital_orchestration",
        "inference.py",
    )
    runpy.run_path(project_script, run_name="__main__")


if __name__ == "__main__":
    main()

