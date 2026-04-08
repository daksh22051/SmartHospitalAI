from __future__ import annotations

import os

from app import app


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))


# Allow running this module directly for validators that execute
# `python server/app.py` and expect the server to start.
if __name__ == "__main__":
    main()
