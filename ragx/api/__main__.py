"""``python -m ragx.api`` - run the RAGX server."""

from __future__ import annotations

import uvicorn

from ragx.api.app import create_app

app = create_app()


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
