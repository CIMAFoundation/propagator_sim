"""Entry point for `uv run propagator-web`. Binds to localhost only —
this is a local, single-user tool, not meant to be exposed on a network."""

from __future__ import annotations


def main() -> None:
    import uvicorn

    uvicorn.run("propagator.web.app:app", host="127.0.0.1", port=8765)


if __name__ == "__main__":
    main()
