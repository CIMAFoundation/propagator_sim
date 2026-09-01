"""Entry point for `uv run propagator-web`. Binds to all interfaces so it
is reachable from other machines on the LAN — this is still a
single-user, unauthenticated tool (one simulation job at a time, no
login), so only run it on a trusted network."""

from __future__ import annotations


def main() -> None:
    import uvicorn

    uvicorn.run("propagator.web.app:app", host="0.0.0.0", port=8765)


if __name__ == "__main__":
    main()
