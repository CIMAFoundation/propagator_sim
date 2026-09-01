"""Entry point for `uv run propagator-web`. Binds to localhost by default.

This is a single-user, unauthenticated tool (one simulation job at a time,
no login), so exposing it on a network is opt-in: set
`PROPAGATOR_WEB_HOST=0.0.0.0` to make it reachable from other machines on
the LAN, and only do so on a trusted network.
"""

from __future__ import annotations

import os

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def main() -> None:
    import uvicorn

    host = os.environ.get("PROPAGATOR_WEB_HOST", DEFAULT_HOST)
    port = int(os.environ.get("PROPAGATOR_WEB_PORT", DEFAULT_PORT))
    uvicorn.run("propagator.web.app:app", host=host, port=port)


if __name__ == "__main__":
    main()
