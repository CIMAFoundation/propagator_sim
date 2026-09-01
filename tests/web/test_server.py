from __future__ import annotations

import sys
import types

from propagator.web import server


def _run_main_capturing_uvicorn(monkeypatch):
    """Call `server.main()` with a stub uvicorn, returning the kwargs it
    would have been started with. `main` imports uvicorn lazily, so the
    stub goes into sys.modules rather than onto the module."""
    captured = {}

    fake_uvicorn = types.ModuleType("uvicorn")
    fake_uvicorn.run = lambda app, **kwargs: captured.update(app=app, **kwargs)
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)

    server.main()
    return captured


def test_binds_to_localhost_by_default(monkeypatch):
    """The app has no authentication and no per-user isolation, so
    network exposure must be opt-in rather than the default."""
    monkeypatch.delenv("PROPAGATOR_WEB_HOST", raising=False)
    monkeypatch.delenv("PROPAGATOR_WEB_PORT", raising=False)

    captured = _run_main_capturing_uvicorn(monkeypatch)

    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8765


def test_host_and_port_are_opt_in_via_environment(monkeypatch):
    monkeypatch.setenv("PROPAGATOR_WEB_HOST", "0.0.0.0")
    monkeypatch.setenv("PROPAGATOR_WEB_PORT", "9000")

    captured = _run_main_capturing_uvicorn(monkeypatch)

    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 9000
