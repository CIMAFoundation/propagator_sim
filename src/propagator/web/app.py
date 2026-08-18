from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from propagator.web.routers import simulate

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="PROPAGATOR")
app.include_router(simulate.router)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
