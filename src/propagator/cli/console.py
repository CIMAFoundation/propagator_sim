from __future__ import annotations

import atexit
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.console import Console

# from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# from rich import box
from rich.traceback import install as rich_traceback_install

# Pretty tracebacks for unhandled exceptions
rich_traceback_install(show_locals=False)


# ---- Console singleton ------------------------------------------------------
_console: Optional[Console] = None


def get_console() -> Console:
    global _console
    if _console is None:
        # not recording by default; setup_console() can enable it
        _console = Console()
    return _console


# ---- Export configuration & atexit writer -----------------------------------
@dataclass
class _ExportConf:
    enabled: bool = False
    output_folder: Path = Path(".")
    basename: str = "propagator_run"
    export_html: bool = True
    export_text: bool = True


_export_conf: _ExportConf = _ExportConf()
_export_registered: bool = False


def _export_once() -> None:
    """
    Called at process exit by atexit; writes whatever is recorded.
    If recording was never turned on, nothing is exported.
    """
    c = get_console()
    if not _export_conf.enabled or not c.record:
        return

    outdir = _export_conf.output_folder
    outdir.mkdir(parents=True, exist_ok=True)

    if _export_conf.export_html:
        (outdir / f"{_export_conf.basename}.html").write_text(
            c.export_html(inline_styles=True), encoding="utf-8"
        )
    if _export_conf.export_text:
        (outdir / f"{_export_conf.basename}.log").write_text(
            c.export_text(), encoding="utf-8"
        )


# ---- Public: single entrypoint to set everything up -------------------------
def setup_console(
    *,
    record_path: str | Path | None = None,
    basename: str = "propagator_run",
    export_html: bool = True,
    export_text: bool = True,
) -> Console:
    """
    - If `record_path` is given, enables recording and writes HTML/log at exit.
    - Always prints to terminal, regardless of recording.
    """
    c = get_console()

    if record_path is not None:
        _export_conf.enabled = True
        _export_conf.output_folder = Path(record_path)
        _export_conf.basename = basename
        _export_conf.export_html = export_html
        _export_conf.export_text = export_text

        c.record = True  # start buffering everything printed from now on

        global _export_registered
        if not _export_registered:
            atexit.register(_export_once)
            _export_registered = True

    return c


# ---------- message helpers ----------
def info_msg(message: str) -> None:
    get_console().print(Text(message))


def ok_msg(message: str) -> None:
    get_console().print(Text(message, style="bold green"))


def warn_msg(message: str) -> None:
    get_console().print(Text(message, style="yellow"))


def error_msg(message: str) -> None:
    get_console().print(
        Panel.fit(Text(message, style="bold red"), border_style="red")
    )
