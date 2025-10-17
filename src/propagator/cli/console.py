from __future__ import annotations

import atexit
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional

from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.traceback import install as rich_traceback_install

from propagator.core.models import PropagatorStats

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
            c.export_html(inline_styles=True, clear=False), encoding="utf-8"
        )
    if _export_conf.export_text:
        (outdir / f"{_export_conf.basename}.log").write_text(
            c.export_text(clear=False), encoding="utf-8"
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


def status_propagator_msg(
    init_date: datetime, time: int, stats: PropagatorStats
) -> None:
    """
    Print a one-line status message with current time and stats.
    """
    date_str = (init_date + timedelta(seconds=time)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    msg = (
        f"Time: {timedelta(seconds=time)} | "
        f"{date_str} | "
        f"Active: {stats.n_active} | "
        f"Mean area: {(stats.area_mean / 10000):.2f} ha | "
        f"Area 50%: {(stats.area_50 / 10000):.2f} ha | "
        f"Area 75%: {(stats.area_75 / 10000):.2f} ha | "
        f"Area 90%: {(stats.area_90 / 10000):.2f} ha"
    )
    get_console().print(msg)


# ---------- printers ----------


def _geom_to_custom_str(g) -> str:
    """
    Return geometry as 'TYPE:[y1 y2 ...];[x1 x2 ...]'.
    """

    def _yx_lists_from_coords(coords):
        xs, ys = zip(*coords)  # shapely gives (x, y)
        # we want y first then x
        ys_s = " ".join(f"{v:.15f}".rstrip("0").rstrip(".") for v in ys)
        xs_s = " ".join(f"{v:.15f}".rstrip("0").rstrip(".") for v in xs)
        return ys_s, xs_s

    # Point
    if hasattr(g, "geom_type") and g.geom_type == "Point":
        x, y = g.x, g.y
        y_s = f"{y:.15f}".rstrip("0").rstrip(".")
        x_s = f"{x:.15f}".rstrip("0").rstrip(".")
        return f"POINT:[{y_s};{x_s}]"

    # LineString
    if hasattr(g, "geom_type") and g.geom_type == "LineString":
        ys_s, xs_s = _yx_lists_from_coords(g.coords)
        return f"LINE:[{ys_s}];[{xs_s}]"

    # Polygon (use exterior ring)
    if hasattr(g, "geom_type") and g.geom_type == "Polygon":
        ys_s, xs_s = _yx_lists_from_coords(g.exterior.coords)
        return f"POLYGON:[{ys_s}];[{xs_s}]"

    # Fallback to str()
    return str(g)


def _format_geoms_custom(geoms) -> str:
    if not geoms:
        return "-"
    s = "[" + " , ".join(_geom_to_custom_str(g) for g in geoms) + "]"
    return s


def _format_actions(actions) -> str:
    if not actions:
        return "-"
    lines = []
    for a in actions:
        a_type = (
            getattr(a, "action_type", type(a))
            .__str__()
            .split(".")[-1]
            .replace("ActionType.", "")
            .lower()
        )
        geoms = getattr(a, "geometries", None) or []
        geoms_str = _format_geoms_custom(geoms)
        # one block per action
        block = f"{a_type}: {geoms_str}"
        lines.append(block)
    return "\n".join(lines)


def print_model_table(cfg: BaseModel, *, title="Title"):
    console = get_console()
    table = Table(title=title, show_lines=False)
    table.add_column("Field", style="bold cyan")
    table.add_column("Value", style="magenta")

    for name, field in cfg.__class__.model_fields.items():
        if name == "boundary_conditions":
            continue  # printed separately
        # read current value directly from the instance
        value = getattr(cfg, name, None)
        if name == "ignitions":
            value_str = "-" if not value else _format_geoms_custom(value)
        else:
            value_str = str(value) if value is not None else "None"
        table.add_row(name, value_str)

    console.print(table)


def print_boundary_conditions_table(
    bcs: Iterable[Any],
    *,
    title="Boundary Conditions",
):
    """
    bcs: iterable of TimedInput with attributes:
         time, w_dir, w_speed, moisture, actions, ignitions
    """
    console = get_console()
    t = Table(title=title, show_lines=True)
    t.add_column("time [s]", justify="right", style="bold cyan", no_wrap=True)
    t.add_column("w_dir [°]", justify="right", no_wrap=True)
    t.add_column("w_speed [km/h]", justify="right", no_wrap=True)
    t.add_column("moisture [%]", justify="right", no_wrap=True)
    t.add_column("actions", overflow="fold", style="magenta")
    t.add_column("ignitions", overflow="fold", style="green")

    for ti in bcs:
        time = getattr(ti, "time", "-")
        w_dir = getattr(ti, "w_dir", None)
        w_speed = getattr(ti, "w_speed", None)
        moisture = getattr(ti, "moisture", None)
        actions = getattr(ti, "actions", None)
        igns = getattr(ti, "ignitions", None)

        actions_cell = _format_actions(actions)
        igns_cell = "-" if not igns else _format_geoms_custom(igns)

        t.add_row(
            str(time),
            "-" if w_dir is None else f"{w_dir:g}",
            "-" if w_speed is None else f"{w_speed:g}",
            "-" if moisture is None else f"{moisture:g}",
            actions_cell,
            igns_cell,
        )

    console.print(t)
