from __future__ import annotations

import atexit
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional

from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
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
    Setup the global console for the CLI.
    The console will always print to terminal, regardless of recording.
    Parameters
    ----------
    record_path: str|Path|None
        Optional. Enables recording and writes HTML/log at exit.

    basename: str
        Base name for exported files (without extension).
    export_html: bool
        If True, export HTML file.
    export_text: bool
        If True, export plain text log file.

    Returns
    -------
    Console
        The global Console instance.
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
    Parameters
    ----------
    init_date: datetime
        Simulation initial date.
    time: int
        Current simulation time in seconds.
    stats: PropagatorStats
        Current simulation statistics.
    """
    date_str = (init_date + timedelta(seconds=time)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    msg = (
        f"Time: {timedelta(seconds=time)!s:>8} | "
        f"Date: {date_str} | "
        f"Active: {stats.n_active:>3} | "
        f"Mean area: {(stats.area_mean / 10000):>7.2f} ha | "
        f"Area 50%: {(stats.area_50 / 10000):>7.2f} ha | "
        f"Area 75%: {(stats.area_75 / 10000):>7.2f} ha | "
        f"Area 90%: {(stats.area_90 / 10000):>7.2f} ha"
    )
    get_console().print(msg)


# ---------- printers ----------


def _geom_to_custom_str(geom: Any) -> str:
    """
    Convert a geometry object to a custom string representation.
    Return geometry as 'TYPE:[y1 y2 ...];[x1 x2 ...]'.

    Parameters
    ----------
    geom: Any
        Geometry object (Point, LineString, Polygon).

    Returns
    -------
    str
        Custom string representation of the geometry.
    """

    def _yx_lists_from_coords(coords):
        xs, ys = zip(*coords)  # shapely gives (x, y)
        # we want y first then x
        ys_s = " ".join(f"{v:.15f}".rstrip("0").rstrip(".") for v in ys)
        xs_s = " ".join(f"{v:.15f}".rstrip("0").rstrip(".") for v in xs)
        return ys_s, xs_s

    # Point
    if hasattr(geom, "geom_type") and geom.geom_type == "Point":
        x, y = geom.x, geom.y
        y_s = f"{y:.15f}".rstrip("0").rstrip(".")
        x_s = f"{x:.15f}".rstrip("0").rstrip(".")
        return f"POINT:[{y_s};{x_s}]"

    # LineString
    if hasattr(geom, "geom_type") and geom.geom_type == "LineString":
        ys_s, xs_s = _yx_lists_from_coords(geom.coords)
        return f"LINE:[{ys_s}];[{xs_s}]"

    # Polygon (use exterior ring)
    if hasattr(geom, "geom_type") and geom.geom_type == "Polygon":
        ys_s, xs_s = _yx_lists_from_coords(geom.exterior.coords)
        return f"POLYGON:[{ys_s}];[{xs_s}]"

    # Fallback to str()
    return str(geom)


def _format_geoms_custom(geoms: list[Any]) -> str:
    """
    Format a list of geometries as a custom string representation.
    """
    if not geoms:
        return "-"
    s = "[" + " , ".join(_geom_to_custom_str(g) for g in geoms) + "]"
    return s


def _format_actions(actions: list[Any]) -> str:
    """
    Format a list of Action objects as a custom string representation.
    Each action is represented by its type and geometries.

    Parameters
    ----------
    actions: list[Any]
        List of Action objects.
    Returns
    -------
    str
        Custom string representation of the actions.
    """
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


def print_table(
    models: dict[str, BaseModel | dict[str, Any]],
    *,
    title: Optional[str] = "📋 Models",
    skip_none: bool = False,
    skip_fields: Optional[list[str] | dict[str, list[str]]] = None,
    sort_fields: bool = True,
    section_style: str = "bold magenta",
    header_style: str = "bold blue",
    zebra: bool = False,
) -> None:
    """
    Print a single Rich table with a faux-rowspan 'Section' column,
    and two data columns: 'field' and 'value'.
    Each entry in `models` becomes a section.

    Parameters
    ----------
    models: dict[str, BaseModel|dict]
        Mapping of section title -> BaseModel or dict of fields.
    title: str|None
        Optional table title.
    skip_none: bool
        If True, omit fields whose value is None.
    skip_fields: list[str]|dict[str, list[str]]
        List or dict of fields to skip. Can be:
            - A list: applies globally to all sections.
            - A dict: mapping of section name -> list of fields to skip.
    sort_fields: bool
        If True, sort fields by display name.
    section_style: str
        Rich style for the section label cell.
    header_style: str
        Rich style for the header row.
    zebra: bool
        If True, apply gentle zebra striping to rows.

    Returns:
        None
    """

    table = Table(
        title=title,
        header_style=header_style,
        show_lines=True,
        row_styles=("none", "dim") if zebra else None,
    )
    table.add_column("Section", style=section_style, no_wrap=True)
    table.add_column("field", no_wrap=True)
    table.add_column("value", overflow="fold")

    def iter_fields(
        obj: BaseModel | dict[str, Any],
    ) -> Iterable[tuple[str, Any]]:
        """Yield (field_name, value) pairs for either BaseModel or dict."""
        if isinstance(obj, BaseModel):
            # Pydantic v2
            if hasattr(obj, "model_fields"):
                fields = obj.model_fields
                items = []
                for name, f in fields.items():
                    alias = getattr(f, "alias", None)
                    display = alias or name
                    items.append((display, getattr(obj, name)))
                return (
                    sorted(items, key=lambda x: x[0].lower())
                    if sort_fields
                    else items
                )

            # Pydantic v1
            elif hasattr(obj, "__fields__"):
                fields = obj.__fields__
                items = []
                for name, f in fields.items():
                    alias = getattr(f, "alias", None)
                    display = alias or name
                    items.append((display, getattr(obj, name)))
                return (
                    sorted(items, key=lambda x: x[0].lower())
                    if sort_fields
                    else items
                )

            else:
                data = getattr(obj, "__dict__", {})
                return (
                    sorted(data.items(), key=lambda x: str(x[0]).lower())
                    if sort_fields
                    else data.items()
                )

        elif isinstance(obj, dict):
            items = list(obj.items())
            return (
                sorted(items, key=lambda x: str(x[0]).lower())
                if sort_fields
                else items
            )

        else:
            raise TypeError(f"Unsupported model type: {type(obj).__name__}")

    for section, model in models.items():
        section_skip: list[str] = []

        # Determine fields to skip for this section
        if isinstance(skip_fields, dict):
            section_skip = skip_fields.get(section, [])
        elif isinstance(skip_fields, list):
            section_skip = skip_fields

        rows = [
            (fname, fval)
            for fname, fval in iter_fields(model)
            if fname not in section_skip and not (skip_none and fval is None)
        ]

        if not rows:
            table.add_row(
                section, "[dim](no fields)[/dim]", "", end_section=True
            )
            continue

        for i, (fname, fval) in enumerate(rows):
            section_cell = section if i == 0 else ""
            is_last = i == len(rows) - 1
            value_cell = "—" if fval is None else Pretty(fval, overflow="fold")
            table.add_row(
                section_cell, str(fname), value_cell, end_section=is_last
            )
    get_console().print(table)


def print_boundary_conditions_table(
    bcs: Iterable[Any],
    *,
    title="Boundary Conditions",
):
    """
    Print a table summarizing the boundary conditions.
    Parameters
    ----------
    bcs: Iterable[Any]
        An iterable of TimedInput with attributes:
         time, w_dir, w_speed, moisture, actions, ignitions
    """

    table = Table(title=title, show_lines=True)
    table.add_column(
        "time [s]", justify="right", style="bold cyan", no_wrap=True
    )
    table.add_column("w_dir [°]", justify="right", no_wrap=True)
    table.add_column("w_speed [km/h]", justify="right", no_wrap=True)
    table.add_column("moisture [%]", justify="right", no_wrap=True)
    table.add_column("actions", overflow="fold", style="magenta")
    table.add_column("ignitions", overflow="fold", style="green")

    for ti in bcs:
        time = getattr(ti, "time", "-")
        w_dir = getattr(ti, "w_dir", None)
        w_speed = getattr(ti, "w_speed", None)
        moisture = getattr(ti, "moisture", None)
        actions = getattr(ti, "actions", None)
        igns = getattr(ti, "ignitions", None)

        actions_cell = _format_actions(actions)  # type: ignore
        igns_cell = "-" if not igns else _format_geoms_custom(igns)

        table.add_row(
            str(time),
            "-" if w_dir is None else f"{w_dir:g}",
            "-" if w_speed is None else f"{w_speed:g}",
            "-" if moisture is None else f"{moisture:g}",
            actions_cell,
            igns_cell,
        )
    get_console().print(table)
