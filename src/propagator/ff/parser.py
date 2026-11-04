from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Union

# `CommandName[arg1=val1;arg2=val2]` optionally followed by a schedule
# operator: `@t=<seconds>` or `@nowplus=<seconds>`.
_COMMAND_RE = re.compile(
    r"""^(?P<name>[A-Za-z_][A-Za-z0-9_]*)
        \[(?P<body>.*)\]
        (?:@(?P<sched_kind>t|nowplus)=(?P<sched_val>[-+]?\d+(?:\.\d+)?))?
        \s*$""",
    re.VERBOSE,
)

_TUPLE_RE = re.compile(
    r"^\((?P<values>[^()]*)\)$",
)


class FFParseError(ValueError):
    """Raised when a `.ff` script line cannot be parsed."""


@dataclass
class Schedule:
    kind: str  # "t" | "nowplus"
    value: float


@dataclass
class Command:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    positional: list[Any] = field(default_factory=list)
    schedule: Optional[Schedule] = None
    indent: int = 0
    raw: str = ""
    line_no: int = 0

    def require(self, *keys: str) -> tuple:
        missing = [k for k in keys if k not in self.args]
        if missing:
            raise FFParseError(
                f"{self.name}[]: missing required argument(s) "
                f"{', '.join(missing)} (line {self.line_no}: {self.raw!r})"
            )
        return tuple(self.args[k] for k in keys)


def _coerce_scalar(text: str) -> Union[int, float, str]:
    text = text.strip()
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _coerce_value(text: str) -> Any:
    text = text.strip()
    m = _TUPLE_RE.match(text)
    if m:
        parts = [p for p in m.group("values").split(",")]
        return tuple(_coerce_scalar(p) for p in parts)
    return _coerce_scalar(text)


def _split_top_level(body: str) -> list[str]:
    """Split `body` on ';' that are not nested inside parentheses."""
    tokens: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in body:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == ";" and depth == 0:
            tokens.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current or tokens:
        tokens.append("".join(current))
    return [t for t in tokens if t.strip() != ""]


def parse_line(line: str, *, line_no: int = 0) -> Optional[Command]:
    """Parse a single `.ff` script line into a `Command`, or None for a
    blank/comment-only line."""
    indent = len(line) - len(line.lstrip(" "))
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    # strip trailing inline comments (`# ...`) that aren't inside brackets
    stripped = _strip_trailing_comment(stripped)

    m = _COMMAND_RE.match(stripped)
    if not m:
        raise FFParseError(f"line {line_no}: cannot parse command: {line!r}")

    body = m.group("body")
    args: dict[str, Any] = {}
    positional: list[Any] = []
    for token in _split_top_level(body):
        if "=" in token:
            key, _, value = token.partition("=")
            args[key.strip()] = _coerce_value(value)
        else:
            positional.append(_coerce_value(token))

    schedule = None
    if m.group("sched_kind") is not None:
        schedule = Schedule(
            kind=m.group("sched_kind"), value=float(m.group("sched_val"))
        )

    return Command(
        name=m.group("name"),
        args=args,
        positional=positional,
        schedule=schedule,
        indent=indent,
        raw=line.rstrip("\n"),
        line_no=line_no,
    )


def _strip_trailing_comment(stripped: str) -> str:
    depth = 0
    for i, ch in enumerate(stripped):
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        elif ch == "#" and depth == 0:
            return stripped[:i].rstrip()
    return stripped


def parse_lines(lines: Iterator[str]) -> Iterator[Command]:
    for i, line in enumerate(lines, start=1):
        cmd = parse_line(line, line_no=i)
        if cmd is not None:
            yield cmd


def parse_script(text: str) -> list[Command]:
    return list(parse_lines(text.splitlines()))
