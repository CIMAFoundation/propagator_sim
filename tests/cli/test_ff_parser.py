import pytest

from propagator.ff.parser import FFParseError, parse_line, parse_script


def test_parse_simple_command():
    cmd = parse_line("startFire[loc=(0.0,0.0,0.0);t=0.]", line_no=1)
    assert cmd is not None
    assert cmd.name == "startFire"
    assert cmd.args["loc"] == (0.0, 0.0, 0.0)
    assert cmd.args["t"] == 0.0
    assert cmd.schedule is None


def test_parse_positional_args():
    cmd = parse_line("loadData[data.nc;2025-02-10T17:35:54Z]", line_no=1)
    assert cmd is not None
    assert cmd.name == "loadData"
    assert cmd.positional == ["data.nc", "2025-02-10T17:35:54Z"]
    assert cmd.args == {}


def test_parse_schedule_t():
    cmd = parse_line(
        "trigger[fuelType=wind;vel=(5.0,2.0,0.0)]@t=1800", line_no=1
    )
    assert cmd is not None
    assert cmd.args["fuelType"] == "wind"
    assert cmd.args["vel"] == (5.0, 2.0, 0.0)
    assert cmd.schedule is not None
    assert cmd.schedule.kind == "t"
    assert cmd.schedule.value == 1800.0


def test_parse_schedule_nowplus():
    cmd = parse_line("save[]@nowplus=600", line_no=1)
    assert cmd is not None
    assert cmd.schedule is not None
    assert cmd.schedule.kind == "nowplus"
    assert cmd.schedule.value == 600.0


def test_parse_no_args():
    cmd = parse_line("print[]", line_no=1)
    assert cmd is not None
    assert cmd.name == "print"
    assert cmd.args == {}
    assert cmd.positional == []


def test_blank_and_comment_lines_are_skipped():
    assert parse_line("") is None
    assert parse_line("   ") is None
    assert parse_line("# a comment") is None


def test_trailing_inline_comment_is_stripped():
    cmd = parse_line("step[dt=600]  # advance ten minutes", line_no=1)
    assert cmd is not None
    assert cmd.args["dt"] == 600


def test_indentation_is_recorded_but_not_required():
    cmd = parse_line("    FireFront[id=26;domain=0;t=0]", line_no=1)
    assert cmd is not None
    assert cmd.indent == 4
    assert cmd.name == "FireFront"


def test_bbox_wsen_tuple():
    cmd = parse_line(
        "FireDomain[sw=(0,0,0);ne=(64000,64000,0);"
        "BBoxWSEN=(8.36215875,41.711125,9.1366311,42.28667);t=2400]",
        line_no=1,
    )
    assert cmd is not None
    assert cmd.args["sw"] == (0, 0, 0)
    assert cmd.args["BBoxWSEN"] == (8.36215875, 41.711125, 9.1366311, 42.28667)
    assert cmd.args["t"] == 2400


def test_unparseable_line_raises():
    with pytest.raises(FFParseError):
        parse_line("not a valid command", line_no=1)


def test_require_missing_argument_raises():
    cmd = parse_line("step[]", line_no=1)
    assert cmd is not None
    with pytest.raises(FFParseError):
        cmd.require("dt")


def test_parse_script_multiple_lines():
    text = (
        "# header comment\n"
        "loadData[data.nc;2024-01-01T00:00:00Z]\n"
        "startFire[loc=(0,0,0);t=0]\n"
        "step[dt=600]\n"
    )
    cmds = parse_script(text)
    assert [c.name for c in cmds] == ["loadData", "startFire", "step"]
