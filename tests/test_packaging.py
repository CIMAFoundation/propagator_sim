"""Verify the built wheel ships the web UI's static frontend assets.

Regression test for a packaging bug: `pyproject.toml` had
`include-package-data = false` and no explicit `[tool.setuptools.package-data]`
entry for `propagator.web`, so `static/` (index.html, app.js, manual.html,
...) was silently dropped from the wheel. `propagator.web.app` mounts that
directory at import time (`STATIC_DIR = Path(__file__).parent / "static"`),
so a packaged (non-editable) install raised `RuntimeError: Directory
'.../propagator/web/static' does not exist` the moment the app module was
imported.
"""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

STATIC_DIR = REPO_ROOT / "src" / "propagator" / "web" / "static"

# A minimum set, asserted by name so the test still fails loudly if the
# source tree itself were empty/missing. The real check below compares
# against every file actually present under static/, including
# subdirectories -- a fixed list is what let the locales/ bundles ship
# broken: they were added under a new subdirectory that the
# (non-recursive) `static/*` package-data glob silently skipped, and no
# expectation here mentioned them.
EXPECTED_STATIC_FILES = {
    "index.html",
    "app.js",
    "style.css",
    "manual.html",
    "manual.css",
    "i18n.js",
}


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory) -> Path:
    out_dir = tmp_path_factory.mktemp("wheel")
    # setuptools stages package_data copies under <repo>/build/lib on the
    # way into the wheel and reuses that directory across invocations, so
    # a stale build/ left over from a previous (correctly configured)
    # build would mask a `package-data` regression here; force a clean
    # build every time.
    shutil.rmtree(REPO_ROOT / "build", ignore_errors=True)
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(out_dir)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = list(out_dir.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"
    return wheels[0]


def test_wheel_includes_web_static_assets(built_wheel: Path) -> None:
    with zipfile.ZipFile(built_wheel) as zf:
        names = zf.namelist()

    static_files = {
        Path(n).name
        for n in names
        if "propagator/web/static/" in n.replace("\\", "/")
    }
    missing = EXPECTED_STATIC_FILES - static_files
    assert not missing, (
        f"wheel is missing web UI static assets: {sorted(missing)} "
        f"(found under static/: {sorted(static_files)})"
    )


def test_wheel_includes_every_static_asset_including_subdirectories(
    built_wheel: Path,
) -> None:
    """Compare against the source tree rather than a hand-kept list, so a
    newly added asset -- or a whole new asset subdirectory -- is covered
    automatically. `static/locales/{en,it}.json` shipped missing exactly
    because nothing enumerated them: the app then 404s on every locale
    fetch and renders raw i18n keys instead of text."""
    expected = {
        p.relative_to(STATIC_DIR).as_posix()
        for p in STATIC_DIR.rglob("*")
        if p.is_file()
    }
    assert expected, f"no static assets found under {STATIC_DIR}"

    with zipfile.ZipFile(built_wheel) as zf:
        names = [n.replace("\\", "/") for n in zf.namelist()]

    prefix = "propagator/web/static/"
    shipped = {n[len(prefix) :] for n in names if n.startswith(prefix)}

    missing = expected - shipped
    assert not missing, (
        f"wheel is missing web UI static assets: {sorted(missing)} "
        f"(shipped: {sorted(shipped)})"
    )
