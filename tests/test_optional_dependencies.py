from __future__ import annotations

import subprocess
import sys


def test_core_package_imports_without_pyproj() -> None:
    """The core install must not require dependencies from optional extras."""
    script = """
import builtins

real_import = builtins.__import__

def without_pyproj(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "pyproj" or name.startswith("pyproj."):
        raise ModuleNotFoundError("No module named 'pyproj'", name="pyproj")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = without_pyproj
import propagator
assert propagator.Propagator.__name__ == "Propagator"
"""
    subprocess.run([sys.executable, "-c", script], check=True)
