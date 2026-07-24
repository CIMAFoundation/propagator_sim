"""Interpreter for a subset of forefire's `.ff` command language, driving
propagator's own raster/ensemble core instead of forefire's front-tracker.

See the command-mapping table in the project plan for which forefire verbs
are supported and why others (FireFront/FireNode, addLayer, plot,
listenHTTP, systemExec, parallel restarts) are not.
"""

from propagator.ff.interpreter import (
    ForeFireScriptRunner,
    UnsupportedCommandError,
)

__all__ = ["ForeFireScriptRunner", "UnsupportedCommandError"]
