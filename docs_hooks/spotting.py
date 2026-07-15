"""MkDocs hook that keeps ``docs/spotting.md`` aligned with the code.

On every build it (1) regenerates the fire-spotting explanatory figures from
the live model constants, and (2) substitutes ``{{spotting.NAME}}`` tokens in
the spotting page with the current constant values and derived quantities, so
the constant table and the inline numbers can never drift from the source.

Registered via ``hooks:`` in ``mkdocs.yml`` (a native MkDocs mechanism, no
plugin required). Requires ``matplotlib`` and ``scipy`` (both already dev/docs
dependencies).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, for docs builds

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colormaps  # noqa: E402
from scipy.stats import lognorm  # noqa: E402

from propagator.core.numba.propagation import (  # noqa: E402
    LAMBDA_SPOTTING,
    P_C0,
    SPOTTING_ANISOTROPY,
    SPOTTING_DISTANCE_LOG_SIGMA,
    SPOTTING_DISTANCE_REF,
    SPOTTING_FLI_EXPONENT,
    SPOTTING_FLI_REF,
    SPOTTING_MIN_ALIGNMENT,
    SPOTTING_TIME_TO_PROPAGATION_MEDIAN,
    SPOTTING_WIND_REF,
)

log = logging.getLogger("mkdocs.hooks.spotting")

SPOTTING_PAGE = "spotting.md"
CELLSIZE = (
    20.0  # illustrative cell size for the plots and the filter threshold
)
FILTER = 2 * CELLSIZE
CONIFER_FLI = (
    30000.0  # representative legacy-conifer fireline intensity [kW/m]
)

# ink / grid styling (light background, recessive axes)
INK = "#1a1a1a"
MUTED = "#8a8a8a"
GRID = "#e4e4e4"
# Okabe-Ito colourblind-safe categorical palette (fixed order)
OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]


def d_median(wind_kmh, fli, alignment=1.0):
    """Median landing distance [m] for the given wind, intensity and alignment."""
    directional = np.exp(SPOTTING_ANISOTROPY * (alignment - 1.0))
    return (
        SPOTTING_DISTANCE_REF
        * (wind_kmh / SPOTTING_WIND_REF)
        * (fli / SPOTTING_FLI_REF) ** SPOTTING_FLI_EXPONENT
        * directional
    )


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _style():
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _fig_distance_and_direction(out: Path):
    fig, (axm, _axp) = plt.subplots(
        1,
        2,
        figsize=(12, 4.8),
        gridspec_kw={"width_ratios": [1.35, 1]},
    )
    _axp.remove()
    axp = fig.add_subplot(1, 2, 2, projection="polar")

    # (a) median distance vs wind, one curve per fireline intensity
    intensities = [500, 2000, 10000, 30000, 80000]
    labels = ["500", "2 000", "10 000", "30 000", "80 000"]
    ramp = colormaps["YlOrRd"](np.linspace(0.35, 0.95, len(intensities)))
    wind = np.linspace(0, 60, 300)
    for fli, lab, col in zip(intensities, labels, ramp):
        y = d_median(wind, fli)
        axm.plot(wind, y, color=col, lw=2, solid_capstyle="round")
        axm.annotate(
            lab,
            xy=(wind[-1], y[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=col,
        )
    axm.axhline(FILTER, ls="--", lw=1.3, color=MUTED)
    axm.annotate(
        f"filter: 2·cellsize = {FILTER:.0f} m — shorter embers act as contact spread",
        xy=(34, FILTER),
        xytext=(34, FILTER - 28),
        fontsize=8.2,
        color=MUTED,
        ha="center",
    )
    axm.plot(
        [SPOTTING_WIND_REF],
        [SPOTTING_DISTANCE_REF],
        "o",
        ms=7,
        mfc="white",
        mec=INK,
        mew=1.5,
        zorder=5,
    )
    axm.annotate(
        f"reference state\n{SPOTTING_WIND_REF:.0f} km/h, "
        f"{SPOTTING_FLI_REF:,.0f} kW/m → {SPOTTING_DISTANCE_REF:.0f} m".replace(
            ",", " "
        ),
        xy=(SPOTTING_WIND_REF, SPOTTING_DISTANCE_REF),
        xytext=(24, 210),
        fontsize=8.5,
        color=INK,
        ha="left",
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8),
    )
    axm.set_xlim(0, 66)
    axm.set_ylim(0, 620)
    axm.set_xlabel("wind speed  [km/h]")
    axm.set_ylabel("median downwind landing distance  [m]")
    axm.set_title(
        "(a)  Distance grows with wind and intensity",
        loc="left",
        fontsize=11.5,
        color=INK,
        pad=10,
    )
    axm.text(
        0.02,
        0.97,
        "line colour = fireline intensity [kW/m]",
        transform=axm.transAxes,
        fontsize=8.5,
        color=MUTED,
        va="top",
    )
    axm.grid(True, color=GRID, lw=0.8)
    axm.set_axisbelow(True)

    # (b) directional anisotropy (polar)
    delta = np.linspace(0, 2 * np.pi, 361)
    for wind_kmh, col in zip([10, 30, 60], OKABE):
        r = d_median(wind_kmh, 10000, alignment=np.cos(delta))
        axp.plot(delta, r, color=col, lw=2, label=f"{wind_kmh} km/h")
    axp.set_theta_zero_location("E")
    axp.set_theta_direction(1)
    axp.set_title(
        f"(b)  Downwind concentration\n(anisotropy = {SPOTTING_ANISOTROPY:g})",
        loc="center",
        fontsize=11.5,
        color=INK,
        pad=18,
    )
    axp.set_rlabel_position(135)
    axp.tick_params(colors=MUTED)
    axp.grid(True, color=GRID, lw=0.8)
    axp.legend(
        loc="lower right",
        bbox_to_anchor=(1.18, -0.08),
        frameon=False,
        fontsize=9,
        title="wind",
        title_fontsize=9,
    )

    fig.tight_layout(w_pad=3)
    fig.savefig(out / "spotting_distance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fig_distribution(out: Path):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    scenarios = [
        ("U = 10 km/h,  I = 10 000 kW/m", 10, 10000),
        ("U = 30 km/h,  I = 10 000 kW/m", 30, 10000),
        ("U = 30 km/h,  I = 1 000 kW/m", 30, 1000),
        ("U = 50 km/h,  I = 30 000 kW/m", 50, 30000),
    ]
    x = np.linspace(0, 900, 1000)
    for (lab, u, fli), col in zip(scenarios, OKABE):
        med = d_median(u, fli)
        dist = lognorm(s=SPOTTING_DISTANCE_LOG_SIGMA, scale=med)
        ax.plot(x, dist.pdf(x), color=col, lw=2, label=lab)
        ax.plot([med], [dist.pdf(med)], "o", ms=5, color=col)
    ax.axvspan(0, FILTER, color=MUTED, alpha=0.12, lw=0)
    ax.axvline(FILTER, ls="--", lw=1.2, color=MUTED)
    ax.annotate(
        f"filtered\n(< {FILTER:.0f} m)",
        xy=(FILTER, ax.get_ylim()[1]),
        xytext=(FILTER + 8, 0.0135),
        fontsize=8.5,
        color=MUTED,
    )
    ax.set_xlim(0, 900)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("downwind landing distance  [m]")
    ax.set_ylabel("probability density")
    ax.set_title(
        "Lognormal landing distance (dots = median)",
        loc="left",
        fontsize=11.5,
        color=INK,
        pad=10,
    )
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(
        out / "spotting_distribution.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def generate_figures(docs_dir: Path):
    out = docs_dir / "img"
    out.mkdir(parents=True, exist_ok=True)
    _style()
    _fig_distance_and_direction(out)
    _fig_distribution(out)
    log.info("regenerated fire-spotting figures in %s", out)


# --------------------------------------------------------------------------- #
# Token substitutions
# --------------------------------------------------------------------------- #
_SUPER = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _num(v: float) -> str:
    """Format a distance/scalar without a trailing ``.0`` and space-grouped."""
    s = f"{v:,.0f}" if abs(v - round(v)) < 1e-9 else f"{v:,.1f}"
    return s.replace(",", " ")


def _sci(v: float) -> str:
    """Compact scientific notation, e.g. 4.54e-5 -> ``4.5×10⁻⁵``."""
    if v >= 0.01:
        return f"{v:.2g}"
    exp = int(np.floor(np.log10(abs(v))))
    mant = v / 10**exp
    return f"{mant:.1f}×10{str(exp).translate(_SUPER)}"


def _substitutions() -> dict[str, str]:
    upwind_factor = float(
        np.exp(-2.0 * SPOTTING_ANISOTROPY)
    )  # median scale at Δθ = π
    p05, p95 = lognorm(s=SPOTTING_DISTANCE_LOG_SIGMA).ppf([0.05, 0.95])
    return {
        "DISTANCE_REF": _num(SPOTTING_DISTANCE_REF),
        "WIND_REF": _num(SPOTTING_WIND_REF),
        "FLI_REF": _num(SPOTTING_FLI_REF),
        "FLI_EXPONENT": f"{SPOTTING_FLI_EXPONENT:.3g}",
        "ANISOTROPY": f"{SPOTTING_ANISOTROPY:g}",
        "LOG_SIGMA": f"{SPOTTING_DISTANCE_LOG_SIGMA:g}",
        "MIN_ALIGNMENT": f"{SPOTTING_MIN_ALIGNMENT:g}",
        "LAMBDA": f"{LAMBDA_SPOTTING:g}",
        "P_C0": f"{P_C0:g}",
        "TTP_MEDIAN": _num(SPOTTING_TIME_TO_PROPAGATION_MEDIAN),
        # derived quantities
        "UPWIND_FACTOR": _sci(upwind_factor),
        "UPWIND_RATIO": _num(round(1.0 / upwind_factor)),
        "INTENSITY_10X": f"{10**SPOTTING_FLI_EXPONENT:.2f}",
        "LOG_P05": f"{p05:.2f}",
        "LOG_P95": f"{p95:.2f}",
        "CONIFER_I": _num(CONIFER_FLI),
        "DIST_50": _num(d_median(50, CONIFER_FLI)),
        "DIST_5": _num(d_median(5, CONIFER_FLI)),
        "DIST_01": _num(d_median(0.1, CONIFER_FLI)),
        "FILTER": _num(FILTER),
        "CELLSIZE": _num(CELLSIZE),
    }


# --------------------------------------------------------------------------- #
# MkDocs event hooks
# --------------------------------------------------------------------------- #
def on_pre_build(config, **kwargs):
    generate_figures(Path(config["docs_dir"]))


def on_page_markdown(markdown, page, config, files, **kwargs):
    if page.file.src_uri != SPOTTING_PAGE:
        return markdown
    for name, value in _substitutions().items():
        markdown = markdown.replace(f"{{{{spotting.{name}}}}}", value)
    return markdown
