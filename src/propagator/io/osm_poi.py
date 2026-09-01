"""Fetch OpenStreetMap points of interest (POIs) for an area, via the
public Overpass API, to report as "assets at risk" against the simulated
fire front.

Reuses `data_prep.wgs84_bbox_from_center` for the query area (the same
square bbox already used to download DEM/fuel data), so POIs cover exactly
the area the simulation runs on. Overpass responses are cached on disk
under `cache_dir / "osm"`, keyed by a hash of the query, following the
same download-cache convention as `data_prep.download`.

The main `overpass-api.de` endpoint is unreachable/rate-limited on some
networks; `OVERPASS_URL` below defaults to `z.overpass-api.de`, a
different IP of that same official service that has proven reachable
where the main one wasn't. Override at runtime with the
`PROPAGATOR_OVERPASS_URL` environment variable without changing code —
useful to switch mirrors per-deployment, or to point at a self-hosted
instance. Not every public mirror accepts arbitrary traffic (e.g.
`overpass.openstreetmap.fr` requires prior whitelisting and returns 403
otherwise) — confirm a candidate mirror is actually reachable before
relying on it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import requests

from propagator.io.data_prep import wgs84_bbox_from_center

# Confirmed reachable where the main overpass-api.de endpoint was not.
OVERPASS_URL = "https://z.overpass-api.de/api/interpreter"

# Other public mirrors, for reference (override via PROPAGATOR_OVERPASS_URL
# instead of editing this file). Not all are usable from every network —
# see the module docstring.
# OVERPASS_URL = "https://overpass-api.de/api/interpreter"  # official, round-robin DNS
# OVERPASS_URL = "https://lz4.overpass-api.de/api/interpreter"  # official, another IP
# OVERPASS_URL = "https://overpass.kumi.systems/api/interpreter"
# OVERPASS_URL = "https://overpass.openstreetmap.ru/api/interpreter"
# OVERPASS_URL = "https://overpass.openstreetmap.fr/api/interpreter"  # requires whitelisting

# Cap on the number of POIs kept for one area, since `building=*` alone
# can return thousands of ways even for a 15 km radius; this keeps the
# per-frame sampling and the map overlay responsive on a local run.
DEFAULT_MAX_POIS = 1000

CRITICAL_AMENITIES = ("hospital", "school", "fire_station", "police")
MAJOR_HIGHWAYS = ("motorway", "trunk", "primary", "secondary")

# Lower priority is kept first when truncating to max_pois.
CATEGORY_PRIORITY = {
    "hospital": 0,
    "fire_station": 0,
    "police": 1,
    "school": 1,
    "emergency": 1,
    "road": 3,
    "building": 4,
}

# Priority within power_* categories (see `_category_priority`): kept
# separate from CATEGORY_PRIORITY since the category string itself is
# dynamic (`power_<subtype>`), not a fixed key.
POWER_SUBTYPE_PRIORITY = {
    "substation": 1,
    "plant": 1,
    "generator": 1,
    "transformer": 2,
    "switch": 2,
    "line": 2,
    "minor_line": 2,
    "cable": 2,
}

# User-facing category groups, for filtering (see `fetch_area_pois`'s
# `categories` parameter): every `power_<subtype>` value that
# `_categorize`/`_power_category` can produce is bucketed under the
# single "power" group, since the subtype is data-driven and unbounded
# (not a fixed, checkbox-able set). Kept in sync manually with
# `propagator.web.schemas.SimulateRequest.poi_categories`'s Literal.
POI_CATEGORIES = (
    "hospital",
    "fire_station",
    "police",
    "school",
    "emergency",
    "road",
    "building",
    "power",
)


def _category_group(category: str) -> str:
    """Return the user-facing category group for a POI's specific
    `category` (collapsing any `power_<subtype>` to "power")."""
    return "power" if category.startswith("power_") else category


class OverpassError(Exception):
    """Raised when the Overpass API request fails after retries."""


@dataclass(frozen=True)
class POI:
    """One OpenStreetMap element of interest."""

    osm_id: int
    osm_type: str  # "node" | "way" | "relation"
    category: str  # one of CATEGORY_PRIORITY's keys, or "power_<subtype>"
    name: str | None
    lat: float
    lon: float
    tags: dict[str, str] = field(default_factory=dict)
    voltage: str | None = None
    operator: str | None = None
    # Full (lat, lon) vertex list for a way/relation, when available (via
    # Overpass `out geom`); None for a plain point (node, or a way with
    # no usable geometry). Lets callers sample fire arrival along an
    # entire line/polygon instead of a single representative point.
    geometry: tuple[tuple[float, float], ...] | None = None

    @property
    def key(self) -> str:
        """Stable id for this POI, suitable as a `Propagator.sample_cells`
        key (e.g. "node/123456")."""
        return f"{self.osm_type}/{self.osm_id}"


def build_overpass_query(
    west: float, south: float, east: float, north: float
) -> str:
    """Return an Overpass QL query for critical buildings/infrastructure
    in the given WGS84 bbox: hospitals/schools/fire stations/police,
    other emergency features, major roads, and generic buildings (all
    reduced to a representative point via `out center`); power
    infrastructure is queried separately with `out geom` so lines and
    polygonal elements (substations, plants) keep their full geometry
    for accurate fire-arrival sampling along their whole extent."""
    bbox = f"{south},{west},{north},{east}"
    amenities = "|".join(CRITICAL_AMENITIES)
    highways = "|".join(MAJOR_HIGHWAYS)
    return (
        "[out:json][timeout:25];\n"
        "(\n"
        f'  node["amenity"~"^({amenities})$"]({bbox});\n'
        f'  way["amenity"~"^({amenities})$"]({bbox});\n'
        f'  node["emergency"]({bbox});\n'
        f'  way["emergency"]({bbox});\n'
        f'  way["highway"~"^({highways})$"]({bbox});\n'
        f'  way["building"]({bbox});\n'
        ");\n"
        "out center tags;\n"
        "(\n"
        f'  node["power"]({bbox});\n'
        f'  way["power"]({bbox});\n'
        ");\n"
        "out geom tags;\n"
    )


def _cache_path(cache_dir: Path, query: str) -> Path:
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
    return cache_dir / "osm" / f"{digest}.json"


def default_overpass_url() -> str:
    """Return the Overpass endpoint to use: `PROPAGATOR_OVERPASS_URL` if
    set, else the official `OVERPASS_URL`. Resolved at call time (not a
    module-level default) so the env var can be changed between calls,
    e.g. in tests."""
    return os.environ.get("PROPAGATOR_OVERPASS_URL", OVERPASS_URL)


def fetch_overpass(
    query: str,
    *,
    cache_dir: Path | None = None,
    endpoint: str | None = None,
    max_retries: int = 3,
    backoff_s: float = 2.0,
    timeout: float | tuple[float, float] = (5.0, 30.0),
) -> dict:
    """Run `query` against the Overpass API, caching the raw JSON
    response on disk (skipped entirely on a cache hit).

    `endpoint` defaults to `default_overpass_url()` (the official
    endpoint, or `PROPAGATOR_OVERPASS_URL` if set) when not given
    explicitly.

    `timeout` is a `(connect, read)` pair by default: a short connect
    timeout so an unreachable/unresponsive endpoint fails fast (and
    retries/gives up quickly) rather than blocking the whole "preparing
    data" phase for minutes, while the read timeout stays generous enough
    to cover the query's own `[timeout:25]` execution budget plus
    transfer time.

    Retries with exponential backoff on connection/timeout/HTTP errors,
    since the public Overpass endpoint is far more rate-limit-sensitive
    than the static COG URLs used for DEM/WorldCover; raises
    `OverpassError` once retries are exhausted.
    """
    endpoint = endpoint or default_overpass_url()
    dest = None
    if cache_dir is not None:
        dest = _cache_path(Path(cache_dir), query)
        if dest.exists():
            return json.loads(dest.read_text(encoding="utf-8"))

    # Overpass's usage policy (https://wiki.openstreetmap.org/wiki/Overpass_API)
    # asks clients to identify themselves; some mirrors also reject
    # requests with no descriptive User-Agent.
    headers = {
        "User-Agent": "propagator-sim/osm_poi (CIMA Research Foundation)"
    }

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = requests.post(
                endpoint,
                data={"data": query},
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            break
        except (
            requests.HTTPError,
            requests.ConnectionError,
            requests.Timeout,
        ) as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(backoff_s * (2**attempt))
    else:
        raise OverpassError(
            f"Overpass request failed after {max_retries} attempts: "
            f"{last_error}"
        )

    if dest is not None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.rename(dest)
    return data


def _power_category(tags: dict[str, str]) -> str:
    """Return a specific `power_<subtype>` category (e.g. "power_line",
    "power_substation") instead of the generic "power", from the tag's
    value (falling back to "power_other" for an empty/unusual value)."""
    subtype = tags.get("power", "").strip().lower().split(";")[0] or "other"
    subtype = re.sub(r"[^a-z0-9]+", "_", subtype).strip("_") or "other"
    return f"power_{subtype}"


def _category_priority(category: str) -> int:
    if category.startswith("power_"):
        return POWER_SUBTYPE_PRIORITY.get(category[len("power_") :], 3)
    return CATEGORY_PRIORITY.get(category, len(CATEGORY_PRIORITY))


def _categorize(tags: dict[str, str]) -> str | None:
    amenity = tags.get("amenity")
    if amenity in CRITICAL_AMENITIES:
        return amenity
    if "emergency" in tags:
        return "emergency"
    if "power" in tags:
        return _power_category(tags)
    if tags.get("highway") in MAJOR_HIGHWAYS:
        return "road"
    if "building" in tags:
        return "building"
    return None


def _element_geometry(el: dict) -> tuple[tuple[float, float], ...] | None:
    """Return the full (lat, lon) vertex list from an Overpass `out geom`
    way/relation element, or None if absent/degenerate (a single point
    carries no extra information over `lat`/`lon`)."""
    raw = el.get("geometry")
    if not raw:
        return None
    points = tuple(
        (pt["lat"], pt["lon"])
        for pt in raw
        if pt and "lat" in pt and "lon" in pt
    )
    return points if len(points) > 1 else None


def parse_overpass_elements(data: dict) -> list[POI]:
    """Parse an Overpass JSON response into `POI`s, categorizing each
    element and skipping any without usable coordinates or a matching
    category."""
    pois = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        category = _categorize(tags)
        if category is None:
            continue

        geometry = _element_geometry(el)
        if geometry is not None:
            lat, lon = geometry[len(geometry) // 2]
        elif "lat" in el and "lon" in el:
            lat, lon = el["lat"], el["lon"]
        elif "center" in el:
            lat, lon = el["center"]["lat"], el["center"]["lon"]
        else:
            continue

        pois.append(
            POI(
                osm_id=el["id"],
                osm_type=el["type"],
                category=category,
                name=tags.get("name"),
                lat=lat,
                lon=lon,
                tags=tags,
                voltage=tags.get("voltage"),
                operator=tags.get("operator"),
                geometry=geometry,
            )
        )
    return pois


def _truncate(
    pois: list[POI], lat: float, lon: float, max_pois: int
) -> list[POI]:
    if len(pois) <= max_pois:
        return pois

    def sort_key(poi: POI) -> tuple[int, float]:
        priority = _category_priority(poi.category)
        distance = (poi.lat - lat) ** 2 + (poi.lon - lon) ** 2
        return priority, distance

    return sorted(pois, key=sort_key)[:max_pois]


def fetch_area_pois(
    lat: float,
    lon: float,
    radius_km: float,
    *,
    cache_dir: Path | None = None,
    max_pois: int = DEFAULT_MAX_POIS,
    categories: Sequence[str] | None = None,
) -> list[POI]:
    """Fetch critical-building/infrastructure POIs from OpenStreetMap for
    the same square bbox `data_prep.prepare_area_data` uses for DEM/fuel,
    deduplicated, optionally filtered to `categories` (a subset of
    `POI_CATEGORIES`; `None` keeps every category, matching the previous
    default behavior), and capped to `max_pois` (keeping higher-priority
    categories and those closest to the center first).

    The Overpass query itself always fetches every category (so the
    on-disk response cache stays valid across different `categories`
    selections for the same area) — filtering happens after parsing,
    before truncation, so `max_pois` only bites into the categories the
    caller actually wants."""
    west, south, east, north, _utm_epsg = wgs84_bbox_from_center(
        lat, lon, radius_km
    )
    query = build_overpass_query(west, south, east, north)
    data = fetch_overpass(query, cache_dir=cache_dir)
    pois = parse_overpass_elements(data)

    if categories is not None:
        wanted = set(categories)
        pois = [p for p in pois if _category_group(p.category) in wanted]

    seen: set[tuple[str, int]] = set()
    deduped = []
    for poi in pois:
        dedup_key = (poi.osm_type, poi.osm_id)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        deduped.append(poi)

    return _truncate(deduped, lat, lon, max_pois)
