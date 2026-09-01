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
import math
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
    west: float,
    south: float,
    east: float,
    north: float,
    categories: Sequence[str] | None = None,
) -> str:
    """Return an Overpass QL query for critical buildings/infrastructure
    in the given WGS84 bbox: hospitals/schools/fire stations/police,
    other emergency features, major roads, and generic buildings (all
    reduced to a representative point via `out center`); power
    infrastructure is queried separately with `out geom` so lines and
    polygonal elements (substations, plants) keep their full geometry
    for accurate fire-arrival sampling along their whole extent.

    `categories` (a subset of `POI_CATEGORIES`; `None` means all) selects
    which clauses are emitted, so a narrowed selection doesn't make the
    server collect, and this client parse and cache, elements that will
    be filtered out immediately afterwards -- `way["building"]` alone
    returns tens of thousands of elements for a 10 km radius over a city.
    The cache key is the query text, so a different selection
    transparently gets its own cache entry.
    """
    wanted = set(POI_CATEGORIES if categories is None else categories)
    bbox = f"{south},{west},{north},{east}"
    highways = "|".join(MAJOR_HIGHWAYS)

    point_clauses = []
    amenities = [a for a in CRITICAL_AMENITIES if a in wanted]
    if amenities:
        pattern = "|".join(amenities)
        point_clauses.append(f'  node["amenity"~"^({pattern})$"]({bbox});\n')
        point_clauses.append(f'  way["amenity"~"^({pattern})$"]({bbox});\n')
    if "emergency" in wanted:
        point_clauses.append(f'  node["emergency"]({bbox});\n')
        point_clauses.append(f'  way["emergency"]({bbox});\n')
    if "road" in wanted:
        point_clauses.append(f'  way["highway"~"^({highways})$"]({bbox});\n')
    if "building" in wanted:
        point_clauses.append(f'  way["building"]({bbox});\n')

    query = "[out:json][timeout:25];\n"
    if point_clauses:
        query += "(\n" + "".join(point_clauses) + ");\n" + "out center tags;\n"
    if "power" in wanted:
        query += (
            "(\n"
            f'  node["power"]({bbox});\n'
            f'  way["power"]({bbox});\n'
            ");\n"
            "out geom tags;\n"
        )
    return query


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
            # Overpass reports a server-side query timeout or an
            # out-of-memory abort with HTTP 200 plus a `remark` and a
            # partial (often empty) `elements` list, so
            # raise_for_status() sees nothing wrong. Fail immediately
            # rather than retrying: the outcome is a property of this
            # query over this bbox, so re-sending it unchanged just
            # burns another full server-side timeout (~25 s each) before
            # failing the same way. Raising here also skips the cache
            # write below, which would otherwise freeze that truncated
            # POI set for this bbox until the cache is deleted by hand.
            remark = data.get("remark")
            if remark:
                raise OverpassError(
                    f"Overpass could not complete the query: {remark}. "
                    "The area is likely too large or too dense — reduce "
                    "the radius, or select fewer POI categories."
                )
            break
        except (
            requests.HTTPError,
            requests.ConnectionError,
            requests.Timeout,
            # A truncated or HTML body from an overloaded mirror surfaces
            # as JSONDecodeError (a ValueError, and a RequestException,
            # but neither of the three above) -- exactly the transient
            # condition this loop exists to ride out, so retry it too.
            ValueError,
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


def _matching_groups(tags: dict[str, str]) -> set[str]:
    """Return *every* user-facing category group these tags match, unlike
    `_categorize`, which resolves them to a single winner by precedence.

    Filtering has to use this: OSM elements carry several of these tags
    at once (a hospital way is normally `amenity=hospital` *and*
    `building=yes`), so filtering on the precedence winner alone would
    drop hospitals, schools and police stations from a "buildings"
    selection -- elements the query deliberately asked for and the server
    returned."""
    groups = set()
    amenity = tags.get("amenity")
    if amenity in CRITICAL_AMENITIES:
        groups.add(amenity)
    if "emergency" in tags:
        groups.add("emergency")
    if "power" in tags:
        groups.add("power")
    if tags.get("highway") in MAJOR_HIGHWAYS:
        groups.add("road")
    if "building" in tags:
        groups.add("building")
    return groups


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

    # A degree of longitude is shorter than a degree of latitude away
    # from the equator (~0.71x at 45 deg), so comparing raw squared
    # degrees would rank east/west POIs as ~1.4x farther than equally
    # distant north/south ones and bias the retained set along the N-S
    # axis whenever the cap bites. One cos(lat) scale factor for the
    # whole (small, at these radii) area is enough here.
    lon_scale = math.cos(math.radians(lat))

    def sort_key(poi: POI) -> tuple[int, float]:
        priority = _category_priority(poi.category)
        distance = (poi.lat - lat) ** 2 + ((poi.lon - lon) * lon_scale) ** 2
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
    deduplicated, restricted to `categories` (a subset of
    `POI_CATEGORIES`; `None` keeps every category), and capped to
    `max_pois` (keeping higher-priority categories and those closest to
    the center first).

    `categories` narrows the Overpass query itself, not just the parsed
    result, so a narrowed selection is materially cheaper end to end
    (`way["building"]` alone dominates the response for a city-sized
    bbox). The parsed result is filtered again before truncation, so
    `max_pois` only ever bites into the requested categories. Each
    distinct selection gets its own cache entry, since the cache is keyed
    by the query text."""
    if categories is not None and not categories:
        # An empty selection can only ever yield an empty result, and the
        # query for it carries no `out` statement at all -- don't spend a
        # round trip (plus its retry budget) and a cache entry proving it.
        return []

    west, south, east, north, _utm_epsg = wgs84_bbox_from_center(
        lat, lon, radius_km
    )
    query = build_overpass_query(west, south, east, north, categories)
    data = fetch_overpass(query, cache_dir=cache_dir)
    pois = parse_overpass_elements(data)

    if categories is not None:
        # Match on every group an element's tags belong to, not just the
        # precedence winner `_categorize` assigned: the query fetched it
        # because it matched a *selected* filter, so dropping it here
        # because some other tag outranks that one would silently hide
        # elements the user asked for (see `_matching_groups`).
        wanted = set(categories)
        pois = [p for p in pois if _matching_groups(p.tags) & wanted]

    # An element can legitimately come back twice: the query emits an
    # `out center tags` block and an `out geom tags` block, so a feature
    # matching both (a substation tagged power=* *and* building=*, say) is
    # returned by each -- and only one of the two copies carries
    # `geometry`. Keep the copy that has it, otherwise exactly those
    # features fall back to a single centroid cell, defeating the
    # per-vertex sampling `build_sample_cells` exists for. A dict keeps
    # insertion order, so replacing a value preserves the original
    # position.
    by_key: dict[tuple[str, int], POI] = {}
    for poi in pois:
        dedup_key = (poi.osm_type, poi.osm_id)
        existing = by_key.get(dedup_key)
        if existing is None or (not existing.geometry and poi.geometry):
            by_key[dedup_key] = poi

    return _truncate(list(by_key.values()), lat, lon, max_pois)
