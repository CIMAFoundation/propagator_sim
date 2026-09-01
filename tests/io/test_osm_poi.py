from __future__ import annotations

import requests

from propagator.io.osm_poi import (
    OVERPASS_URL,
    POI_CATEGORIES,
    OverpassError,
    build_overpass_query,
    default_overpass_url,
    fetch_area_pois,
    fetch_overpass,
    parse_overpass_elements,
)


def test_build_overpass_query_contains_bbox_and_filters():
    query = build_overpass_query(12.0, 42.0, 12.2, 42.2)
    assert "42.0,12.0,42.2,12.2" in query
    assert "hospital|school|fire_station|police" in query
    assert '"emergency"' in query
    assert "motorway|trunk|primary|secondary" in query
    assert '"building"' in query


def test_build_overpass_query_isolates_power_in_its_own_geom_block():
    query = build_overpass_query(12.0, 42.0, 12.2, 42.2)
    assert query.count("out center tags;") == 1
    assert query.count("out geom tags;") == 1
    assert '"power"' in query
    # the power block must come after the "out center" that covers
    # every other category, and be followed by "out geom"
    center_idx = query.index("out center tags;")
    power_idx = query.index('"power"')
    geom_idx = query.index("out geom tags;")
    assert center_idx < power_idx < geom_idx


def test_build_overpass_query_emits_only_the_requested_categories():
    """A narrowed selection must narrow the query itself: `way["building"]`
    alone dominates the response for a city-sized bbox, so fetching it
    only to filter it out afterwards is pure waste."""
    query = build_overpass_query(
        12.0, 42.0, 12.2, 42.2, categories=["hospital", "power"]
    )
    assert '"amenity"~"^(hospital)$"' in query
    assert "school" not in query
    assert '"building"' not in query
    assert '"emergency"' not in query
    assert "motorway" not in query
    assert '"power"' in query
    assert query.count("out center tags;") == 1
    assert query.count("out geom tags;") == 1


def test_build_overpass_query_omits_empty_blocks():
    only_power = build_overpass_query(
        12.0, 42.0, 12.2, 42.2, categories=["power"]
    )
    assert "out center tags;" not in only_power
    assert only_power.count("out geom tags;") == 1

    no_power = build_overpass_query(
        12.0, 42.0, 12.2, 42.2, categories=["building"]
    )
    assert "out geom tags;" not in no_power
    assert no_power.count("out center tags;") == 1


def test_build_overpass_query_default_is_every_category():
    assert build_overpass_query(
        12.0, 42.0, 12.2, 42.2
    ) == build_overpass_query(
        12.0, 42.0, 12.2, 42.2, categories=list(POI_CATEGORIES)
    )


def test_fetch_area_pois_caches_each_category_selection_separately(
    monkeypatch, tmp_path
):
    """The cache is keyed by query text, so narrowing the query must not
    make a narrower selection reuse (or clobber) the full response."""
    queries = []

    def fake_fetch_overpass(query, **kwargs):
        queries.append(query)
        return {"elements": []}

    monkeypatch.setattr(
        "propagator.io.osm_poi.fetch_overpass", fake_fetch_overpass
    )

    fetch_area_pois(42.0, 12.0, 1.0, cache_dir=tmp_path)
    fetch_area_pois(
        42.0, 12.0, 1.0, cache_dir=tmp_path, categories=["hospital"]
    )

    assert len(queries) == 2
    assert queries[0] != queries[1]


def test_parse_overpass_elements_node_and_way_center():
    data = {
        "elements": [
            {
                "type": "node",
                "id": 1,
                "lat": 42.1,
                "lon": 12.1,
                "tags": {"amenity": "hospital", "name": "Test Hospital"},
            },
            {
                "type": "way",
                "id": 2,
                "center": {"lat": 42.2, "lon": 12.2},
                "tags": {"building": "yes"},
            },
        ]
    }
    pois = parse_overpass_elements(data)
    assert len(pois) == 2
    hospital = next(p for p in pois if p.osm_id == 1)
    assert hospital.category == "hospital"
    assert hospital.name == "Test Hospital"
    assert hospital.key == "node/1"
    building = next(p for p in pois if p.osm_id == 2)
    assert building.category == "building"
    assert building.lat == 42.2 and building.lon == 12.2


def test_parse_overpass_elements_categorization_precedence():
    # amenity=hospital beats a building=yes tag on the same element
    data = {
        "elements": [
            {
                "type": "node",
                "id": 1,
                "lat": 1.0,
                "lon": 1.0,
                "tags": {"amenity": "hospital", "building": "yes"},
            }
        ]
    }
    pois = parse_overpass_elements(data)
    assert pois[0].category == "hospital"


def test_parse_overpass_elements_skips_missing_coords_and_unmatched():
    data = {
        "elements": [
            {"type": "node", "id": 1, "tags": {"amenity": "hospital"}},
            {
                "type": "node",
                "id": 2,
                "lat": 1.0,
                "lon": 1.0,
                "tags": {"shop": "bakery"},
            },
        ]
    }
    assert parse_overpass_elements(data) == []


def test_parse_overpass_elements_power_subtype_and_attributes():
    data = {
        "elements": [
            {
                "type": "node",
                "id": 1,
                "lat": 42.0,
                "lon": 12.0,
                "tags": {
                    "power": "substation",
                    "voltage": "132000",
                    "operator": "Terna",
                    "name": "Cabina Test",
                },
            }
        ]
    }
    pois = parse_overpass_elements(data)
    assert len(pois) == 1
    poi = pois[0]
    assert poi.category == "power_substation"
    assert poi.voltage == "132000"
    assert poi.operator == "Terna"
    assert poi.name == "Cabina Test"


def test_parse_overpass_elements_power_line_keeps_full_geometry():
    data = {
        "elements": [
            {
                "type": "way",
                "id": 5,
                "tags": {"power": "line"},
                "geometry": [
                    {"lat": 42.0, "lon": 12.0},
                    {"lat": 42.01, "lon": 12.01},
                    {"lat": 42.02, "lon": 12.02},
                ],
            }
        ]
    }
    pois = parse_overpass_elements(data)
    assert len(pois) == 1
    poi = pois[0]
    assert poi.category == "power_line"
    assert poi.geometry == (
        (42.0, 12.0),
        (42.01, 12.01),
        (42.02, 12.02),
    )
    # representative lat/lon is the middle vertex of the line
    assert (poi.lat, poi.lon) == (42.01, 12.01)


def test_parse_overpass_elements_power_way_without_geometry_falls_back_to_center():
    data = {
        "elements": [
            {
                "type": "way",
                "id": 6,
                "center": {"lat": 41.0, "lon": 11.0},
                "tags": {"power": "tower"},
            }
        ]
    }
    pois = parse_overpass_elements(data)
    assert pois[0].category == "power_tower"
    assert pois[0].geometry is None
    assert (pois[0].lat, pois[0].lon) == (41.0, 11.0)


def test_default_overpass_url_uses_env_override(monkeypatch):
    monkeypatch.delenv("PROPAGATOR_OVERPASS_URL", raising=False)
    assert default_overpass_url() == OVERPASS_URL

    monkeypatch.setenv(
        "PROPAGATOR_OVERPASS_URL", "https://z.overpass-api.de/api/interpreter"
    )
    assert (
        default_overpass_url() == "https://z.overpass-api.de/api/interpreter"
    )


def test_fetch_overpass_uses_env_endpoint_when_not_given(
    monkeypatch, tmp_path
):
    monkeypatch.setenv(
        "PROPAGATOR_OVERPASS_URL", "https://mirror.example/api/interpreter"
    )
    seen_urls = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"elements": []}

    def fake_post(url, data, headers, timeout):
        seen_urls.append(url)
        return FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)

    fetch_overpass("q", cache_dir=tmp_path)
    assert seen_urls == ["https://mirror.example/api/interpreter"]


def test_fetch_overpass_caches_response(monkeypatch, tmp_path):
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"elements": []}

    def fake_post(url, data, headers, timeout):
        calls.append(url)
        return FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)

    query = "test query"
    result1 = fetch_overpass(query, cache_dir=tmp_path)
    result2 = fetch_overpass(query, cache_dir=tmp_path)

    assert result1 == {"elements": []}
    assert result2 == {"elements": []}
    assert len(calls) == 1


def test_fetch_overpass_retries_then_succeeds(monkeypatch, tmp_path):
    attempts = {"n": 0}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"elements": []}

    def fake_post(url, data, headers, timeout):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise requests.ConnectionError("boom")
        return FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr("propagator.io.osm_poi.time.sleep", lambda s: None)

    result = fetch_overpass(
        "q", cache_dir=tmp_path, max_retries=3, backoff_s=0
    )
    assert result == {"elements": []}
    assert attempts["n"] == 3


def test_fetch_overpass_raises_after_exhausting_retries(monkeypatch, tmp_path):
    def fake_post(url, data, headers, timeout):
        raise requests.ConnectionError("boom")

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr("propagator.io.osm_poi.time.sleep", lambda s: None)

    try:
        fetch_overpass("q", cache_dir=tmp_path, max_retries=2, backoff_s=0)
        assert False, "expected OverpassError"
    except OverpassError:
        pass


def test_fetch_overpass_retries_a_malformed_body(monkeypatch, tmp_path):
    """A truncated or HTML body from an overloaded mirror surfaces as
    JSONDecodeError, which is neither HTTPError, ConnectionError nor
    Timeout -- it used to abort on the first attempt, though it is
    exactly the transient condition the retry loop exists for."""
    attempts = {"n": 0}

    class FakeResponse:
        def __init__(self, ok):
            self._ok = ok

        def raise_for_status(self):
            pass

        def json(self):
            if not self._ok:
                raise ValueError("Expecting value: line 1 column 1")
            return {"elements": []}

    def fake_post(url, data, headers, timeout):
        attempts["n"] += 1
        return FakeResponse(ok=attempts["n"] >= 3)

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr("propagator.io.osm_poi.time.sleep", lambda s: None)

    result = fetch_overpass(
        "q", cache_dir=tmp_path, max_retries=3, backoff_s=0
    )
    assert result == {"elements": []}
    assert attempts["n"] == 3


def test_fetch_overpass_fails_fast_on_a_remark_and_never_caches_it(
    monkeypatch, tmp_path
):
    """Overpass reports a server-side query timeout / out-of-memory abort
    with HTTP 200 plus a `remark` and a partial `elements` list, which
    raise_for_status() lets through. That outcome is deterministic for
    the same query, so re-sending it just burns another server-side
    timeout; and caching it would freeze a truncated POI set for this
    bbox permanently."""
    attempts = {"n": 0}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"remark": "runtime error: Query timed out", "elements": []}

    def fake_post(url, data, headers, timeout):
        attempts["n"] += 1
        return FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr("propagator.io.osm_poi.time.sleep", lambda s: None)

    try:
        fetch_overpass("q", cache_dir=tmp_path, max_retries=3, backoff_s=0)
        assert False, "expected OverpassError"
    except OverpassError as e:
        assert "Query timed out" in str(e)

    assert attempts["n"] == 1, "a remark must not be retried"
    assert not (tmp_path / "osm").exists()


def test_fetch_area_pois_keeps_elements_matching_any_selected_category(
    monkeypatch, tmp_path
):
    """Regression test: filtering used `_categorize`'s single winning
    category, so a hospital way (which is also `building=yes`) was
    dropped from a "building"-only selection -- an element the query
    explicitly asked for and the server returned."""
    hospital_building = {
        "type": "way",
        "id": 7,
        "center": {"lat": 42.0, "lon": 12.0},
        "tags": {"amenity": "hospital", "building": "yes", "name": "Osp"},
    }
    monkeypatch.setattr(
        "propagator.io.osm_poi.fetch_overpass",
        lambda *a, **k: {"elements": [hospital_building]},
    )

    only_buildings = fetch_area_pois(
        42.0, 12.0, 1.0, cache_dir=tmp_path, categories=["building"]
    )
    assert [p.osm_id for p in only_buildings] == [7]

    only_hospitals = fetch_area_pois(
        42.0, 12.0, 1.0, cache_dir=tmp_path, categories=["hospital"]
    )
    assert [p.osm_id for p in only_hospitals] == [7]

    only_roads = fetch_area_pois(
        42.0, 12.0, 1.0, cache_dir=tmp_path, categories=["road"]
    )
    assert only_roads == []


def test_fetch_area_pois_short_circuits_an_empty_category_selection(
    monkeypatch, tmp_path
):
    """An empty selection cannot match anything, so it must not spend a
    request (nor a cache entry) proving it."""
    calls = []

    def fake_fetch_overpass(query, **kwargs):
        calls.append(query)
        return {"elements": []}

    monkeypatch.setattr(
        "propagator.io.osm_poi.fetch_overpass", fake_fetch_overpass
    )

    assert (
        fetch_area_pois(42.0, 12.0, 1.0, cache_dir=tmp_path, categories=[])
        == []
    )
    assert calls == []


def test_fetch_area_pois_dedup_keeps_the_copy_carrying_geometry(
    monkeypatch, tmp_path
):
    """A feature tagged both power=* and building=* is returned by the
    query's `out center tags` block *and* its `out geom tags` block; only
    the latter carries `geometry`. Dedup used to keep whichever came
    first, silently reducing those features to a single centroid cell."""
    element_center = {
        "type": "way",
        "id": 42,
        "center": {"lat": 42.0, "lon": 12.0},
        "tags": {"power": "substation", "building": "yes"},
    }
    element_geom = {
        "type": "way",
        "id": 42,
        "center": {"lat": 42.0, "lon": 12.0},
        "geometry": [
            {"lat": 42.0, "lon": 12.0},
            {"lat": 42.001, "lon": 12.001},
            {"lat": 42.002, "lon": 12.0},
        ],
        "tags": {"power": "substation", "building": "yes"},
    }

    monkeypatch.setattr(
        "propagator.io.osm_poi.fetch_overpass",
        lambda *a, **k: {"elements": [element_center, element_geom]},
    )

    pois = fetch_area_pois(42.0, 12.0, 1.0, cache_dir=tmp_path)

    assert len(pois) == 1
    assert pois[0].geometry is not None
    assert len(pois[0].geometry) == 3


def test_fetch_area_pois_truncates_by_priority(monkeypatch, tmp_path):
    elements = []
    for i in range(10):
        elements.append(
            {
                "type": "node",
                "id": i,
                "lat": 42.0 + i * 0.001,
                "lon": 12.0,
                "tags": {"building": "yes"},
            }
        )
    elements.append(
        {
            "type": "node",
            "id": 100,
            "lat": 42.0,
            "lon": 12.0,
            "tags": {"amenity": "hospital", "name": "Priority Hospital"},
        }
    )

    monkeypatch.setattr(
        "propagator.io.osm_poi.fetch_overpass",
        lambda query, cache_dir=None: {"elements": elements},
    )

    pois = fetch_area_pois(42.0, 12.0, 5.0, cache_dir=tmp_path, max_pois=3)
    assert len(pois) == 3
    assert any(p.category == "hospital" for p in pois)


def test_fetch_area_pois_filters_by_categories(monkeypatch, tmp_path):
    elements = [
        {
            "type": "node",
            "id": 1,
            "lat": 42.0,
            "lon": 12.0,
            "tags": {"amenity": "hospital"},
        },
        {
            "type": "node",
            "id": 2,
            "lat": 42.001,
            "lon": 12.0,
            "tags": {"building": "yes"},
        },
        {
            "type": "way",
            "id": 3,
            "geometry": [
                {"lat": 42.0, "lon": 12.0},
                {"lat": 42.002, "lon": 12.002},
            ],
            "tags": {"power": "line"},
        },
    ]
    monkeypatch.setattr(
        "propagator.io.osm_poi.fetch_overpass",
        lambda query, cache_dir=None: {"elements": elements},
    )

    pois = fetch_area_pois(
        42.0, 12.0, 5.0, cache_dir=tmp_path, categories=["hospital", "power"]
    )
    categories = {p.category for p in pois}
    assert categories == {"hospital", "power_line"}

    pois_none = fetch_area_pois(
        42.0, 12.0, 5.0, cache_dir=tmp_path, categories=[]
    )
    assert pois_none == []

    pois_all = fetch_area_pois(42.0, 12.0, 5.0, cache_dir=tmp_path)
    assert len(pois_all) == 3
