from __future__ import annotations

import numpy as np
import requests
from rasterio import Affine

from propagator.io.data_prep import (
    ITALY_BBOX,
    NON_VEGETATED_FUEL,
    build_target_grid,
    dem_tile_urls,
    download_italy_tiles,
    ignition_cell_fuel,
    remap_worldcover_to_fuel,
    utm_epsg_for,
    wgs84_bbox_from_center,
    worldcover_tile_urls,
)


def test_utm_epsg_for_viterbo():
    # Viterbo, Italy: lon ~12.1E -> UTM zone 33N
    assert utm_epsg_for(42.4207, 12.1077) == "EPSG:32633"


def test_utm_epsg_for_southern_hemisphere():
    assert utm_epsg_for(-33.87, 151.21) == "EPSG:32756"


def test_utm_epsg_for_norway_irregular_zone():
    # the naive (lon + 180) // 6 + 1 formula gives zone 31 here; the
    # actual UTM zone, accounting for Norway's irregular zone boundary,
    # is 32 (matching io.loader.tiles, which uses utm.from_latlon)
    assert utm_epsg_for(60.0, 5.0) == "EPSG:32632"


def test_wgs84_bbox_from_center_is_centered_and_ordered():
    west, south, east, north, utm_epsg = wgs84_bbox_from_center(
        42.4207, 12.1077, 10.0
    )
    assert west < 12.1077 < east
    assert south < 42.4207 < north
    assert utm_epsg == "EPSG:32633"


def test_dem_tile_urls_single_tile():
    urls = list(dem_tile_urls(12.05, 42.10, 12.20, 42.20))
    assert len(urls) == 1
    url, name = urls[0]
    assert name == "Copernicus_DSM_COG_10_N42_00_E012_00_DEM"
    assert url.endswith(f"{name}/{name}.tif")


def test_dem_tile_urls_spans_two_tiles():
    # bbox straddling the 12-degree meridian -> two 1x1 degree tiles
    names = {name for _, name in dem_tile_urls(11.9, 42.1, 12.1, 42.2)}
    assert names == {
        "Copernicus_DSM_COG_10_N42_00_E011_00_DEM",
        "Copernicus_DSM_COG_10_N42_00_E012_00_DEM",
    }


def test_worldcover_tile_urls_rounds_to_3_degrees():
    urls = list(worldcover_tile_urls(12.05, 42.10, 12.20, 42.20))
    assert len(urls) == 1
    url, name = urls[0]
    assert name == "ESA_WorldCover_10m_2021_v200_N42E012_Map"
    assert url.endswith(f"{name}.tif")


# N/S must depend only on latitude and E/W only on longitude, so each of
# the four lat/lon sign combinations is exercised independently below
# (regression for a bug where the E/W hemisphere was derived from `lat`
# instead of `lon`, which silently pointed DEM/WorldCover downloads at
# the wrong side of the globe for any negative-longitude bbox).


def test_dem_tile_urls_northern_western_hemisphere():
    # San Francisco, USA: lat +37, lon -122 (west of Greenwich)
    names = {name for _, name in dem_tile_urls(-122.95, 37.05, -122.80, 37.15)}
    assert names == {"Copernicus_DSM_COG_10_N37_00_W123_00_DEM"}


def test_dem_tile_urls_southern_eastern_hemisphere():
    # Sydney, Australia: lat -34, lon +151
    names = {name for _, name in dem_tile_urls(151.05, -33.95, 151.20, -33.85)}
    assert names == {"Copernicus_DSM_COG_10_S34_00_E151_00_DEM"}


def test_dem_tile_urls_southern_western_hemisphere():
    # Santiago, Chile: lat -34, lon -71
    names = {name for _, name in dem_tile_urls(-70.70, -33.50, -70.55, -33.40)}
    assert names == {"Copernicus_DSM_COG_10_S34_00_W071_00_DEM"}


def test_worldcover_tile_urls_northern_western_hemisphere():
    names = {
        name
        for _, name in worldcover_tile_urls(-122.95, 37.05, -122.80, 37.15)
    }
    assert names == {"ESA_WorldCover_10m_2021_v200_N36W123_Map"}


def test_worldcover_tile_urls_southern_eastern_hemisphere():
    names = {
        name
        for _, name in worldcover_tile_urls(151.05, -33.95, 151.20, -33.85)
    }
    assert names == {"ESA_WorldCover_10m_2021_v200_S36E150_Map"}


def test_worldcover_tile_urls_southern_western_hemisphere():
    names = {
        name
        for _, name in worldcover_tile_urls(-70.70, -33.50, -70.55, -33.40)
    }
    assert names == {"ESA_WorldCover_10m_2021_v200_S36W072_Map"}


def test_build_target_grid_covers_requested_radius():
    transform, width, height = build_target_grid(
        42.4207, 12.1077, 5.0, 30.0, "EPSG:32633"
    )
    assert width == height
    # half-width in meters must cover at least the requested radius
    assert (width / 2) * 30.0 >= 5000.0
    assert isinstance(transform, Affine)
    assert transform.a == 30.0
    assert transform.e == -30.0


def test_remap_worldcover_to_fuel_known_classes():
    wc = np.array([[10, 20, 30], [40, 50, 200]], dtype=np.uint8)
    fuel = remap_worldcover_to_fuel(wc)
    assert fuel.dtype == np.int16
    assert fuel.tolist() == [[1, 2, 4], [6, 3, NON_VEGETATED_FUEL]]


def test_ignition_cell_fuel_inside_and_outside_grid():
    utm_epsg = "EPSG:32633"
    transform, width, height = build_target_grid(
        42.4207, 12.1077, 5.0, 30.0, utm_epsg
    )
    fuel = np.full((height, width), 4, dtype=np.int16)  # all grassland

    # center of the grid must be inside
    code = ignition_cell_fuel(fuel, transform, 42.4207, 12.1077, utm_epsg)
    assert code == 4

    # far outside the grid must be None
    code_outside = ignition_cell_fuel(fuel, transform, 10.0, 10.0, utm_epsg)
    assert code_outside is None


def test_download_italy_tiles_covers_expected_tile_counts(
    monkeypatch, tmp_path
):
    expected_dem = len(list(dem_tile_urls(*ITALY_BBOX)))
    expected_wc = len(list(worldcover_tile_urls(*ITALY_BBOX)))
    assert expected_dem == 169
    assert expected_wc == 25

    def fake_download(url, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x" * 10)
        return dest

    monkeypatch.setattr("propagator.io.data_prep.download", fake_download)

    events = []
    summary = download_italy_tiles(
        cache_dir=tmp_path,
        progress_cb=lambda kind, name, size: events.append((kind, name, size)),
    )

    assert summary["n_dem_downloaded"] == expected_dem
    assert summary["n_dem_missing"] == 0
    assert summary["n_worldcover_downloaded"] == expected_wc
    assert summary["n_worldcover_missing"] == 0
    assert summary["total_bytes"] == 10 * (expected_dem + expected_wc)
    assert len(events) == expected_dem + expected_wc
    assert all(kind in ("dem_ok", "wc_ok") for kind, _, _ in events)


def test_download_italy_tiles_skips_missing_coverage(monkeypatch, tmp_path):
    def fake_download(url, dest):
        if "N40_00_E007" in dest.name:
            raise requests.HTTPError("404")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x")
        return dest

    monkeypatch.setattr("propagator.io.data_prep.download", fake_download)

    summary = download_italy_tiles(cache_dir=tmp_path)

    assert summary["n_dem_missing"] == 1
    assert summary["n_dem_downloaded"] == 168
