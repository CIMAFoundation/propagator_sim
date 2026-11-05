"""Crown fire module plumbing: configuration fields and canopy rasters."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio as rio
from rasterio.transform import from_origin

from propagator.core.numba import (  # type: ignore
    active_crowning_alexander,
    p_crowning_initiation_cruz,
    p_crowning_initiation_perrakis,
)
from propagator.io.configuration import PropagatorConfigurationLegacy
from propagator.io.loader.geotiff import PropagatorDataFromGeotiffs

BASE_CONFIG = {
    "ignitions": ["POINT: [44.0;8.0]"],
    "boundary_conditions": [
        {"time": 0, "w_dir": 0, "w_speed": 10, "moisture": 5}
    ],
}


def test_configuration_defaults_keep_crowning_off():
    cfg = PropagatorConfigurationLegacy(**BASE_CONFIG)
    assert cfg.do_crowning is False
    assert cfg.crowning_initiation_model == "cruz"
    assert cfg.p_crowning_init_fn is p_crowning_initiation_cruz
    assert cfg.active_crowning_fn is active_crowning_alexander


def test_configuration_selects_crowning_models():
    cfg = PropagatorConfigurationLegacy(
        **BASE_CONFIG,
        do_crowning=True,
        crowning_initiation_model="perrakis",
    )
    assert cfg.do_crowning is True
    assert cfg.p_crowning_init_fn is p_crowning_initiation_perrakis
    with pytest.raises(ValueError):
        PropagatorConfigurationLegacy(
            **BASE_CONFIG, crowning_initiation_model="nope"
        )


def write_raster(path, values, nodata=None):
    profile = dict(
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype=values.dtype,
        crs="EPSG:32632",
        transform=from_origin(400000, 4900000, 20, 20),
        nodata=nodata,
    )
    with rio.open(path, "w", **profile) as dst:
        dst.write(values, 1)


def test_geotiff_loader_reads_optional_canopy(tmp_path):
    shape = (8, 10)
    write_raster(tmp_path / "dem.tif", np.zeros(shape, np.float32))
    write_raster(tmp_path / "veg.tif", np.full(shape, 5, np.int32))
    fsg = np.full(shape, 2.5, np.float32)
    fsg[0, 0] = -9999
    write_raster(tmp_path / "fsg.tif", fsg, nodata=-9999)
    write_raster(tmp_path / "cbd.tif", np.full(shape, 0.2, np.float32))

    plain = PropagatorDataFromGeotiffs(
        dem_file=str(tmp_path / "dem.tif"),
        veg_file=str(tmp_path / "veg.tif"),
    )
    assert plain.get_fsg() is None and plain.get_cbd() is None

    loader = PropagatorDataFromGeotiffs(
        dem_file=str(tmp_path / "dem.tif"),
        veg_file=str(tmp_path / "veg.tif"),
        fsg_file=str(tmp_path / "fsg.tif"),
        cbd_file=str(tmp_path / "cbd.tif"),
    )
    loaded_fsg = loader.get_fsg()
    assert loaded_fsg is not None and loaded_fsg.shape == shape
    assert np.isnan(loaded_fsg[0, 0])  # nodata -> no canopy
    assert loaded_fsg[1, 1] == pytest.approx(2.5)
    np.testing.assert_allclose(loader.get_cbd(), 0.2)
