import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Literal, Mapping, Optional
from warnings import warn

import yaml
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pyproj import CRS

from propagator.cli.console import (
    info_msg,
    ok_msg,
    print_boundary_conditions_table,
    print_model_table,
    setup_console,
    status_propagator_msg,
)
from propagator.core import Propagator, PropagatorOutOfBoundsError
from propagator.core.numba import FUEL_SYSTEM_LEGACY, fuelsystem_from_dict
from propagator.core.numba.models import FuelSystem
from propagator.io.configuration import PropagatorConfigurationLegacy
from propagator.io.loader.geotiff import PropagatorDataFromGeotiffs
from propagator.io.loader.protocol import PropagatorInputDataProtocol
from propagator.io.loader.tiles import PropagatorDataFromTiles
from propagator.io.writer import (
    GeoTiffWriter,
    IsochronesGeoJSONWriter,
    MetadataJSONWriter,
)
from propagator.io.writer.protocol import OutputWriter


# --- CLI configuration -------------------------------------------------------
class PropagatorCLILegacy(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True)

    config: Path = Field(..., description="Path to configuration file (JSON)")
    fuel_config: Optional[Path] = Field(
        None, description="Path to fuel configuration file (YAML)"
    )
    mode: Literal["tiles", "geotiff"] = Field(
        "tiles",
        description="Mode of static data load: 'tiles' for automatic, "
        "'geotiff' for giving DEM and FUEL in input.",
    )
    dem: Optional[Path] = Field(
        None,
        description="Path to DEM file (GeoTIFF), required in 'geotiff' mode",
    )
    fuel: Optional[Path] = Field(
        None,
        description="Path to FUEL file (GeoTIFF), required in 'geotiff' mode",
    )
    tilespath: Optional[Path] = Field(
        None,
        description="Base Path to TILES file (GeoTIFF), required in 'tiles' mode",
    )
    tileset: Optional[str] = Field(
        None,
        description="Tileset to be used in 'tiles' mode (default: 'default')",
    )
    output: Path = Field(
        ...,
        description="Path to output folder where results will be saved",
    )
    isochrones: list[float] = Field(
        [0.5, 0.75, 0.9],
        description="Isochrones thresholds to be saved. \
            Default: [0.5,0.75,0.9]",
    )
    record: bool = Field(
        False,
        description="Export run logs",
    )

    # ---------- checks ----------
    @field_validator("config", mode="before")
    @classmethod
    def _check_config_file(cls, v: str | Path) -> Path:
        if isinstance(v, str):
            v = Path(v)
        # check if the file exists
        if not v.is_file():
            raise ValueError("Configuration file not found.")
        return v

    @field_validator("fuel_config", mode="before")
    @classmethod
    def _check_fuel_config_file(cls, v: str | Path | None) -> Optional[Path]:
        if v is None:
            return None
        if isinstance(v, str):
            v = Path(v)
        # check if the file exists
        if not v.is_file():
            raise ValueError("Fuel configuration file not found.")
        return v

    @field_validator("output", mode="before")
    @classmethod
    def _check_output_folder(cls, v: str | Path) -> Path:
        if isinstance(v, str):
            v = Path(v)
        # check if the folder exists
        if not v.is_dir():
            os.makedirs(v, exist_ok=True)
        return v

    @model_validator(mode="after")
    def _check_mode_files(self):
        # if you provide dem and fuel, then automatically set in geotiff mode
        if self.dem is not None and self.fuel is not None:
            if self.mode == "tiles":
                warn(
                    "DEM and FUEL files provided, switching to 'geotiff' mode"
                )
            self.mode = "geotiff"

        # check required files based on mode
        if self.mode == "geotiff":
            if self.dem is None or self.fuel is None:
                raise ValueError(
                    "DEM and FUEL files must be \
                    provided in 'geotiff' mode"
                )
            if self.tileset is not None:
                warn("TILESET will be ignored in 'geotiff' mode")
            if self.tilespath is not None:
                warn("TILESPATH will be ignored in 'geotiff' mode")
            # check if files exist
            self.dem = Path(self.dem)
            self.fuel = Path(self.fuel)
            if not self.dem.is_file():
                raise ValueError(f"DEM file {self.dem} not found.")
            if not self.fuel.is_file():
                raise ValueError(f"FUEL file {self.fuel} not found.")

        elif self.mode == "tiles":
            if self.tilespath is None:
                raise ValueError(
                    "TILESPATH path must be provided in 'tiles' mode"
                )
            if not self.tilespath.exists():
                raise ValueError(
                    f"TILESPATH path {self.tilespath} does not exist"
                )

        return self

    def build_configuration(self) -> PropagatorConfigurationLegacy:
        """Create configuration object from provided JSON file."""
        with open(self.config) as f:
            json_cfg = json.load(f)
        return PropagatorConfigurationLegacy(**json_cfg)


def fuels_from_yaml(path: str | Path) -> FuelSystem:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    fuels_node = data.get("fuels")
    if not isinstance(fuels_node, Mapping):
        raise ValueError("YAML must contain 'fuels' (mapping)")
    # coerce IDs to int and build Fuel objects
    fs = fuelsystem_from_dict(fuels_node)  # type: ignore
    return fs


# --- main function -----------------------------------------------------------
def main() -> None:
    simulation_time = datetime.now()
    start = time.time()

    info_msg("Initializing CLI...")
    # pydantic-settings is taking care of it
    cli = PropagatorCLILegacy()  # type: ignore
    ok_msg("CLI initialized")

    if cli.record:
        setup_console(record_path=cli.output, basename="run")
    else:
        setup_console()
    ok_msg("Console initialized")
    info_msg(f"Run time: {simulation_time}")

    print_model_table(cli, title="CLI Settings")

    info_msg("Loading configuration from JSON file...")
    cfg = cli.build_configuration()
    ok_msg("Configuration loaded")

    print_model_table(cfg, title="Simulation Configuration")

    info_msg("Loading fuel system...")
    if cli.fuel_config is not None:
        fuel_system = fuels_from_yaml(cli.fuel_config)
        info_msg(f"Fuel system loaded from {cli.fuel_config}")
    else:
        fuel_system = FUEL_SYSTEM_LEGACY
        info_msg("Using legacy fuel system")
    ok_msg("Fuel system ready")

    info_msg("Setting up data loader...")

    loader: PropagatorInputDataProtocol | None = None

    if cli.mode == "tiles":
        # first extract middle point from configuration
        mid_point = cfg.get_ignitions_middle_point()
        if mid_point is None:
            raise ValueError("Ignitions must be defined in the configuration.")

        mid_lat, mid_lon = mid_point[1], mid_point[0]
        loader = PropagatorDataFromTiles(
            base_path=str(cli.tilespath),
            tileset=cli.tileset if cli.tileset is not None else "default",
            mid_lat=mid_lat,
            mid_lon=mid_lon,
            grid_dim=2000,
        )
    elif cli.mode == "geotiff":
        # loader geographic information
        loader = PropagatorDataFromGeotiffs(
            dem_file=str(cli.dem),
            veg_file=str(cli.fuel),
        )
    else:
        raise ValueError(f"Unknown mode: {cli.mode}")

    ok_msg("Data loader ready")

    # Load the data
    dem = loader.get_dem()
    veg = loader.get_veg()
    geo_info = loader.get_geo_info()
    dst_crs = CRS.from_epsg(4326)
    ok_msg("Static data loaded")

    info_msg("Setting up output writer...")
    raster_writer = GeoTiffWriter(
        start_date=cfg.init_date,
        raster_variables_mapping={
            "fire_probability": lambda output: output.fire_probability,
            "fireline_intensity_mean": lambda output: output.fli_mean,
            "fireline_intensity_max": lambda output: output.fli_max,
            "ros_mean": lambda output: output.ros_mean,
            "ros_max": lambda output: output.ros_max,
        },
        output_folder=cli.output,
        geo_info=geo_info,
        dst_crs=dst_crs,
    )

    metadata_writer = MetadataJSONWriter(
        start_date=cfg.init_date, output_folder=cli.output, prefix="metadata"
    )

    isochrones_writer = IsochronesGeoJSONWriter(
        start_date=cfg.init_date,
        output_folder=cli.output,
        prefix="isochrones",
        thresholds=cli.isochrones,
        geo_info=geo_info,
        dst_crs=dst_crs,
    )

    writer = OutputWriter(
        raster_writer=raster_writer,
        metadata_writer=metadata_writer,
        isochrones_writer=isochrones_writer,
    )
    ok_msg("Output writer ready")

    info_msg("Setting up simulator...")

    simulator = Propagator(
        dem=dem,
        veg=veg,
        realizations=cfg.realizations,
        fuels=fuel_system,
        do_spotting=cfg.do_spotting,
        out_of_bounds_mode="raise",
        p_time_fn=cfg.p_time_fn if cfg.p_time_fn is not None else None,
        p_moist_fn=cfg.p_moist_fn if cfg.p_moist_fn is not None else None,
    )
    ok_msg("Simulator ready")

    info_msg("Setting up boundary conditions...")
    non_vegetated = fuel_system.get_non_vegetated()
    boundary_conditions_list = cfg.get_boundary_conditions(
        geo_info, non_vegetated
    )
    for boundary_condition in boundary_conditions_list:
        simulator.set_boundary_conditions(boundary_condition)
    ok_msg("Boundary conditions set")

    print_boundary_conditions_table(cfg.boundary_conditions)

    info_msg("Start simulation...")
    while True:
        next_time = simulator.next_time()
        if next_time is None:
            break

        try:
            simulator.step()
        except PropagatorOutOfBoundsError as e:
            warn(f"Simulation stopped due to PropagatorOutOfBoundsError: {e}")
            break
        finally:
            if simulator.time % cfg.time_resolution == 0:
                output = simulator.get_output()
                status_propagator_msg(
                    cfg.init_date, int(simulator.time), output.stats
                )
                # Save the output to the specified folder
                writer.write_output(output)

        if simulator.time > cfg.time_limit:
            break
    ok_msg("Simulation completed")

    end = time.time()
    info_msg(f"Execution time: {end - start:.2f} seconds")


# %%
if __name__ == "__main__":
    main()
