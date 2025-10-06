import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, Optional
from warnings import warn

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pyproj import CRS

from propagator.cli.console import info_msg, ok_msg, setup_console
from propagator.core import Propagator
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
        [0.9, 0.95],
        description="Isochrones thresholds to be saved (e.g. 0.9, 0.95)",
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

        elif self.mode == "tiles":
            if self.dem is not None or self.fuel is not None:
                warn(
                    "DEM and FUEL files shouldn't be \
                    provided in 'tiles' mode and will be ignored."
                )
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
        """Merge CLI config and JSON config into one validated object.
        NOTE: CLI config override JSON config in case of overlapping"""
        with open(self.config) as f:
            json_cfg = json.load(f)
        # CLI values override JSON if both are provided
        return PropagatorConfigurationLegacy(**json_cfg, **self.model_dump())


# --- main function -----------------------------------------------------------
def main():
    simulation_time = datetime.now()

    info_msg("Initializing CLI...")
    # pydantic-settings is taking care of it
    cli = PropagatorCLILegacy()  # type: ignore
    ok_msg("CLI initialized")
    # print(cli.model_dump())

    if cli.record:
        basename = (
            f"propagator_run_{simulation_time.strftime('%Y%m%d_%H%M%S')}"
        )
        setup_console(record_path=cli.output, basename=basename)
    else:
        setup_console()
    ok_msg("Console initialized")

    info_msg("Loading configuration from JSON file...")
    cfg = cli.build_configuration()
    ok_msg("Configuration loaded")

    loader: PropagatorInputDataProtocol
    if cli.mode == "tiles":
        # first extract middle point from configuration
        mid_point = cfg.get_ignitions_middle_point()
        if mid_point is None:
            raise ValueError("Ignitions must be defined in the configuration.")

        mid_lat, mid_lon = mid_point[1], mid_point[0]
        loader = PropagatorDataFromTiles(
            base_path=str(cfg.tilespath),
            tileset=cli.tileset if cli.tileset is not None else "default",
            mid_lat=mid_lat,
            mid_lon=mid_lon,
            grid_dim=2000,
        )

    elif cli.mode == "geotiff":
        # loader geographic information
        loader = PropagatorDataFromGeotiffs(
            dem_file=str(cfg.dem),
            veg_file=str(cfg.fuel),
        )
    else:
        raise ValueError(f"Unknown mode {cli.mode}")

    # Load the data
    dem = loader.get_dem()
    veg = loader.get_veg()
    geo_info = loader.get_geo_info()
    dst_crs = CRS.from_epsg(4326)

    raster_writer = GeoTiffWriter(
        start_date=cfg.init_date,
        raster_variables_mapping={
            "fire_probability": lambda output: output.fire_probability,
            "fireline_intensity_mean": lambda output: output.fli_mean,
            "fireline_intensity_max": lambda output: output.fli_max,
            "ros_mean": lambda output: output.ros_mean,
            "ros_max": lambda output: output.ros_max,
        },
        output_folder=cfg.output,
        geo_info=geo_info,
        dst_crs=dst_crs,
    )

    metadata_writer = MetadataJSONWriter(
        start_date=cfg.init_date, output_folder=cfg.output, prefix="metadata"
    )

    isochrones_writer = IsochronesGeoJSONWriter(
        start_date=cfg.init_date,
        output_folder=cfg.output,
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

    args = dict()
    if cfg.p_time_fn is not None:
        args.update(dict(p_time_fn=cfg.p_time_fn))
    if cfg.p_moist_fn is not None:
        args.update(dict(p_moist_fn=cfg.p_moist_fn))

    simulator = Propagator(
        dem=dem,
        veg=veg,
        realizations=cfg.realizations,
        fuels=cfg.fuel_system,
        do_spotting=cfg.do_spotting,
        **args,
    )

    non_vegetated = cfg.fuel_system.get_non_vegetated()
    boundary_conditions_list = cfg.get_boundary_conditions(
        geo_info, non_vegetated
    )
    for boundary_condition in boundary_conditions_list:
        simulator.set_boundary_conditions(boundary_condition)

    while True:
        next_time = simulator.next_time()
        if next_time is None:
            break

        simulator.step()

        if simulator.time % cfg.time_resolution == 0:
            ref_date = cfg.init_date + timedelta(minutes=int(simulator.time))
            info_msg(f"Time: {simulator.time} -> {ref_date}")
            output = simulator.get_output()
            # Save the output to the specified folder
            writer.write_output(output)

        if simulator.time > cfg.time_limit:
            break


# %%
if __name__ == "__main__":
    main()
