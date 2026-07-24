import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Literal, Mapping, Optional
from warnings import warn

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, CliImplicitFlag, SettingsConfigDict
from pyproj import CRS

from propagator.cli.console import (
    info_msg,
    print_boundary_conditions_table,
    print_table,
    setup_console,
    status_propagator_msg,
)
from propagator.core import Propagator, PropagatorOutOfBoundsError
from propagator.core.constants import FUEL_SYSTEM_LEGACY_DICT, TILE_SIZE
from propagator.core.numba import (
    FUEL_SYSTEM_LEGACY,
    fuelsystem_from_dict,
)
from propagator.core.numba.models import FuelSystem
from propagator.core.runner import SimulationRunner
from propagator.io.configuration import PropagatorConfigurationLegacy
from propagator.io.loader.cog import PropagatorDataFromCogs
from propagator.io.loader.geotiff import PropagatorDataFromGeotiffs
from propagator.io.loader.protocol import PropagatorInputDataProtocol
from propagator.io.loader.tiles import PropagatorDataFromTiles
from propagator.io.writer import (
    GeoTiffWriter,
    MetadataJSONWriter,
)
from propagator.io.writer.isochrones_geojson import build_isochrones_writer
from propagator.io.writer.protocol import OutputWriter


# --- CLI configuration -------------------------------------------------------
class PropagatorCLILegacy(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True)

    config: Path = Field(..., description="Path to configuration file (JSON)")
    fuel_config: Optional[Path] = Field(
        None, description="Path to fuel configuration file (YAML)"
    )
    mode: Literal["tiles", "geotiff", "cog"] = Field(
        "tiles",
        description="Mode of static data load: 'tiles' for automatic, "
        "'geotiff' for giving DEM and FUEL in input, 'cog' for windowed "
        "reads from cloud-optimized GeoTIFFs with automatic domain "
        "growth when the fire reaches the boundary.",
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
    cog_dem: Optional[str] = Field(
        None,
        description="Comma-separated DEM COG URLs (s3://, https:// or "
        "paths), one per UTM zone; required in 'cog' mode",
    )
    cog_fuel: Optional[str] = Field(
        None,
        description="Comma-separated fuel COG URLs matching --cog-dem "
        "one to one; required in 'cog' mode",
    )
    grid_dim: int = Field(
        3072,
        gt=0,
        description="Initial window size in cells around the ignition "
        "('cog' mode)",
    )
    grow_margin: int = Field(
        512,
        gt=0,
        description="Cells added on every side when the fire reaches the "
        f"domain boundary ('cog' mode); must be a multiple of {TILE_SIZE}",
    )
    freeze_dir: Optional[Path] = Field(
        None,
        description="Directory for freezing burned-out tiles to disk "
        "(keeps memory proportional to the active front)",
    )
    seed: Optional[int] = Field(
        None,
        description="Seed for the simulation RNGs (reproducible for a "
        "fixed machine and numba thread count)",
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
    isochrones_mode: Literal["none", "single", "multiple", "jsonl"] = Field(
        "none",
        description="Isochrones output mode: 'none' (off, default), "
        "'single' (one consolidated file, rewritten each step), 'multiple' "
        "(one file per timestep, that step only), or 'jsonl' (one GeoJSON "
        "FeatureCollection per line, appended).",
    )
    isochrones_format: Literal["geojson", "gpkg"] = Field(
        "geojson",
        description="Isochrones file format for 'single'/'multiple' modes: "
        "'geojson' (default) or 'gpkg' (GeoPackage). 'jsonl' is always "
        "GeoJSON lines.",
    )

    record: CliImplicitFlag[bool] = Field(
        False,
        description="Export run logs",
    )

    ignore_out_of_bounds: CliImplicitFlag[bool] = Field(
        False,
        description="Continue simulation when reaching bounds.",
    )

    core: Literal["numba", "rust"] = Field(
        "numba",
        description="Simulation core: 'numba' (default) or 'rust' (native "
        "extension, requires the propagator_rust wheel).",
    )

    # Quiet mode to suppress console output, set to true when --quiet is passed
    verbose: CliImplicitFlag[bool] = Field(
        False,
        description="Enable verbose output",
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

        elif self.mode == "cog":
            if self.cog_dem is None or self.cog_fuel is None:
                raise ValueError(
                    "COG_DEM and COG_FUEL URL lists must be provided in "
                    "'cog' mode"
                )
            if self.grow_margin % TILE_SIZE:
                raise ValueError(
                    f"GROW_MARGIN must be a multiple of {TILE_SIZE} cells"
                )

        return self

    def cog_url_lists(self) -> tuple[list[str], list[str]]:
        assert self.cog_dem is not None and self.cog_fuel is not None
        dem_urls = [u.strip() for u in self.cog_dem.split(",") if u.strip()]
        fuel_urls = [u.strip() for u in self.cog_fuel.split(",") if u.strip()]
        return dem_urls, fuel_urls

    def build_configuration(self) -> PropagatorConfigurationLegacy:
        """Create configuration object from provided JSON file."""
        with open(self.config) as f:
            json_cfg = json.load(f)
        return PropagatorConfigurationLegacy(**json_cfg)


def _fuels_node_from_yaml(path: str | Path) -> Mapping:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    fuels_node = data.get("fuels")
    if not isinstance(fuels_node, Mapping):
        raise ValueError("YAML must contain 'fuels' (mapping)")
    return fuels_node


def fuels_from_yaml(path: str | Path) -> FuelSystem:
    # coerce IDs to int and build Fuel objects
    return fuelsystem_from_dict(_fuels_node_from_yaml(path))  # type: ignore


def fuel_dict_from_yaml(path: str | Path) -> dict:
    """Raw fuel definitions (config units) for the Rust core."""
    return {int(k): v for k, v in _fuels_node_from_yaml(path).items()}


# --- main function -----------------------------------------------------------
def main() -> None:
    simulation_time = datetime.now()
    start = time.time()

    # pydantic-settings is taking care of it
    cli = PropagatorCLILegacy()  # type: ignore

    if cli.record:
        setup_console(record_path=cli.output, basename="run")
    else:
        setup_console()

    if cli.verbose:
        info_msg(f"Run time: {simulation_time}")

    cfg = cli.build_configuration()

    if cli.verbose:
        table_data: dict[str, BaseModel | dict] = {
            "Run Info": {"Sim time": simulation_time.isoformat()},
            "CLI Args": cli,
            "Loaded Config": cfg,
        }
        print_table(
            table_data,
            title="Simulation Configuration",
            skip_fields=["boundary_conditions", "verbose"],  # too verbose
            header_style="bold green",
            section_style="bold yellow",
        )

    if cli.fuel_config is not None:
        fuel_system = fuels_from_yaml(cli.fuel_config)
        fuel_dict = fuel_dict_from_yaml(cli.fuel_config)
        if cli.verbose:
            info_msg(f"Fuel system loaded from {cli.fuel_config}")
    else:
        fuel_system = FUEL_SYSTEM_LEGACY
        fuel_dict = FUEL_SYSTEM_LEGACY_DICT
        if cli.verbose:
            info_msg("Using legacy fuel system")

    loader: PropagatorInputDataProtocol | None = None
    cog_loader: PropagatorDataFromCogs | None = None

    if cli.mode in ("tiles", "cog"):
        # first extract middle point from configuration
        mid_point = cfg.get_ignitions_middle_point()
        if mid_point is None:
            raise ValueError("Ignitions must be defined in the configuration.")
        mid_lat, mid_lon = mid_point[1], mid_point[0]

    if cli.mode == "tiles":
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
    elif cli.mode == "cog":
        dem_urls, fuel_urls = cli.cog_url_lists()
        cog_loader = PropagatorDataFromCogs(
            dem_urls=dem_urls,
            fuel_urls=fuel_urls,
            mid_lon=mid_lon,
            mid_lat=mid_lat,
            grid_dim=cli.grid_dim,
        )
        loader = cog_loader
    else:
        raise ValueError(f"Unknown mode: {cli.mode}")

    # Load the data
    dem = loader.get_dem()
    veg = loader.get_veg()
    geo_info = loader.get_geo_info()
    dst_crs = CRS.from_epsg(4326)

    raster_variables_mapping = {
        "fire_probability": lambda output: output.fire_probability,
        "mean_arrival_time": lambda output: output.mean_arrival_time,
        "min_arrival_time": lambda output: output.min_arrival_time,
        "fireline_intensity_mean": lambda output: output.fli_mean,
        "fireline_intensity_max": lambda output: output.fli_max,
        "ros_mean": lambda output: output.ros_mean,
        "ros_max": lambda output: output.ros_max,
    }
    if cfg.do_spotting:
        raster_variables_mapping.update(
            {
                "spotting_generation_probability": (
                    lambda output: output.spotting_generation_probability
                ),
                "spotting_receiving_probability": (
                    lambda output: output.spotting_receiving_probability
                ),
            }
        )

    # The isochrones writer is persistent across domain growth (it may
    # accumulate state); only its geo_info is refreshed when the grid grows.
    isochrones_writer = build_isochrones_writer(
        cli.isochrones_mode,
        start_date=cfg.init_date,
        output_folder=cli.output,
        prefix="isochrones",
        geo_info=geo_info,
        dst_crs=dst_crs,
        fmt=cli.isochrones_format,
        thresholds=cli.isochrones,
    )

    def build_writer(current_geo_info) -> OutputWriter:
        """(Re)build the shape-dependent writers; called again after every
        domain growth. The isochrones writer is reused (its geo_info is
        refreshed in grow_domain)."""
        if isochrones_writer is not None:
            isochrones_writer.geo_info = current_geo_info
        return OutputWriter(
            raster_writer=GeoTiffWriter(
                start_date=cfg.init_date,
                raster_variables_mapping=raster_variables_mapping,
                output_folder=cli.output,
                geo_info=current_geo_info,
                dst_crs=dst_crs,
            ),
            metadata_writer=MetadataJSONWriter(
                start_date=cfg.init_date,
                output_folder=cli.output,
                prefix="metadata",
            ),
            isochrones_writer=isochrones_writer,
        )

    writer = build_writer(geo_info)

    origin = cog_loader.initial_origin if cog_loader is not None else (0, 0)
    oob_mode = "ignore" if cli.ignore_out_of_bounds else "raise"

    if cli.core == "rust":
        from propagator.rust_core import Propagator as RustPropagator

        simulator = RustPropagator(
            dem=dem,
            veg=veg,
            realizations=cfg.realizations,
            fuels_dict=fuel_dict,
            do_spotting=cfg.do_spotting,
            origin=origin,
            seed=cli.seed,
            freeze_dir=cli.freeze_dir,
            out_of_bounds_mode=oob_mode,
            ros_model=cfg.ros_model,
            moisture_model=cfg.prob_moist_model,
            cellsize=cfg.cellsize,
        )
        if cli.verbose:
            info_msg("Using Rust core (propagator_rust)")
    else:
        simulator = Propagator(
            dem=dem,
            veg=veg,
            realizations=cfg.realizations,
            fuels=fuel_system,
            do_spotting=cfg.do_spotting,
            origin=origin,
            seed=cli.seed,
            freeze_dir=cli.freeze_dir,
            out_of_bounds_mode=oob_mode,
            p_time_fn=cfg.p_time_fn if cfg.p_time_fn is not None else None,
            p_moist_fn=cfg.p_moist_fn if cfg.p_moist_fn is not None else None,
        )

    non_vegetated = fuel_system.get_non_vegetated()
    boundary_conditions_list = cfg.get_boundary_conditions(
        geo_info, non_vegetated
    )
    for boundary_condition in boundary_conditions_list:
        simulator.set_boundary_conditions(boundary_condition)

    if cli.verbose:
        print_boundary_conditions_table(cfg.boundary_conditions)

    def _on_domain_grow(new_geo_info) -> None:
        nonlocal writer
        writer = build_writer(new_geo_info)

    runner = SimulationRunner(
        simulator=simulator,
        cog_loader=cog_loader,
        grow_margin=cli.grow_margin,
        freeze_dir=cli.freeze_dir,
        on_domain_grow=_on_domain_grow,
        verbose=cli.verbose,
    )

    while True:
        next_time = simulator.next_time()
        if next_time is None:
            break

        target_time = simulator.time + cfg.time_resolution
        try:
            output = runner.advance_to(target_time)
        except PropagatorOutOfBoundsError as e:
            warn(f"Simulation stopped due to PropagatorOutOfBoundsError: {e}")
            break

        status_propagator_msg(
            cfg.init_date,
            output.time,
            output.stats,
            verbose=cli.verbose,
        )

        writer.write_output(output)

        if simulator.time > cfg.time_limit:
            break

    end = time.time()
    if cli.verbose:
        info_msg(f"Execution time: {end - start:.2f} seconds")


# %%
if __name__ == "__main__":
    main()
