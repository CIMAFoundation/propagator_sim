from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Protocol

from pyproj import Proj

from propagator.core.models import PropagatorOutput
from propagator.io.geo import GeographicInfo


class BaseWriterProtocol(Protocol):
    start_date: datetime

    def ref_date(self, output: PropagatorOutput) -> datetime:
        ref_date = self.start_date + timedelta(seconds=int(output.time))
        return ref_date


class RasterWriterProtocol(BaseWriterProtocol):
    geo_info: GeographicInfo
    dst_prj: Proj

    def write_rasters(
        self,
        output: PropagatorOutput,
    ) -> None: ...


class MetadataWriterProtocol(BaseWriterProtocol):
    def write_metadata(self, output: PropagatorOutput) -> None: ...


class IsochronesWriterProtocol(BaseWriterProtocol):
    geo_info: GeographicInfo
    dst_prj: Proj

    def write_isochrones(
        self,
        output: PropagatorOutput,
    ) -> None: ...


@dataclass
class OutputWriter:
    raster_writer: Optional[RasterWriterProtocol] = None
    metadata_writer: Optional[MetadataWriterProtocol] = None
    isochrones_writer: Optional[IsochronesWriterProtocol] = None

    def write_output(self, output: PropagatorOutput) -> None:
        if self.raster_writer:
            self.raster_writer.write_rasters(output)

        if self.metadata_writer:
            self.metadata_writer.write_metadata(output)

        if self.isochrones_writer:
            self.isochrones_writer.write_isochrones(output)
