import logging
from typing import Any
from attr import dataclass, field
from pyproj import Proj, transform
import rasterio as rio
import numpy as np
from scipy import ndimage
import utm

from propagator.geo import GeographicInfo
from propagator.propagator import PropagatorError
from propagator.settings import DEFAULT_TILES_TAG
from propagator.utils import add_point, rasterize_actions, read_actions


class PropagatorDataLoaderException(Exception): ...


# def load_ignitions_from_string(self, ignition_string):
#     mid_lat, mid_lon, polys, lines, points = read_actions(ignition_string)
#     easting, northing, zone_number, zone_letter = utm.from_latlon(mid_lat, mid_lon)
#     return easting, northing, zone_number, zone_letter, polys, lines, points


# WATERLINE_ACTION_TAG = "waterline_action"
# HEAVY_ACTION_TAG = "heavy_action"
# HELICOPTER_TAG = "helicopter"
# CANADAIR_TAG = "canadair"
# MOISTURE_TAG = "moisture"

# WATERLINE_ACTION_VALUE = 0.27
# CANADAIR_VALUE = 0.25
# CANADAIR_BUFFER_VALUE = 0.22
# HELICOPTER_VALUE = 0.22
# HELICOPTER_BUFFER_VALUE = 0.2


# def rasterize_fighting_actions_moisture(
#     bc: dict[str, Any], geo_info: GeographicInfo
# ) -> np.ndarray:
#     """
#     Rasterize the fighting actions from the boundary conditions.
#     :param bc: Boundary conditions dictionary
#     :param geo_info: Geographic information object
#     :return: Rasterized additional moisture due to fighting actions
#     """
#     west, south, east, north = geo_info.bounds
#     step_x, step_y = geo_info.get_stepx_stepy()

#     additional_moisture = np.zeros(geo_info.shape, dtype=np.float32)

#     for tag in (WATERLINE_ACTION_TAG, CANADAIR_TAG, HELICOPTER_TAG, HEAVY_ACTION_TAG):
#         actions_strings = bc.get(WATERLINE_ACTION_TAG, None)
#         if actions_strings is None:
#             continue

#         action_string = "\n".join(actions_strings)
#         mid_lat, mid_lon, polys, lines, points = read_actions(action_string)

#         easting, northing, zone_number, zone_letter = utm.from_latlon(mid_lat, mid_lon)

#         rasterizd, points = rasterize_actions(
#             geo_info.shape,
#             points,
#             lines,
#             polys,
#             west,
#             north,
#             step_x,
#             step_y,
#             zone_number,
#             base_value=1,
#         )
#         mask = rasterizd == 1

#         if tag == WATERLINE_ACTION_TAG:
#             # create a 1 pixel buffer around the selected points
#             img_mask = ndimage.binary_dilation(mask)
#             # moisture value of the points of the buffer
#             additional_moisture[img_mask] += WATERLINE_ACTION_VALUE
#         elif tag == CANADAIR_TAG:
#             # create a 1 pixel buffer around the selected points
#             img_buffer_mask = ndimage.binary_dilation(mask)
#             # moisture value of the points of the buffer
#             additional_moisture[img_buffer_mask & ~mask] += CANADAIR_BUFFER_VALUE
#             # moisture value of the points directly interested by the canadair actions
#             additional_moisture[mask] += CANADAIR_VALUE
#         elif tag == HELICOPTER_TAG:
#             mask_around = mask.copy()
#             for (
#                 point
#             ) in points:  # create a randomness in the points where the helicopter acts
#                 new_x = point[0] - 1 + round(2 * np.random.uniform())
#                 new_y = point[1] - 1 + round(2 * np.random.uniform())
#                 add_point(mask_around, new_y, new_x, 0.6)
#                 # create a 1 pixel buffer around the selected points
#                 img_new_heli_mask = ndimage.binary_dilation(mask_around)
#                 # moisture value of the points of the buffer
#                 additional_moisture[img_new_heli_mask & ~mask] += (
#                     HELICOPTER_BUFFER_VALUE
#                 )
#                 # moisture value of the points directly interested by the helicopter actions
#                 additional_moisture[mask] += HELICOPTER_VALUE

#         return additional_moisture


# def rasterize_heavy_actions(bc: dict[str, Any], geo_info: GeographicInfo) -> np.ndarray:
#     """
#     Rasterize the heavy actions from the boundary conditions.
#     :param bc: Boundary conditions dictionary
#     :param geo_info: Geographic information object
#     :return: Rasterized heavy actions
#     """
#     west, south, east, north = geo_info.bounds
#     step_x, step_y = geo_info.get_stepx_stepy()


# def __rasterize_newignitions(self, bc):
#     west, south, east, north = self.__bounds
#     new_ignitions = bc.get(IGNITIONS_TAG, None)

#     if new_ignitions:
#         new_ignitions_string = "\n".join(new_ignitions)
#         mid_lat, mid_lon, polys, lines, points = read_actions(new_ignitions_string)
#         easting, northing, zone_number, zone_letter = utm.from_latlon(mid_lat, mid_lon)

#         img, ignition_pixels = rasterize_actions(
#             (self.__shape[0], self.__shape[1]),
#             points,
#             lines,
#             polys,
#             west,
#             north,
#             self.step_x,
#             self.step_y,
#             zone_number,
#         )

#         bc[IGNITIONS_RASTER_TAG] = ignition_pixels


# def rasterize_ignitions():
#     west, south, east, north = self.__bounds

#     initial_ignitions, active_ignitions = rasterize_actions(
#         (self.shape[0], self.shape[1]),
#         points,
#         lines,
#         polys,
#         west,
#         north,
#         self.step_x,
#         self.step_y,
#         zone_number,
#     )
#     self.__preprocess_bc(self.settings.boundary_conditions)


# normalize((180 - w_dir_deg + 90) * np.pi / 180.0)
