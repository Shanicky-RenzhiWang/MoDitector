"""
This select route for the ego
"""
import carla

from dataclasses import dataclass, asdict
from typing import Dict, List

@dataclass
class WeatherSection:
    cloudiness: float  # [0, 100]
    precipitation: float  # [0, 100]
    precipitation_deposits: float  # [0, 100]
    wind_intensity: float  # [0, 100]
    sun_azimuth_angle: float  # [0, 360]
    sun_altitude_angle: float  # [-90, +90]
    fog_density: float  # [0, 100]
    fog_distance: float  # [0, np.inf]
    wetness: float  # [0, 100]
    fog_falloff: float  # [0, np.inf]]

    @property
    def carla_parameters(self):
        # carla format
        return carla.WeatherParameters(self.cloudiness, self.precipitation, self.precipitation_deposits,
                                       self.wind_intensity, self.sun_azimuth_angle, self.sun_altitude_angle,
                                       self.fog_density, self.fog_distance, self.wetness, self.fog_falloff)

    def json_data(self):
        return asdict(self)

    @classmethod
    def from_json(cls, json_node: Dict):
        return cls(**json_node)