import carla

from typing import List
from importlib import import_module

from scenario.configuration import WaypointUnit

def load_entry_point(name):
    mod_name, attr_name = name.split(":")
    mod = import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn

def convert_waypoint_unit_to_carla_transform(waypoints: List[WaypointUnit]):
    carla_transforms = list()

    for i, wp in enumerate(waypoints):
        carla_transforms.append(
            carla.Transform(
                location = carla.Location(
                    x=wp.position.x,
                    y=wp.position.y,
                    z=wp.position.z
                ),
                rotation = carla.Rotation(
                    pitch=wp.position.pitch,
                    yaw=wp.position.yaw,
                    roll=wp.position.roll
                )
            )
        )
    return carla_transforms