import cv2

import pylot.utils
import numpy as np

from collections import deque
from pylot.planning.waypoints import Waypoints

class DisplayInterface(object):

    def __init__(self):
        self._width = 1200
        self._height = 400
        self._surface = None

    def run_interface(self, input_data):
        rgb_front = cv2.cvtColor(input_data['recorder_rgb_front'][1][:, :, :3], cv2.COLOR_BGR2RGB)

        rgb = cv2.resize(rgb_front, (1200, 400))
        surface = np.zeros((400, 1200, 3), np.uint8)
        surface[:, :1200] = rgb

        if 'vis_text' in input_data:
            vis_text = input_data['vis_text']
            surface = cv2.putText(surface, vis_text['basic_info'], (20, 290), cv2.FONT_HERSHEY_TRIPLEX, 0.6,
                                  (0, 0, 255), 1)
            # surface = cv2.putText(surface, vis_text['plan'], (20, 710), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0, 0, 255), 1)
            surface = cv2.putText(surface, vis_text['control'], (20, 330), cv2.FONT_HERSHEY_TRIPLEX, 0.6,
                                  (0, 0, 255),
                                  1)
            surface = cv2.putText(surface, vis_text['others'], (20, 370), cv2.FONT_HERSHEY_TRIPLEX, 0.6,
                                  (0, 0, 255),
                                  1)

        return surface

def get_recorder_sensors():
    return [
        {
            'type': 'sensor.camera.rgb',
            'x': - 6.0, 'y': 0.0, 'z': 2.0,
            'roll': 0.0, 'pitch': -15.0, 'yaw': 0.0,
            'width': 1200, 'height': 400, 'fov': 100,
            'id': 'recorder_rgb_front'
        }
    ]

def convert_global_route(global_word_coord):
    waypoints = deque([])
    road_options = deque([])
    for (transform, road_option) in global_word_coord:
        waypoints.append(pylot.utils.Transform.from_simulator_transform(transform))
        road_options.append(pylot.utils.RoadOption(road_option.value))
    waypoints = Waypoints(waypoints, road_options=road_options)
    return waypoints

def cut_waypoints(ego_transform, route_waypoints):
    ego_transform = pylot.utils.Transform.from_simulator_transform(ego_transform)
    route_waypoints.remove_completed(ego_transform.location, ego_transform)
    return route_waypoints