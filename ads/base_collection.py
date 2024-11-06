from __future__ import print_function
from __future__ import annotations

import os
import carla
import shutil

import numpy as np
from omegaconf import OmegaConf
from typing import List

from scenario.utils.timer import GameTime
from scenario.utils.route_manipulation import downsample_route
from scenario.utils.carla_data_provider import CarlaDataProvider
from scenario.utils.route_tool import interpolate_trajectory
from scenario.utils.misc import convert_transform_to_location
from ads.sensors import BaseSensorInterface
from ads.sensors import BaseCallBack

from .tools.saving_utils import DataWriter
from .tools.bev_observer import BEVObserver, RunStopSign
from .tools.meta_observer import MetaObserver


class AutonomousAgent(object):

    def __init__(
            self,
            save_root: str,
            seed_id: str,
            path_to_conf_file,
            route_config,
            require_saver: bool = False
    ):
        self.save_folder = save_root
        self.require_saver = require_saver
        self.seed_id = str(seed_id)
        self.wallclock_t0 = None
        # this data structure will contain all sensor data
        self.sensor_interface = BaseSensorInterface()  # can be replaced by other implementations
        self.route_config = route_config

        self._global_route = None
        self._global_plan = None
        self._global_plan_world_coord = None
        self.set_global_plan()

        # inner parameters
        self.step = -1
        self._perception_range = 50.0

        if self.require_saver:
            self.save_path = os.path.join(self.save_folder, 'agent_data', self.seed_id)
            if os.path.exists(self.save_path):
                shutil.rmtree(self.save_path)
            os.makedirs(self.save_path)

            self.bev_observer = BEVObserver()
            self.meta_observer = MetaObserver()
            self.data_saver = DataWriter(self.save_path, render=True)

        # agent's initialization
        self.setup(path_to_conf_file)
        self.setup_collector()

    def setup(self, path_to_conf_file):
        pass

    def setup_collector(self):
        world = CarlaDataProvider.get_world()

        _bev_configs = {
            'width_in_pixels': 192,
            'pixels_ev_to_bottom': 40,
            'pixels_per_meter': 5.0,
            'history_idx': [-1],
            'scale_bbox': True,
            'scale_mask_col': 1.0
        }
        _criteria_stop = RunStopSign(world)

        self.bev_observer.setup(_bev_configs, _criteria_stop)
        self.meta_observer.setup()

    def _truncate_global_route_till_local_target(self, windows_size=5):
        """
        This function aims to: truncate global route from current location, find future waypoints
        :param windows_size: set the max comparison for saving time
        :return:
        """
        ego_vehicle = CarlaDataProvider.get_ego()
        ev_location = ego_vehicle.get_location()
        closest_idx = 0
        for i in range(len(self._global_route)-1):
            if i > windows_size:
                break
            loc0 = self._global_route[i][0].transform.location
            loc1 = self._global_route[i+1][0].transform.location
            wp_dir = loc1 - loc0
            wp_veh = ev_location - loc0
            dot_ve_wp = wp_veh.x * wp_dir.x + wp_veh.y * wp_dir.y + wp_veh.z * wp_dir.z
            if dot_ve_wp > 0:
                closest_idx = i+1

        if closest_idx > 0:
            self._last_route_location = carla.Location(self._global_route[0][0].transform.location)

        self._global_route = self._global_route[closest_idx:]

    def tick_save(
            self,
            step,
            curr_control,
            speed,
            features,
            gps,
            imu,
            target_point,  # array
            view_img,
            shift_distance=0.0,
            repaired_action=None
    ):
        ego_vehicle = CarlaDataProvider.get_ego()

        self._truncate_global_route_till_local_target()
        birdview_obs = self.bev_observer.get_observation(self._global_route)
        meta_obs = self.meta_observer.get_observation(self._global_route, curr_control)

        # a_t-1
        control = ego_vehicle.get_control()
        prev_action = np.array([control.throttle, control.steer, control.brake], dtype=np.float32)
        curr_action = np.array([curr_control.throttle, curr_control.steer, curr_control.brake], dtype=np.float32)

        save_dict = {
            'step': step,
            'action': curr_action,
            'speed': speed,
            'features': features,
            'gps': gps,
            'imu': imu,
            'target_point': target_point,
            'bev_masks': birdview_obs['masks'],
            'view_img': view_img,
            'bev_img': birdview_obs['rendered'],
            'meta_data': meta_obs,
            'shift_distance': shift_distance,
            'repaired_action': repaired_action,
            'prev_action': prev_action,
        }
        self.data_saver.write(**save_dict)

    def sensors(self):  # pylint: disable=no-self-use
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}
        ]

        """
        sensors = []

        return sensors

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        :return: control
        """
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False

        return control

    def destroy(self):
        if self.require_saver:
            self.data_saver.close()

    def __call__(self, *args, **kwargs):
        """
        Execute the agent call, e.g. agent()
        Returns the next vehicle controls
        """
        input_data = self.sensor_interface.get_data()
        timestamp = GameTime.get_time()
        if not self.wallclock_t0:
            self.wallclock_t0 = GameTime.get_wallclocktime()
        wallclock = GameTime.get_wallclocktime()
        wallclock_diff = (wallclock - self.wallclock_t0).total_seconds()

        if wallclock_diff % 10 == 0:
            print('\r======[Agent] Wallclock_time = {} / {} / Sim_time = {} / {}x'.format(wallclock,
                  wallclock_diff, timestamp, timestamp / (wallclock_diff + 0.001)), end='')

        control = self.run_step(input_data, timestamp)
        control.manual_gear_shift = False
        return control

    def set_global_plan(self):
        """
        Set the plan (route) for the agent
        """
        ego_waypoint_route = self.route_config
        world = CarlaDataProvider.get_world()
        gps_route, route, wp_route = interpolate_trajectory(world, ego_waypoint_route)
        CarlaDataProvider.set_ego_vehicle_route(convert_transform_to_location(route))
        ds_ids = downsample_route(route, 50)
        self._global_route = wp_route
        self._global_plan_world_coord = [(route[x][0], route[x][1]) for x in ds_ids]
        self._global_plan = [gps_route[x] for x in ds_ids]

    def get_sensor_callback(self):
        """
        Should be overridden in subclasses.
        :return:
        """
        return BaseCallBack

    @staticmethod
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
