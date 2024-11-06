from __future__ import print_function

import os
import carla
import pathlib
import pickle
import math
import shutil
import numpy as np

from PIL import Image
from typing import List, Dict

from scenario.configuration import WaypointUnit
from scenario.utils.timer import GameTime
from scenario.utils.route_manipulation import downsample_route
from scenario.utils.carla_data_provider import CarlaDataProvider
from scenario.utils.route_tool import interpolate_trajectory
from scenario.utils.misc import convert_transform_to_location, get_forward_value, rotate_point_traffic_signal
from ads.sensors import BaseSensorInterface
from ads.tools.recorder_tools import DisplayInterface
from ads.sensors import BaseCallBack

class AutonomousAgent(object):

    def __init__(
            self,
            save_root: str,
            seed_id: str,
            path_to_conf_file,
            route_config: List[WaypointUnit],
            require_saver: bool = False
    ):

        self.require_saver = require_saver
        self.save_folder = os.path.join(save_root, 'agent_data')
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        self.seed_id = seed_id
        self.wallclock_t0 = None
        # this data structure will contain all sensor data
        self.sensor_interface = BaseSensorInterface() # can be replaced by other implementations
        self.route_config = route_config

        self._global_route = None
        self._global_plan = None
        self._global_plan_world_coord = None
        self.set_global_plan()

        # inner parameters
        self.step = -1
        self._perception_range = 50.0

        if self.require_saver:
            self._hic = DisplayInterface()
            # create folders
            self.save_path = pathlib.Path(self.save_folder) / self.seed_id
            if os.path.exists(self.save_path):
                shutil.rmtree(self.save_path)
            self.save_path.mkdir(parents=True, exist_ok=False)
            # for visualization
            (self.save_path / 'monitor_view').mkdir(parents=True, exist_ok=True)
            (self.save_path / 'key_frames').mkdir(parents=True, exist_ok=True)
            (self.save_path / 'meta').mkdir(parents=True, exist_ok=True)

        # agent's initialization
        self.setup(path_to_conf_file)

    def setup(self, path_to_conf_file):
        pass

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
        """
        Destroy (clean-up) the agent
        :return:
        """
        pass

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

        print('\r======[Agent] Wallclock_time = {} / {} / Sim_time = {} / {}x'.format(wallclock, wallclock_diff, timestamp, timestamp / (wallclock_diff + 0.001)), end='')

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

    def save_meta_data(self, tick_data: Dict, ego_control):
        ego_vehicle = CarlaDataProvider.get_ego()
        ego_location = ego_vehicle.get_location()
        ego_transform = ego_vehicle.get_transform()
        ego_velocity = ego_vehicle.get_velocity()
        ego_acceleration = ego_vehicle.get_acceleration()
        ego_speed = get_forward_value(transform=ego_transform, velocity=ego_velocity)  # In m/s
        ego_head_acc = get_forward_value(transform=ego_transform, velocity=ego_acceleration)
        original_control = [ego_control.throttle, ego_control.steer, ego_control.brake]
        vis_data = {
            "basic_info": f"(Basic) x: {ego_location.x:.2f} y: {ego_location.y:.2f} speed: {ego_speed:.2f} acc: {ego_head_acc:.2f}",
            "control": f"(Control) throttle: {original_control[0]:.2f} steer: {original_control[1]:.2f} brake: {original_control[2]:.2f}",
            "others": f"None"
        }
        tick_data['vis_text'] = vis_data

        frame = self.step
        image = self._hic.run_interface(tick_data)
        Image.fromarray(image).save(self.save_path / 'monitor_view' / ('%04d.png' % frame))

        if self.step % 10 == 0:
            ego_route = self.get_ego_route(ego_vehicle)
            meta_data = self.get_meta_data(
                ego_vehicle,
                ego_route,
                ego_control)
            data_path = (self.save_path / 'meta' / ('%04d.pkl' % frame))
            with open(data_path, 'wb') as f:
                pickle.dump(meta_data, f)
            Image.fromarray(image).save(self.save_path / 'key_frames' / ('%04d.png' % frame))

    def get_ego_route(self, ego_vehicle) -> List[carla.Waypoint]:
        """
        Should be overwritten for other ADSs.
        :return:
        """
        ego_route = list()
        for i in range(len(self._global_route)):
            ego_route.append(self._global_route[i][0])
        ego_route = self._truncate_plan(ego_vehicle, ego_route)
        return ego_route

    def get_meta_data(
        self,
        ego_vehicle: carla.Actor,
        ego_route: List[carla.Waypoint],
        ego_current_control: carla.VehicleControl
    ):
        world = CarlaDataProvider.get_world()
        # ego_route: pylot waypoint format
        # NOTE that: the ego route is from now to destination
        scene_map = CarlaDataProvider.get_map()
        scene_data = dict()
        # 1. ego info
        ego_info = dict()
        ego_transform = ego_vehicle.get_transform()
        ego_wp = scene_map.get_waypoint(ego_transform.location)
        ego_velocity = ego_vehicle.get_velocity()
        ego_acceleration = ego_vehicle.get_acceleration()
        ego_angular_velocity = ego_vehicle.get_angular_velocity()
        ego_last_control = ego_vehicle.get_control()
        ego_bbox = ego_vehicle.bounding_box
        ego_type_id = ego_vehicle.type_id
        ego_info['type_id'] = ego_type_id
        ego_info['location'] = [ego_transform.location.x, ego_transform.location.y, ego_transform.location.z]
        ego_info['rotation'] = [ego_transform.rotation.pitch, ego_transform.rotation.yaw, ego_transform.rotation.roll]
        ego_info['velocity'] = [ego_velocity.x, ego_velocity.y, ego_velocity.z]
        ego_info['acceleration'] = [ego_acceleration.x, ego_acceleration.y, ego_acceleration.z]
        ego_info['angular_velocity'] = [ego_angular_velocity.x, ego_angular_velocity.y, ego_angular_velocity.z]
        ego_info['last_control'] = [ego_last_control.throttle, ego_last_control.steer, ego_last_control.brake]
        ego_info['current_control'] = [ego_current_control.throttle, ego_current_control.steer, ego_current_control.brake]
        ego_info['bbox'] = {
            'location': [ego_bbox.location.x, ego_bbox.location.y, ego_bbox.location.z],
            'extent': [ego_bbox.extent.x, ego_bbox.extent.y, ego_bbox.extent.z]
        }
        ego_info['matrix'] = np.array(ego_transform.get_matrix()).astype(np.float32)
        ego_info['inverse_matrix'] = np.array(ego_transform.get_inverse_matrix()).astype(np.float32)

        # get road info
        if ego_wp.is_junction:
            junction_distance = 0.0
        else:
            # calculate the distance to the next junction
            find_junction = False
            junction_distance = 100.0
            waypoint = ego_wp
            for i in range(int(self._perception_range) + 1):
                next_wps = waypoint.next(1.0)
                if next_wps is None or len(next_wps) == 0:
                    break
                if len(next_wps) > 1:
                    find_junction = True
                    junction_distance = (i + 1) * 1.0
                else:
                    wp = next_wps[0]
                    if wp.is_junction:
                        find_junction = True
                        junction_distance = (i + 1) * 1.0

                if find_junction:
                    break

                waypoint = next_wps[0]

        # determine lane type
        found_lanes = [f"{ego_wp.road_id}_{ego_wp.section_id}_{ego_wp.lane_id}"]
        l = ego_wp.get_left_lane()
        l_count = 0
        while l and l.lane_type == carla.LaneType.Driving:
            l_id = f"{l.road_id}_{l.section_id}_{l.lane_id}"
            if l_id in found_lanes:
                break
            l_count += 1
            found_lanes.append(l_id)
            # print(target_point)
            l = l.get_left_lane()
        found_lanes = [f"{ego_wp.road_id}_{ego_wp.section_id}_{ego_wp.lane_id}"]
        r = ego_wp.get_right_lane()
        r_count = 0
        while r and r.lane_type == carla.LaneType.Driving:
            r_id = f"{r.road_id}_{r.section_id}_{r.lane_id}"
            if r_id in found_lanes:
                break
            r_count += 1
            found_lanes.append(r_id)
            r = r.get_right_lane()
        lane_num = 1 + l_count + r_count

        ego_info['road_info'] = {
            "section_id": ego_wp.section_id,
            "road_id": ego_wp.road_id,
            "lane_id": ego_wp.lane_id,
            "is_junction": ego_wp.is_junction,
            "junction_id": ego_wp.junction_id,
            "dist2junction": junction_distance,
            "s": ego_wp.s,
            "lane_num": lane_num,
            "lane_order": l_count + 1 # from left
        }

        ego_info['route'] = list()
        for iwi in range(len(ego_route)):
            wp = ego_route[iwi]
            ego_info['route'].append([wp.transform.location.x, wp.transform.location.y])

        # ! Add ego info
        scene_data['ego'] = ego_info

        # 2. traffic actors
        actor_dict = dict()
        actors = world.get_actors()
        for actor_type in ["vehicle", "walker", "traffic_light", "stop"]: # ignore static
            selected_actors = actors.filter('*' + actor_type + '*')
            for selected_actor in selected_actors:
                if (actor_type != "vehicle") or (selected_actor.id != ego_vehicle.id):
                    selected_actor_transform = selected_actor.get_transform()
                    selected_actor_location = selected_actor_transform.location
                    selected_actor_rotation = selected_actor_transform.rotation
                    selected_actor_id = selected_actor.id
                    if hasattr(selected_actor, 'bounding_box'):
                        selected_actor_bbox_location = [
                            selected_actor.bounding_box.location.x,
                            selected_actor.bounding_box.location.y,
                            selected_actor.bounding_box.location.z
                        ]
                        selected_actor_extent = [
                            selected_actor.bounding_box.extent.x,
                            selected_actor.bounding_box.extent.y,
                            selected_actor.bounding_box.extent.z
                        ]
                    elif hasattr(selected_actor, 'trigger_volume'):
                        selected_actor_bbox_location = [
                            selected_actor.trigger_volume.location.x,
                            selected_actor.trigger_volume.location.y,
                            selected_actor.trigger_volume.location.z
                        ]
                        selected_actor_extent = [
                            selected_actor.trigger_volume.extent.x,
                            selected_actor.trigger_volume.extent.y,
                            selected_actor.trigger_volume.extent.z
                        ]
                    else:
                        selected_actor_bbox_location = [0.0, 0.0, 0.0]
                        selected_actor_extent = [0.5, 0.5, 2]

                    selected_actor_velocity = selected_actor.get_velocity()
                    selected_actor_acceleration = selected_actor.get_acceleration()
                    selected_actor_brake = None

                    if actor_type in ["vehicle", "walker"]:
                        if actor_type == "vehicle":
                            selected_actor_control = selected_actor.get_control()
                            selected_actor_brake = selected_actor_control.brake

                    light_state = None
                    if actor_type == 'traffic_light' or actor_type == 'stop':
                        if actor_type == 'traffic_light':
                            light_state_carla = selected_actor.get_state()
                            if light_state_carla == carla.TrafficLightState.Red:
                                light_state = 'red'
                            elif light_state_carla == carla.TrafficLightState.Yellow:
                                light_state = 'yellow'
                            elif light_state_carla == carla.TrafficLightState.Green:
                                light_state = 'green'
                            elif light_state_carla == carla.TrafficLightState.Off:
                                light_state = 'off'
                            else:
                                light_state = 'unknown'

                        base_rot = selected_actor_transform.rotation.yaw
                        area_loc = selected_actor_transform.transform(selected_actor.trigger_volume.location)
                        area_ext = selected_actor.trigger_volume.extent

                        point = rotate_point_traffic_signal(carla.Vector3D(0, 0, area_ext.z), math.radians(base_rot))
                        trigger_point_location = area_loc + carla.Location(x=point.x, y=point.y)
                        trigger_location = trigger_point_location
                        actor_wp = scene_map.get_waypoint(trigger_location)
                    else:
                        actor_wp = scene_map.get_waypoint(selected_actor_location)
                        trigger_location = selected_actor_location

                    selected_actor_type_id = selected_actor.type_id
                    actor_result = {
                        "class": actor_type,
                        "type_id": selected_actor_type_id,
                        "location": [selected_actor_location.x, selected_actor_location.y, selected_actor_location.z],
                        "trigger_location": [trigger_location.x, trigger_location.y, trigger_location.z],
                        "rotation": [selected_actor_rotation.pitch, selected_actor_rotation.yaw,
                                     selected_actor_rotation.roll],
                        "velocity": [selected_actor_velocity.x, selected_actor_velocity.y, selected_actor_velocity.z],
                        "acceleration": [selected_actor_acceleration.x, selected_actor_acceleration.y,
                                         selected_actor_acceleration.z],
                        "bbox": {
                            "location": selected_actor_bbox_location,
                            "extent": selected_actor_extent
                        },
                        "brake": selected_actor_brake,
                        "id": int(selected_actor_id),
                        "matrix": np.array(selected_actor_transform.get_matrix()).astype(np.float32),
                        "inverse_matrix": np.array(selected_actor_transform.get_inverse_matrix()).astype(np.float32),
                        "light_state": light_state,
                        "road_info": {
                            "section_id": actor_wp.section_id,
                            "road_id": actor_wp.road_id,
                            "lane_id": actor_wp.lane_id,
                            "is_junction": actor_wp.is_junction,
                            "junction_id": actor_wp.junction_id,
                            "s": actor_wp.s
                        }
                    }

                    actor_dict[f"{actor_type}_{str(selected_actor_id)}"] = actor_result
        scene_data['traffic'] = actor_dict
        return scene_data

    @staticmethod
    def _truncate_plan(ego_actor: carla.Actor, plan: List[carla.Waypoint]) -> List:

        ego_location = ego_actor.get_location()
        closest_idx = 0
        closest_dist = np.inf
        for index, waypoint in enumerate(plan):
            if index == len(plan) - 1:
                break
            loc0 = plan[index].transform.location
            loc1 = plan[index + 1].transform.location
            wp_dir = loc1 - loc0
            wp_veh = ego_location - loc0
            dot_ve_wp = wp_veh.x * wp_dir.x + wp_veh.y * wp_dir.y + wp_veh.z * wp_dir.z
            if dot_ve_wp > 0:
                dist = waypoint.transform.location.distance(ego_location)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_idx = index

        remain_plan = plan[closest_idx:]
        return remain_plan

    def get_sensor_callback(self):
        """
        Should be overridden in subclasses.
        :return:
        """
        return BaseCallBack