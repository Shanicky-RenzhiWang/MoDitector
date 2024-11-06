import carla
import numpy as np
import math

from dataclasses import dataclass
from shapely.geometry import Point
from typing import List

from scenario.utils.carla_data_provider import CarlaDataProvider

@dataclass
class Vector3D:
    x: float
    y: float
    z: float

    def distance(self, target):
        self_point = Point(self.x, self.y)
        target_point = Point(target.x, target.y)
        return self_point.distance(target_point)

def rotate_point_traffic_signal(point, radians):
    """
    rotate a given point by a given angle
    """
    rotated_x = math.cos(radians) * point.x - math.sin(radians) * point.y
    rotated_y = math.sin(radians) * point.x - math.cos(radians) * point.y

    return Vector3D(rotated_x, rotated_y, point.z)

class MetaObserver:

    def __init__(self):
        self._parent_actor = None
        self._world = None
        self._perception_range = 50.0

    def setup(self):
        self._parent_actor = None
        self._world = None
        self._perception_range = 50.0

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

    def attach_ego_vehicle(self, ego_vehicle):
        self._parent_actor = ego_vehicle
        self._world = self._parent_actor.get_world()

    def get_ego_route(self, route_plan) -> List[carla.Waypoint]:
        """
        Should be overwritten for other ADSs.
        :return:
        """
        ego_route = list()
        for i in range(len(route_plan)):
            ego_route.append(route_plan[i][0])
        ego_route = self._truncate_plan(self._parent_actor, ego_route)
        return ego_route

    def get_observation(self, route_plan, ego_current_control):
        ego_route = self.get_ego_route(route_plan)
        ego_vehicle = self._parent_actor
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
        ego_info['current_control'] = [ego_current_control.throttle, ego_current_control.steer,
                                       ego_current_control.brake]
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
            "lane_order": l_count + 1  # from left
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
        for actor_type in ["vehicle", "walker", "traffic_light", "stop"]:  # ignore static
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