import math
import carla

from typing import Dict, List
from collections import deque
from shapely.geometry import Polygon

from scenario.atomic.agents import RoadOption, VehiclePIDController
from scenario.configuration import WaypointUnit
from scenario.utils.misc import is_within_distance

class VehicleAgent:
    # basic waypoint follower
    # minimum distance to target waypoint as a percentage (e.g. within 90% of
    # total distance)
    MIN_DISTANCE_PERCENTAGE = 0.9

    def __init__(self, actor: carla.Actor, opt_dict: Dict):
        # actor & map
        self._actor = actor
        # logger.debug('Agent: {} Agent State: {}', self._actor, self._actor.is_alive)
        self._world = self._actor.get_world()
        self._map = self._world.get_map()
        # parameters
        self._dt = None # frequency, normally is 20 Hz
        self._sampling_radius = None # sample radius for next waypoint
        self._min_wp_distance = None # for filtering waypoint
        self._current_waypoint = None # current waypoint
        self._target_waypoint = None # target waypoint for next action
        self._target_speed = None # target speed for next action
        self._actor_controller = None # actor controller, i.e., PID
        # emergency stop variables
        self._ignore_traffic_light = False
        self._traffic_light_id_to_ignore = -1
        self._ignore_collision = False
        self._proximity_collision_threshold = None
        self._offset = 0.0
        self._use_bbs_detection = True

        # controller parameters
        self._max_brake = None
        self._max_throttle = None
        self._max_steer = None
        # queue of waypoints
        self._waypoint_queue = deque(maxlen=20000)
        self._buffer_size = 5 # buffer for waypoint queue
        self._waypoint_buffer = deque(maxlen=self._buffer_size)
        # initialization
        self._init_agent(opt_dict)

    def _init_agent(self, opt_dict: Dict):
        # default values
        self._dt = 1.0 / 20.0 # 20 Hz
        self._sampling_radius = 1.0  # 1 seconds horizon
        self._min_wp_distance = self._sampling_radius * self.MIN_DISTANCE_PERCENTAGE
        self._target_speed = 0.0
        self._max_brake = 1.0
        self._max_throttle = 0.75
        self._max_steer = 0.8
        args_lateral_dict = {
            'K_P': 1.0,
            'K_D': 0.01,
            'K_I': 0.0,
            'dt': self._dt}
        args_longitudinal_dict = {
            'K_P': 1.0,
            'K_D': 0,
            'K_I': 0.05,
            'dt': self._dt}
        self._ignore_traffic_light = False
        self._traffic_light_id_to_ignore = -1
        self._proximity_collision_threshold = 10.0
        self._offset = 0.0
        self._use_bbs_detection = True

        # parameters overload
        if opt_dict:
            if 'dt' in opt_dict:
                self._dt = opt_dict['dt']
            if 'sampling_radius' in opt_dict:
                self._sampling_radius = opt_dict['sampling_radius']
            if 'lateral_control_dict' in opt_dict:
                args_lateral_dict = opt_dict['lateral_control_dict']
            if 'longitudinal_control_dict' in opt_dict:
                args_longitudinal_dict = opt_dict['longitudinal_control_dict']
            if 'max_throttle' in opt_dict:
                self._max_throt = opt_dict['max_throttle']
            if 'max_brake' in opt_dict:
                self._max_brake = opt_dict['max_brake']
            if 'max_steering' in opt_dict:
                self._max_steer = opt_dict['max_steering']
            if 'ignore_traffic_light' in opt_dict:
                self._ignore_traffic_light = opt_dict['ignore_traffic_light']
            if 'ignore_collision' in opt_dict:
                self._ignore_collision = opt_dict['ignore_collision']
            if 'offset' in opt_dict:
                self._offset = opt_dict['offset']
            if 'use_bbs_detection' in opt_dict:
                self._use_bbs_detection = opt_dict['use_bbs_detection']

        self._current_waypoint = self._map.get_waypoint(self._actor.get_location())
        self._vehicle_controller = VehiclePIDController(self._actor,
                                                        args_lateral=args_lateral_dict,
                                                        args_longitudinal=args_longitudinal_dict,
                                                        max_throttle=self._max_throttle,
                                                        max_brake=self._max_brake,
                                                        max_steering=self._max_steer
                                                        )

    def set_global_route(self, route: List[WaypointUnit]):
        self._waypoint_queue.clear()
        for i, elem in enumerate(route):
            if i == 0:
                continue
            # WaypointUnit to carla waypoint
            elem_location = carla.Location(x=elem.x, y=elem.y, z=elem.z)
            elem_waypoint = self._map.get_waypoint(elem_location)
            self._waypoint_queue.append((elem_waypoint, elem.speed))

        # and the buffer
        self._waypoint_buffer.clear()
        for _ in range(self._buffer_size):
            if self._waypoint_queue:
                self._waypoint_buffer.append(
                    self._waypoint_queue.popleft())
            else:
                break

    def is_task_finished(self) -> bool:
        if len(self._waypoint_queue) == 0:
            return True
        return False

    def run_step(self, debug: bool):

        if len(self._waypoint_queue) == 0 and len(self._waypoint_buffer) == 0:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = False
            control.manual_gear_shift = False
            return control

        # Buffering the waypoints
        if not self._waypoint_buffer:
            for _ in range(self._buffer_size):
                if self._waypoint_queue:
                    self._waypoint_buffer.append(
                        self._waypoint_queue.popleft())
                else:
                    break

        # current actor waypoint
        actor_transform = self._actor.get_transform()
        self._current_waypoint = self._map.get_waypoint(actor_transform.location)
        # target waypoint
        self._target_waypoint, self._target_speed = self._waypoint_buffer[0]
        # print('target speed: {}', self._target_speed)
        # move using PID controllers, setting is m/s -> target_speed is km/h
        control = self._vehicle_controller.run_step(self._target_speed * 3.6, self._target_waypoint)
        # add collision detection & traffic light detection
        actor_location = self._actor.get_location()
        actor_waypoint = self._map.get_waypoint(actor_location)
        current_velocity = self._actor.get_velocity()
        current_speed = math.sqrt(current_velocity.x ** 2 + current_velocity.y ** 2 + current_velocity.z ** 2)

        traffic_light_stop = False
        if not self._ignore_traffic_light:
            traffic_light_stop = self._traffic_light_detection(actor_waypoint)
            if traffic_light_stop:
                control = carla.VehicleControl()
                control.steer = 0.0
                control.throttle = 0.0
                control.brake = 1.0
                control.hand_brake = False
                control.manual_gear_shift = False

        collision_stop = False
        if not self._ignore_collision:
            collision_stop = self._collision_detection(actor_waypoint, current_speed)
            if collision_stop:
                control = carla.VehicleControl()
                control.steer = 0.0
                control.throttle = 0.0
                control.brake = 1.0
                control.hand_brake = False
                control.manual_gear_shift = False

        # self.logger.info(
        #     f"{self._actor.id}, {self._actor.type_id}, route len: {len(self._waypoint_queue)} target speed: {self._target_speed}, current speed: {current_speed} traffic light: {traffic_light_stop} collision: {collision_stop}")

        # purge the queue of obsolete waypoints
        # update min_wp_distance
        self._min_wp_distance = current_speed * self._sampling_radius * self.MIN_DISTANCE_PERCENTAGE
        max_index = -1
        for i, (waypoint, _) in enumerate(self._waypoint_buffer):
            if waypoint.transform.location.distance(actor_transform.location) < self._min_wp_distance:
                max_index = i
        if max_index >= 0:
            for i in range(max_index + 1):
                self._waypoint_buffer.popleft()

        return control

    def _traffic_light_detection(self, waypoint: carla.Waypoint) -> bool:

        light_id = self._actor.get_traffic_light().id if self._actor.get_traffic_light() is not None else -1
        traffic_light_state = str(self._actor.get_traffic_light_state())

        if traffic_light_state == "Red":
            if not waypoint.is_junction and (self._traffic_light_id_to_ignore != light_id or light_id == -1):
                return True
            elif waypoint.is_junction and light_id != -1:
                self._traffic_light_id_to_ignore = light_id

        if self._traffic_light_id_to_ignore != light_id:
            self._traffic_light_id_to_ignore = -1

        return False

    def _collision_detection(self, waypoint: carla.Waypoint, speed: float) -> bool:
        """
        TODO: update this
        :param waypoint:
        :param speed:
        :return:
        """
        # ego_vehicle_waypoint= waypoint
        proximity_brake_distance = speed**2 # assume max acceleration is 4.0
        proximity_collision_threshold = max(self._proximity_collision_threshold, proximity_brake_distance)

        vehicle_list = self._world.get_actors().filter("*vehicle*")
        vehicle_collision = self._vehicle_obstacle_detected(vehicle_list, proximity_collision_threshold)
        if vehicle_collision:
            return True

        walker_list = self._world.get_actors().filter("*walker.pedestrian*")
        walker_collision = self._vehicle_obstacle_detected(walker_list, proximity_collision_threshold)

        if walker_collision:
            return True

        return False

    def get_incoming_waypoint_and_direction(self, steps=3):
        """
        Returns direction and waypoint at a distance ahead defined by the user.

            :param steps: number of steps to get the incoming waypoint.
        """
        if len(self._waypoint_queue) > steps:
            return self._waypoint_queue[steps]

        else:
            try:
                wpt, direction = self._waypoint_queue[-1]
                return wpt, direction
            except IndexError as i:
                return None, RoadOption.VOID

    def _vehicle_obstacle_detected(self, vehicle_list=None, max_distance=None, up_angle_th=90, low_angle_th=0, lane_offset=0):
        """
        Method to check if there is a vehicle in front of the agent blocking its path.

            :param vehicle_list (list of carla.Vehicle): list contatining vehicle objects.
                If None, all vehicle in the scene are used
            :param max_distance: max freespace to check for obstacles.
                If None, the base threshold value is used
        """

        def get_route_polygon():
            route_bb = []
            extent_y = self._actor.bounding_box.extent.y
            r_ext = extent_y + self._offset
            l_ext = -(extent_y + self._offset)
            r_vec = ego_transform.get_right_vector()
            p1 = ego_location + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
            p2 = ego_location + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
            route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]]) # z is 0 or not??

            for wp, _ in self._waypoint_queue:
                if ego_location.distance(wp.transform.location) > max_distance:
                    break
                r_vec = wp.transform.get_right_vector()
                p1 = wp.transform.location + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
                p2 = wp.transform.location + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
                route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

            # Two points don't create a polygon, nothing to check
            if len(route_bb) < 3:
                return None

            return Polygon(route_bb)

        if not max_distance:
            max_distance = self._proximity_collision_threshold

        ego_transform = self._actor.get_transform()
        ego_location = ego_transform.location
        ego_wpt = self._map.get_waypoint(ego_location)

        # Get the right offset
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

        # Get the transform of the front of the ego
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location(
            self._actor.bounding_box.extent.x * ego_transform.get_forward_vector())

        opposite_invasion = abs(self._offset) + self._actor.bounding_box.extent.y > ego_wpt.lane_width / 2
        use_bbs = self._use_bbs_detection or opposite_invasion or ego_wpt.is_junction

        # Get the route bounding box
        route_polygon = get_route_polygon()

        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._actor.id:
                continue

            target_transform = target_vehicle.get_transform()
            if target_transform.location.distance(ego_location) > max_distance:
                continue

            target_wpt = self._map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)

            # General approach for junctions and vehicles invading other lanes due to the offset
            if (use_bbs or target_wpt.is_junction) and route_polygon:

                target_bb = target_vehicle.bounding_box
                target_vertices = target_bb.get_world_vertices(target_vehicle.get_transform())
                target_list = [[v.x, v.y, v.z] for v in target_vertices]
                target_polygon = Polygon(target_list)

                if route_polygon.intersects(target_polygon):
                    return True

            # Simplified approach, using only the plan waypoints (similar to TM)
            else:
                if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id != ego_wpt.lane_id + lane_offset:
                    next_wpt, _ = self.get_incoming_waypoint_and_direction(steps=3)
                    if not next_wpt:
                        continue
                    if target_wpt.road_id != next_wpt.road_id or target_wpt.lane_id != next_wpt.lane_id + lane_offset:
                        continue

                target_forward_vector = target_transform.get_forward_vector()
                target_extent = target_vehicle.bounding_box.extent.x
                target_rear_transform = target_transform
                target_rear_transform.location -= carla.Location(
                    x=target_extent * target_forward_vector.x,
                    y=target_extent * target_forward_vector.y,
                )

                if is_within_distance(target_rear_transform, ego_front_transform, max_distance,
                                      [low_angle_th, up_angle_th]):
                    return True

        return False