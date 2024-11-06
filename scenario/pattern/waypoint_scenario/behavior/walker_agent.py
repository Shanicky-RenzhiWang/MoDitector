import carla
import math

from typing import Dict, List
from collections import deque

from scenario.configuration import WaypointUnit

class WalkerAgent:
    # waypoint follower for walker: ignore traffic light and collision
    MIN_DISTANCE_PERCENTAGE = 0.9
    def __init__(self, actor: carla.Actor, opt_dict: Dict):
        self._actor = actor
        self._world = self._actor.get_world()
        self._map = self._world.get_map()
        # queue of waypoints
        self._waypoint_queue = deque(maxlen=20000)
        self._buffer_size = 5  # buffer for waypoint queue
        self._waypoint_buffer = deque(maxlen=self._buffer_size)
        # init some variables
        self._current_waypoint = self._map.get_waypoint(self._actor.get_location(), project_to_road = True, lane_type = carla.LaneType.Any)
        self._target_waypoint = None  # target waypoint for next action
        self._target_speed = 0.0
        self._sampling_radius = 1.0
        if opt_dict:
            if 'sampling_radius' in opt_dict:
                self._sampling_radius = opt_dict['sampling_radius']
        self._min_wp_distance = self._sampling_radius * self.MIN_DISTANCE_PERCENTAGE
        self.curr_route = None

    def set_global_route(self, route: List[WaypointUnit]):
        self.curr_route = route
        self._waypoint_queue.clear()
        for i, elem in enumerate(route):
            if i == 0:
                continue
            # WaypointUnit to carla waypoint
            elem_location = carla.Location(x=elem.x, y=elem.y, z=elem.z)
            elem_waypoint = self._map.get_waypoint(elem_location, project_to_road = True, lane_type = carla.LaneType.Any)
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
            if self._actor and self._actor.is_alive:
                control = self._actor.get_control()
            else:
                control = carla.WalkerControl()
            control.speed = 0.0
            return control
            # self.get_reverse_global_route()

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
        self._current_waypoint = self._map.get_waypoint(actor_transform.location, project_to_road = True, lane_type = carla.LaneType.Any)
        # target waypoint
        self._target_waypoint, self._target_speed = self._waypoint_buffer[0]
        # move using Walker Control
        location = self._target_waypoint.transform.location
        actor_location = self._actor.get_location()
        direction = location - actor_location
        direction_norm = math.sqrt(direction.x ** 2 + direction.y ** 2)
        control = self._actor.get_control()
        control.speed = self._target_speed
        control.direction = direction / direction_norm
        # add collision detection & traffic light detection
        current_velocity = self._actor.get_velocity()
        current_speed = math.sqrt(current_velocity.x ** 2 + current_velocity.y ** 2 + current_velocity.z ** 2)
        self._min_wp_distance = current_speed * self._sampling_radius * self.MIN_DISTANCE_PERCENTAGE
        max_index = -1
        for i, (waypoint, _) in enumerate(self._waypoint_buffer):
            if waypoint.transform.location.distance(actor_transform.location) < self._min_wp_distance:
                max_index = i
        if max_index >= 0:
            for i in range(max_index + 1):
                self._waypoint_buffer.popleft()

        return control
