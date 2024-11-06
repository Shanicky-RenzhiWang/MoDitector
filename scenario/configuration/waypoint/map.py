from typing import List, Dict
from dataclasses import dataclass, asdict

# from scenario.configuration.basic_config import WaypointUnit

@dataclass
class MapSection:

    town: str
    # update
    lanes_vehicle: List
    lanes_static: List
    nodes_walker: List
    edges_walker: List

    def __init__(self, town, lanes_vehicle, lanes_static, nodes_walker, edges_walker):
        self.town = town
        self.lanes_vehicle = lanes_vehicle
        self.lanes_static = lanes_static # put static obstacles or low speeding objects or hard brake objects
        self.nodes_walker = nodes_walker
        self.edges_walker = edges_walker

    def json_data(self):
        return asdict(self)

    @classmethod
    def from_json(cls, json_node: Dict):

        # vehicle_waypoints_js = json_node['vehicle_waypoints']
        # vehicle_waypoints = list()
        # for wp_js_node in vehicle_waypoints_js:
        #     vehicle_waypoints.append(WaypointUnit.from_json(wp_js_node))
        # json_node['vehicle_waypoints'] = vehicle_waypoints
        #
        # walker_waypoints_js = json_node['walker_waypoints']
        # walker_waypoints = list()
        # for wp_js_node in walker_waypoints_js:
        #     walker_waypoints.append(WaypointUnit.from_json(wp_js_node))
        # json_node['walker_waypoints'] = walker_waypoints

        return cls(**json_node)