from loguru import logger
from typing import Dict, Any
from dataclasses import dataclass, asdict

from .agent import EgoSection, VehicleSection, WalkerSection, StaticSection
from .weather import WeatherSection
from .map import MapSection
from .traffic_light import TrafficLight, TRAFFIC_LIGHT_SPACE

@dataclass
class ScenarioConfig:

    id: str
    ego_section: EgoSection
    vehicle_section: VehicleSection
    walker_section: WalkerSection
    static_section: StaticSection
    weather_section: WeatherSection
    map_section: MapSection
    traffic_light: TrafficLight

    def npc_ids(self):
        return self.vehicle_section.ids + self.walker_section.ids + self.static_section.ids

    def json_data(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_json(cls, json_node: Dict) -> 'ScenarioConfig':
        json_node['ego_section'] = EgoSection.from_json(json_node['ego_section'])
        json_node['vehicle_section'] = VehicleSection.from_json(json_node['vehicle_section'])
        json_node['walker_section'] = WalkerSection.from_json(json_node['walker_section'])
        json_node['static_section'] = StaticSection.from_json(json_node['static_section'])
        json_node['weather_section'] = WeatherSection.from_json(json_node['weather_section'])
        json_node['map_section'] = MapSection.from_json(json_node['map_section'])
        if 'traffic_light' not in json_node:
            # compatible to old seed
            json_node['traffic_light'] = {
                'pattern': None,
                'yellow_time': 2.0,
                'red_time': 1.0
            }
        else:
            if 'yellow_time' not in json_node['traffic_light']:
                json_node['traffic_light']['yellow_time'] = 2.0
            if 'red_time' not in json_node['traffic_light']:
                json_node['traffic_light']['red_time'] = 1.0
        json_node['traffic_light'] = TrafficLight.from_json(json_node['traffic_light'])
        return cls(**json_node)

@dataclass
class SeedConfig:

    id: str
    scenario: ScenarioConfig
    fitness: float
    oracle: Any

    def __init__(self, id: str, scenario: ScenarioConfig, fitness: float = 0.0, oracle: Any = None):
        self.id = id
        self.scenario = scenario
        self.fitness = fitness
        self.oracle = oracle

    def get_label(self):
        """
        0 or 1
        :return:
        """
        if self.oracle is None:
            logger.warning(f"{self.oracle} is None")
            return 1
        if self.oracle['complete']:
            return 0
        return 1

    def set_fitness(self, fitness: float):
        self.fitness = fitness

    def set_oracle(self, oracle: Any):
        self.oracle = oracle

    def json_data(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_json(cls, json_node: Dict) -> 'SeedConfig':
        if 'id' not in json_node:
            json_node['id'] = 'default'
        if 'fitness' not in json_node:
            json_node['fitness'] = 0.0
        if 'oracle' not in json_node:
            json_node['oracle'] = None
        json_node['scenario'] = ScenarioConfig.from_json(json_node['scenario'])
        return cls(**json_node)