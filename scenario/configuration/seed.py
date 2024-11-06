from typing import Dict, Any
from dataclasses import dataclass, asdict

from scenario.configuration.agent import EgoSection, VehicleSection, WalkerSection, StaticSection
from scenario.configuration.weather import WeatherSection
from scenario.configuration.map import MapSection

@dataclass
class ScenarioConfig:

    id: str
    ego_section: EgoSection
    vehicle_section: VehicleSection
    walker_section: WalkerSection
    static_section: StaticSection
    weather_section: WeatherSection
    map_section: MapSection

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
            return 1
        if self.oracle['complete'] is not True:
            return 1
        return 0

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