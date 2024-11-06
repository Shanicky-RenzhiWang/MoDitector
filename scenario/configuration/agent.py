from typing import List, Dict
from dataclasses import dataclass

from scenario.configuration.basic import BasicAgent, BasicSection

EGO_BASE = 0
VEHICLE_BASE = 10000
WALKER_BASE = 20000
STATIC_BASE = 30000

######## Ego Agent & Section ########
@dataclass
class EgoSection(BasicSection):

    def __init__(self, agents: List[BasicAgent]):
        super().__init__(agents)
        self.id_base = EGO_BASE

    @classmethod
    def from_json(cls, json_node: Dict) -> 'EgoSection':
        agents_js = json_node['agents']
        agents = list()
        for a_i, a_js in enumerate(agents_js):
            agents.append(BasicAgent.from_json(a_js))
        create_json = dict()
        create_json['agents'] = agents
        return cls(**create_json)


######## NPC Vehicle Agent & Section ########
@dataclass
class VehicleSection(BasicSection):

    def __init__(self, agents: List[BasicAgent]):
        super().__init__(agents)
        self.id_base = VEHICLE_BASE

    @classmethod
    def from_json(cls, json_node: Dict) -> 'VehicleSection':
        agents_js = json_node['agents']
        agents = list()
        for a_i, a_js in enumerate(agents_js):
            agents.append(BasicAgent.from_json(a_js))
        create_json = dict()
        create_json['agents'] = agents
        return cls(**create_json)

######## NPC Walker Agent & Section ########
@dataclass
class WalkerSection(BasicSection):

    def __init__(self, agents: List[BasicAgent]):
        super().__init__(agents)
        self.id_base = WALKER_BASE

    @classmethod
    def from_json(cls, json_node: Dict) -> 'WalkerSection':
        agents_js = json_node['agents']
        agents = list()
        for a_i, a_js in enumerate(agents_js):
            agents.append(BasicAgent.from_json(a_js))
        create_json = dict()
        create_json['agents'] = agents
        return cls(**create_json)

######## NPC Static Agent & Section ########
@dataclass
class StaticSection(BasicSection):

    def __init__(self, agents: List[BasicAgent]):
        super().__init__(agents)
        self.id_base = STATIC_BASE

    @classmethod
    def from_json(cls, json_node: Dict) -> 'StaticSection':
        agents_js = json_node['agents']
        agents = list()
        for a_i, a_js in enumerate(agents_js):
            agents.append(BasicAgent.from_json(a_js))
        create_json = dict()
        create_json['agents'] = agents
        return cls(**create_json)