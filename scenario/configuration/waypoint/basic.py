import random

from dataclasses import dataclass, asdict
from typing import Dict, Optional, List, TypeVar, Any

@dataclass
class WaypointUnit:
    x: float
    y: float
    z: float
    pitch: float
    yaw: float
    roll: float

    speed: float = 0.0

    def json_data(self):
        return asdict(self)

    @classmethod
    def from_json(cls, json_node: Dict):

        if 'position' in json_node:
            json_node['x'] = float(json_node['position']['x'])
            json_node['y'] = float(json_node['position']['y'])
            json_node['z'] = float(json_node['position']['z'])
            json_node['pitch'] = float(json_node['position']['pitch'])
            json_node['yaw'] = float(json_node['position']['yaw'])
            json_node['roll'] = float(json_node['position']['roll'])

            del json_node['position']

        return cls(**json_node)

@dataclass
class ObjectInfo:
    model: str
    rolename: str
    color: Optional[str] # '({r}, {g}, {b})'
    category: Optional[str] # car

    def json_data(self):
        return asdict(self)

    @classmethod
    def from_json(cls, json_node: Dict):
        return cls(**json_node)


@dataclass
class BasicAgent:
    id: Any
    object_info: ObjectInfo
    route: List[WaypointUnit]
    mutable: bool
    trigger_time: float
    trigger_waypoint: Optional[WaypointUnit]

    def __init__(
        self,
        id: Any,
        route: List[WaypointUnit],
        object_info: ObjectInfo,
        mutable: bool,
        trigger_time: float,
        trigger_waypoint: Optional[WaypointUnit] = None
    ):
        self.id = id
        self.route = route
        self.object_info = object_info
        self.mutable = mutable
        self.trigger_time = trigger_time
        self.trigger_waypoint = trigger_waypoint

    def get_initial_waypoint(self) -> WaypointUnit:
        return self.route[0]

    def get_destination_waypoint(self) -> WaypointUnit:
        return self.route[-1]

    def json_data(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_json(cls, json_node: Dict) -> 'AgentClass':
        object_info_js = json_node['object_info']
        json_node['object_info'] = ObjectInfo.from_json(object_info_js)
        route_js = json_node['route']
        route_lst = list()
        for route_js_node in route_js:
            route_lst.append(WaypointUnit.from_json(route_js_node))
        json_node['route'] = route_lst
        if 'trigger_waypoint' in json_node and json_node['trigger_waypoint']:
            json_node['trigger_waypoint'] = WaypointUnit.from_json(json_node['trigger_waypoint'])
        else:
            json_node['trigger_waypoint'] = None
        return cls(**json_node)

AgentClass = TypeVar("AgentClass", bound=BasicAgent)

@dataclass
class BasicSection:

    agents: List[AgentClass]

    __ids: List[Any]
    __fixed_ids: List[Any]
    __mutant_ids: List[Any]

    id_base: Any

    def __init__(self, agents: List[AgentClass]):
        self.agents = agents
        self.id_base = 0
        self.__ids = list()
        self.__fixed_ids = list()
        self.__mutant_ids = list()

        for item in agents:
            self.__ids.append(item.id)
            if item.mutable:
                self.__mutant_ids.append(item.id)
            else:
                self.__fixed_ids.append(item.id)

    @property
    def mutant_ids(self):
        return self.__mutant_ids

    @property
    def ids(self):
        return self.__ids

    def get_agent(self, idx) -> Optional[AgentClass]:
        for agent_index, _agent in enumerate(self.agents):
            if _agent.id == idx:
                return _agent
        return None

    def get_new_id(self) -> int:
        new_id = random.randint(0, 10000)
        while self.id_base + new_id in self.ids:
            new_id += 1
            new_id = new_id % 10000
        return self.id_base + new_id

    def add_agent(self, agent: AgentClass):
        assert agent.id not in self.__ids

        self.agents.append(agent)
        self.__ids.append(agent.id)
        if agent.mutable:
            self.__mutant_ids.append(agent.id)
        else:
            self.__fixed_ids.append(agent.id)

    def remove_agent(self, idx: Any) -> bool:

        if idx not in self.__ids:
            return False

        target_agent = None
        for item in self.agents:
            if item.id == idx:
                target_agent = item
                break

        if target_agent is None or (not target_agent.mutable):
            return False

        self.agents.remove(target_agent)
        self.__ids.remove(idx)
        self.__mutant_ids.remove(idx)

        return True

    def update_agent(self, idx: Any, agent: AgentClass) -> bool:

        if idx not in self.__ids:
            return False

        for agent_index, _agent in enumerate(self.agents):
            if _agent.id == idx:
                self.agents[agent_index] = agent
                break

    def json_data(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_json(cls, json_node: Dict) -> 'SectionClass':
        raise NotImplemented("Not implemented from_json method")

SectionClass = TypeVar("SectionClass", bound=BasicSection)