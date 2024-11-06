from dataclasses import dataclass, asdict
from typing import Optional, Dict

# from pylot.perception.detection.traffic_light import TrafficLight

TRAFFIC_LIGHT_SPACE = ['none', 'S7left', 'S7right', 'S7opposite', 'S8left', 'S9right']

@dataclass
class TrafficLight:

    pattern: Optional[str] # None means always green
    yellow_time: float
    red_time: float

    def __init__(self, pattern: Optional[str] = None, yellow_time = 2.0, red_time = 1.0):
        self.pattern = pattern
        self.yellow_time = yellow_time
        self.red_time = red_time

    def json_data(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_json(cls, json_node: Dict) -> 'TrafficLight':
        return cls(**json_node)