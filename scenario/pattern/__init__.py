from .basic import BasicScenario, Scenario

from typing import TypeVar

ScenarioTreeType = TypeVar('ScenarioTreeType', bound=Scenario)

from .waypoint_scenario import WaypointScenario