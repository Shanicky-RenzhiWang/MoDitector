import carla
import py_trees
import traceback

from loguru import logger
from typing import List, Tuple

# oracle
from scenario.atomic.oracle import (CollisionTest,
                                    ReachDestinationTest,
                                    BlockingTest)
from scenario.atomic.fitness import (CollisionFitness,
                                     ReachDestinationFitness,
                                     BlockingFitness)
from scenario.atomic.behavior import Idle

from scenario.utils.py_trees_port import oneshot_behavior
from scenario.configuration import SeedConfig, BasicAgent, WaypointUnit
from scenario.pattern.basic import BasicScenario, CarlaDataProvider
from .static_scenario import StaticScenario
from .vehicle_scenario import VehicleScenario
from .walker_scenario import WalkerScenario

class WaypointScenario(BasicScenario):

    """
    Implementation of a RouteScenario, i.e. a scenario that consists of driving along a pre-defined route,
    along which several smaller scenarios are triggered
    """

    name = "WaypointScenario"

    def __init__(
            self,
            config: SeedConfig,
            timeout: float = 600,
            terminate_on_failure: bool = True,
            debug_mode: bool = False,
            criteria_enable: bool = True,
            fitness_enable: bool = True
    ):
        self.config = config
        self.timeout = timeout
        self.terminate_on_failure = terminate_on_failure
        self.debug_mode = debug_mode
        self.criteria_enable = criteria_enable
        self.fitness_enable = fitness_enable

        # inner parameters
        self.world = CarlaDataProvider.get_world()

        # setup ego vehicle
        self.ego_vehicles, self.ego_destination = self._setup_ego_vehicle()

        self.list_scenarios = self._build_scenario_instances()

        super(WaypointScenario, self).__init__(
            self.name,
            self.config,
            self.timeout,
            self.debug_mode,
            self.terminate_on_failure,
            self.criteria_enable,
            self.fitness_enable
        )

    def _setup_ego_vehicle(self) -> Tuple[List, WaypointUnit]:
        """
        Set/Update the start position of the ego_vehicle
        """
        # move ego to correct position
        ego_section = self.config.scenario.ego_section
        ego_agent_config: BasicAgent = ego_section.agents[0]
        ego_start_waypoint = ego_agent_config.get_initial_waypoint()
        _map = self.world.get_map()

        elevate_waypoint = _map.get_waypoint(
            carla.Location(
                x=ego_start_waypoint.x,
                y=ego_start_waypoint.y,
                z=ego_start_waypoint.z
            )
        )
        elevate_transform = elevate_waypoint.transform
        elevate_transform.location.z += 0.5

        ego_agent_config.object_info.model = 'vehicle.lincoln.mkz2017'
        ego_agent_config.object_info.rolename = 'hero'
        ego_agent_config.route[0].x = elevate_transform.location.x
        ego_agent_config.route[0].y = elevate_transform.location.y
        ego_agent_config.route[0].z = elevate_transform.location.z + 0.5
        ego_agent_config.route[0].pitch = elevate_transform.rotation.pitch
        ego_agent_config.route[0].yaw = elevate_transform.rotation.yaw
        ego_agent_config.route[0].roll = elevate_transform.rotation.roll

        ego_vehicle = CarlaDataProvider.request_new_actor_by_config(ego_agent_config)
        spectator = CarlaDataProvider.get_world().get_spectator()
        ego_trans = ego_vehicle.get_transform()
        spectator.set_transform(
            carla.Transform(
                ego_trans.location + carla.Location(z=50),
                carla.Rotation(pitch=-90)
            )
        )

        ego_destination = ego_agent_config.get_destination_waypoint()

        return [ego_vehicle], ego_destination

    def _build_scenario_instances(self) -> List:
        scenario_instances = []
        # create vehicle scenario
        try:
            vehicle_scenario_instance = VehicleScenario(
                self.config,
                self.timeout,
                self.debug_mode,
                self.terminate_on_failure,
                self.criteria_enable,
                self.fitness_enable
            )
            if CarlaDataProvider.is_sync_mode():
                self.world.tick()
            else:
                self.world.wait_for_tick()
        except Exception as e:
            logger.warning("Skipping scenario '{}' due to setup error: {}".format('Vehicle', e))
            traceback.print_exc()
            vehicle_scenario_instance = None
        if vehicle_scenario_instance:
            scenario_instances.append(vehicle_scenario_instance)

        # create walker scenario
        try:
            walker_scenario_instance = WalkerScenario(
                self.config,
                self.timeout,
                self.debug_mode,
                self.terminate_on_failure,
                self.criteria_enable,
                self.fitness_enable
            )
            if CarlaDataProvider.is_sync_mode():
                self.world.tick()
            else:
                self.world.wait_for_tick()
        except Exception as e:
            logger.warning("Skipping scenario '{}' due to setup error: {}".format('Walker', e))
            traceback.print_exc()
            walker_scenario_instance = None
        if walker_scenario_instance:
            scenario_instances.append(walker_scenario_instance)

        # create static scenario
        try:
            static_scenario_instance = StaticScenario(
                self.config,
                self.timeout,
                self.debug_mode,
                self.terminate_on_failure,
                self.criteria_enable,
                self.fitness_enable
            )
            if CarlaDataProvider.is_sync_mode():
                self.world.tick()
            else:
                self.world.wait_for_tick()
        except Exception as e:
            logger.warning("Skipping scenario '{}' due to setup error: {}".format('Static', e))
            traceback.print_exc()
            static_scenario_instance = None
        if static_scenario_instance:
            scenario_instances.append(static_scenario_instance)

        # set traffic lights to green
        # set traffic light
        list_actor = self.world.get_actors()
        selected_actors = list_actor.filter('*' + 'traffic_light' + '*')
        for actor_ in selected_actors:
            if isinstance(actor_, carla.TrafficLight):
                # for any light, first set the light state, then set time. for yellow it is
                # carla.TrafficLightState.Yellow and Red it is carla.TrafficLightState.Red
                actor_.set_state(carla.TrafficLightState.Green)
                actor_.set_green_time(100000.0)

        return scenario_instances

    def _initialize_environment(self):
        world = CarlaDataProvider.get_world()
        world.set_weather(self.config.scenario.weather_section.carla_parameters)

    def _initialize_actors(self):
        self.other_actors = list()
        # Add all the actors of the specific scenarios to self.other_actors
        for scenario in self.list_scenarios:
            self.other_actors.extend(scenario.other_actors)

    def _create_behavior(self):
        """
        Basic behavior do nothing, i.e. Idle
        """
        behavior = py_trees.composites.Parallel(
            name="WaypointScenarioBehavior",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL
        )

        scenario_behaviors = []
        # todo: consider add trigger position?
        for i, scenario in enumerate(self.list_scenarios):
            if scenario.scenario.behavior is not None:
                # only execute once
                name = "{} - {}".format(i, scenario.scenario.behavior.name)
                oneshot_idiom = oneshot_behavior(
                    name=name,
                    variable_name=name,
                    behaviour=scenario.scenario.behavior)
                scenario_behaviors.append(oneshot_idiom)

        behavior.add_children(scenario_behaviors)
        behavior.add_child(Idle())  # The behaviours cannot make the route scenario stop
        return behavior

    def _create_test_criteria(self):
        """
        1. collision
        2. stuck
        3. reach destination
        """
        criteria = []
        collision_criterion = CollisionTest(
            self.ego_vehicles[0],
            terminate_on_failure=self.terminate_on_failure
        ) # collision time -> prev 10s or not
        completion_criterion = ReachDestinationTest(
            self.ego_vehicles[0],
            self.ego_destination
        ) # ignore
        blocked_criterion = BlockingTest(
            self.ego_vehicles[0],
            speed_threshold=0.5,
            below_threshold_max_time=90.0,
            terminate_on_failure=self.terminate_on_failure,
            name="AgentBlockedTest"
        ) # violation time, prev 90s are violation
        criteria.append(completion_criterion)
        criteria.append(collision_criterion)
        criteria.append(blocked_criterion)
        return criteria

    def _create_test_fitness(self):
        fitness = []

        collision_fitness = CollisionFitness(self.ego_vehicles[0])  # collision time -> prev 10s or not
        reach_destination_fitness = ReachDestinationFitness(self.ego_vehicles[0], self.ego_destination)
        blocking_fitness = BlockingFitness(
            self.ego_vehicles[0],
            speed_threshold=0.5,
            below_threshold_max_time=90.0,
        )
        fitness.append(collision_fitness)
        fitness.append(reach_destination_fitness)
        fitness.append(blocking_fitness)
        return fitness

    def remove_all_actors(self):
        """
        Remove all actors upon deletion
        """
        logger.debug(f'Delete the scenario instance {self.__class__.__name__}')
        super(WaypointScenario, self).remove_all_actors()
        logger.debug(f'Delete the scenario instance {self.__class__.__name__}')
        # print(traceback.format_stack())
        # TODO: check ego vehicle remove-
        for i, _ in enumerate(self.ego_vehicles):
            if self.ego_vehicles[i] is not None:
                if CarlaDataProvider.actor_id_exists(self.ego_vehicles[i].id):
                    CarlaDataProvider.remove_actor_by_id(self.ego_vehicles[i].id)
                self.ego_vehicles[i] = None
        self.ego_vehicles = []
