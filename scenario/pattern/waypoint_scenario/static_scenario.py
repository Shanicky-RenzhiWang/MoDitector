import carla
import py_trees

from scenario.atomic.behavior import ActorTransformSetter
from scenario.pattern.basic import BasicScenario, CarlaDataProvider
from scenario.configuration import SeedConfig, BasicAgent

class StaticScenario(BasicScenario):

    name = 'StaticScenario'

    def __init__(
            self,
            config: SeedConfig,
            timeout=60,
            debug_mode=False,
            terminate_on_failure=True,
            criteria_enable: bool = False,
            fitness_enable: bool = False
    ):

        # inner parameters
        self.other_actors_config = list()

        super(StaticScenario, self).__init__(
            self.name,
            config,
            timeout=timeout,
            debug_mode=debug_mode,
            terminate_on_failure=terminate_on_failure,
            criteria_enable=criteria_enable,
            fitness_enable=fitness_enable
        )

    def _initialize_actors(self):
        """
        Custom initialization
        """
        self.other_actors = list()
        self.other_actors_config = list()

        scenario_config = self.config.scenario
        static_section = scenario_config.static_section

        # initialize vehicles
        if static_section:
            static_agents = static_section.agents
            for i, agent_config in enumerate(static_agents):
                new_actor = CarlaDataProvider.request_new_actor_by_config(agent_config)
                if new_actor is None:
                    continue
                new_actor.set_simulate_physics(enabled=True)
                self.other_actors.append(new_actor)
                self.other_actors_config.append(agent_config)

    def _create_single_actor_behavior(self, actor: carla.Actor, actor_config: BasicAgent):
        actor_start_wp = actor_config.route[0]
        actor_transform = carla.Transform(
            location=carla.Location(
                x=actor_start_wp.x,
                y=actor_start_wp.y,
                z=0.5
            ),
            rotation=carla.Rotation(
                pitch=actor_start_wp.pitch,
                yaw=actor_start_wp.yaw,
                roll=actor_start_wp.roll
            )
        )
        start_transform = ActorTransformSetter(actor, actor_transform)
        return start_transform

    def _create_behavior(self):
        static_section_tree = py_trees.composites.Parallel(
            name='static_section_behavior',
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL
        )

        for i, actor_config in enumerate(self.other_actors_config):
            actor = self.other_actors[i]
            if actor and actor.is_alive:
                actor_behavior_tree = self._create_single_actor_behavior(actor, actor_config)
                static_section_tree.add_child(actor_behavior_tree)
        return static_section_tree

