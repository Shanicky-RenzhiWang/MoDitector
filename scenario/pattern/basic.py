import py_trees

from typing import Any, List

from scenario.utils.timer import TimeOut
from scenario.utils.carla_data_provider import CarlaDataProvider

class BasicScenario(object):

    def __init__(
            self,
            name,
            config: Any,
            timeout: float = 60.0,
            debug_mode: bool = False,
            terminate_on_failure: bool = False,
            criteria_enable: bool = False,
            fitness_enable: bool = False,
    ):
        self.name = name
        self.config = config
        self.timeout = timeout
        self.debug_mode = debug_mode
        self.terminate_on_failure = terminate_on_failure
        self.criteria_enable = criteria_enable
        self.fitness_enable = fitness_enable

        # inner parameters
        self.other_actors = []
        self.criteria_list = []  # List of evaluation criteria
        self.fitness_list = []
        self.scenario = None # scenario object

        self.world = CarlaDataProvider.get_world()

        ##### Setup Environment #####
        self._initialize_environment()

        ##### Setup Actors #####
        self._initialize_actors()
        if CarlaDataProvider.is_sync_mode():
            self.world.tick()
        else:
            self.world.wait_for_tick()

        ##### Setup debug mode #####
        if self.debug_mode:
            py_trees.logging.level = py_trees.logging.Level.DEBUG

        ##### Setup Scenario behavior #####
        behavior = self._create_behavior()

        ##### Setup Scenario criteria #####
        criteria = None
        if criteria_enable:
            criteria = self._create_test_criteria()

        ##### Setup Scenario fitness #####
        fitness = None
        if fitness_enable:
            fitness = self._create_test_fitness()

        # Add a trigger condition for the behavior to ensure the behavior is only activated, when it is relevant
        behavior_seq = py_trees.composites.Sequence()
        trigger_behavior = self._setup_scenario_trigger()
        if trigger_behavior is not None:
            behavior_seq.add_child(trigger_behavior)

        if behavior is not None:
            behavior_seq.add_child(behavior)
            behavior_seq.name = behavior.name

        end_behavior = self._setup_scenario_end()
        if end_behavior is not None:
            behavior_seq.add_child(end_behavior)

        self.scenario = Scenario(
            self.name,
            behavior_seq,
            criteria,
            fitness,
            self.timeout
        )

    def _initialize_environment(self):
        pass

    def _initialize_actors(self):
        pass

    def _setup_scenario_trigger(self):
        return None

    def _setup_scenario_end(self):
        return None

    def _create_behavior(self):
        """
        Pure virtual function to setup user-defined scenario behavior
        """
        raise NotImplementedError(
            "This function is re-implemented by all scenarios"
            "If this error becomes visible the class hierarchy is somehow broken")

    def _create_test_criteria(self) -> List:
        """
        Pure virtual function to setup user-defined evaluation criteria for the
        scenario
        """
        return []
    def _create_test_fitness(self) -> List:
        return []

    def remove_all_actors(self):
        """
        Remove all actors
        """
        for i, _ in enumerate(self.other_actors):
            if self.other_actors[i] is not None:
                if CarlaDataProvider.actor_id_exists(self.other_actors[i].id):
                    CarlaDataProvider.remove_actor_by_id(self.other_actors[i].id)
                self.other_actors[i] = None
        self.other_actors = []

class Scenario(object):

    """
    Basic scenario class. This class holds the behavior_tree describing the
    scenario and the test criteria.

    The user must not modify this class.

    Important parameters:
    - behavior: User defined scenario with py_tree
    - criteria_list: List of user defined test criteria with py_tree
    - timeout (default = 60s): Timeout of the scenario in seconds
    - terminate_on_failure: Terminate scenario on first failure
    """

    def __init__(
            self,
            name: str,
            behavior,
            criteria: List,
            fitness: List,
            timeout: float = 60
    ):
        self.name = name

        self.behavior = behavior # py_tree
        self.test_criteria = criteria # list
        self.fitness = fitness # list

        self.timeout = timeout

        # Create criteria py_tree
        if len(self.test_criteria) > 0:
            self.criteria_tree = py_trees.composites.Parallel(
                name="Test Criteria",
                policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
            )
            self.criteria_tree.add_children(self.test_criteria)
            self.criteria_tree.setup(timeout=1)
        else:
            self.criteria_tree = None

        # Create fitness py_tree
        if len(self.fitness) > 0:
            self.fitness_tree = py_trees.composites.Parallel(
                name="Fitness",
                policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL
            )
            self.fitness_tree.add_children(self.fitness)
            self.fitness_tree.setup(timeout=1)
        else:
            self.fitness_tree = None

        # Create node for timeout
        self.timeout_node = TimeOut(self.timeout, name="TimeOut")

        # Create overall py_tree
        self.scenario_tree = py_trees.composites.Parallel(name, policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        # add behavior
        if behavior is not None:
            self.scenario_tree.add_child(self.behavior)
        self.scenario_tree.add_child(self.timeout_node)

        # add criteria
        if self.criteria_tree is not None:
            self.scenario_tree.add_child(self.criteria_tree)

        # add fitness
        if self.fitness_tree is not None:
            self.scenario_tree.add_child(self.fitness_tree)

        self.scenario_tree.setup(timeout=1)

    def _extract_nodes_from_tree(self, tree):  # pylint: disable=no-self-use
        """
        Returns the list of all nodes from the given tree
        """
        if tree is None:
            return []

        node_list = [tree]
        more_nodes_exist = True
        while more_nodes_exist:
            more_nodes_exist = False
            for node in node_list:
                if node.children:
                    node_list.remove(node)
                    more_nodes_exist = True
                    for child in node.children:
                        node_list.append(child)

        if len(node_list) == 1 and isinstance(node_list[0], py_trees.composites.Parallel):
            return []

        return node_list

    def get_criteria(self):
        """
        Return the list of test criteria (all leave nodes)
        """
        criteria_list = self._extract_nodes_from_tree(self.criteria_tree)
        return criteria_list

    def get_fitness(self):
        fitness_list = self._extract_nodes_from_tree(self.fitness_tree)
        return fitness_list

    def terminate(self):
        """
        This function sets the status of all leaves in the scenario tree to INVALID
        """
        # Get list of all nodes in the tree
        node_list = self._extract_nodes_from_tree(self.scenario_tree)

        # Set status to INVALID
        for node in node_list:
            node.terminate(py_trees.common.Status.INVALID)
