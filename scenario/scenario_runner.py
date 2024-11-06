from __future__ import print_function

import signal
import sys
import time

import carla
import py_trees

from typing import Optional

from scenario.utils.timer import GameTime
from scenario.utils.watchdog import Watchdog
from scenario.utils.carla_data_provider import CarlaDataProvider
from scenario.utils.result_writer import ResultOutputProvider
from ads import AgentWrapper

class ScenarioRunner(object):

    def __init__(self, debug_mode=False):
        """
        Setups up the parameters, which will be filled at load_scenario()
        """
        self._debug_mode = debug_mode

        self.scenario = None # py_tree scenario instance
        self.scenario_tree = None # py_tree for the all scenario
        self.ego_vehicles = None
        self.other_actors = None
        self._agent = None

        # add watchdogs
        self._timeout_tick = 120.0
        self._watchdog_tick = Watchdog(self._timeout_tick) # Used to detect if the simulation is down # watchdog timeout (need to be added)

        self._running = False
        self._timestamp_last_run = 0.0
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.start_game_time = None
        self.end_system_time = None
        self.end_game_time = None

        self.result_filename = None

        # Register the scenario tick as callback for the CARLA world
        # Use the callback_id inside the signal handler to allow external interrupts
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        self._running = False

    def cleanup(self):
        """
        Reset all parameters
        """
        self.scenario = None  # py_tree scenario instance
        self.scenario_tree = None  # py_tree for the all scenario
        self.ego_vehicles = None
        self.other_actors = None
        self._agent = None

        self._running = False
        self._timestamp_last_run = 0.0
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.start_game_time = None
        self.end_system_time = None
        self.end_game_time = None

        self.result_filename = None

    def load_scenario(self, scenario, agent, result_filename):
        """
        scenario: scenario instance from BasicScenario
        """
        self.cleanup()

        GameTime.restart()
        self.result_filename = result_filename
        self._agent = AgentWrapper(agent) # wrapped agent
        self.scenario = scenario.scenario
        self.scenario_tree = self.scenario.scenario_tree
        self.ego_vehicles = scenario.ego_vehicles
        self.other_actors = scenario.other_actors

        # To print the scenario tree uncomment the next line
        # py_trees.display.render_dot_tree(self.scenario_tree)
        CarlaDataProvider.set_ego(self.ego_vehicles[0])
        self._agent.setup_sensors(self.ego_vehicles[0])

    def run_scenario(self, scenario, agent, result_filename: Optional[str] = None):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        self.load_scenario(scenario, agent, result_filename)

        self.start_system_time = time.time()
        self.start_game_time = GameTime.get_time()

        self._watchdog_tick.start()
        self._running = True
        while self._running:
            timestamp = None
            world = CarlaDataProvider.get_world()
            CarlaDataProvider.set_time_step((GameTime.get_time()-self.start_game_time)*20) #20Hz
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            if timestamp:
                self._tick_scenario(timestamp)

    def _tick_scenario(self, timestamp):
        """
        Run next tick of scenario and the agent and tick the world.
        """

        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds

            self._watchdog_tick.update()
            # Update game time and actor information
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()

            try:
                ego_action = self._agent()
            except Exception as e:
                raise RuntimeError(e)

            self.ego_vehicles[0].apply_control(ego_action)

            # Tick scenario
            self.scenario_tree.tick_once()

            if self._debug_mode:
                print("\n")
                py_trees.display.print_ascii_tree(
                    self.scenario_tree, show_status=True)
                sys.stdout.flush()

            if self.scenario_tree.status != py_trees.common.Status.RUNNING:
                self._running = False

            spectator = CarlaDataProvider.get_world().get_spectator()
            ego_trans = self.ego_vehicles[0].get_transform()
            spectator.set_transform(carla.Transform(ego_trans.location + carla.Location(z=50),
                                                        carla.Rotation(pitch=-90)))

        if self._running and self.get_running_status():
            CarlaDataProvider.get_world().tick(self._timeout_tick)

    def get_running_status(self):
        """
        returns:
           bool: False if watchdog exception occured, True otherwise
        """
        return self._watchdog_tick.get_status()

    def stop_scenario(self):
        """
        This function triggers a proper termination of a scenario
        """
        self._watchdog_tick.stop()

        self.end_system_time = time.time()
        self.end_game_time = GameTime.get_time()

        self.scenario_duration_system = self.end_system_time - self.start_system_time
        self.scenario_duration_game = self.end_game_time - self.start_game_time

        if self.get_running_status():
            if self.scenario is not None:
                self.scenario.terminate()

            if self._agent is not None:
                self._agent.cleanup()
                self._agent = None

            self.analyze_scenario()

    def analyze_scenario(self):
        """
        Analyzes and prints the results of the route
        TODO: Modify this
        """
        global_result = '\033[92m'+'SUCCESS'+'\033[0m'

        for criterion in self.scenario.get_criteria():
            if criterion.test_status != "SUCCESS":
                global_result = '\033[91m'+'FAILURE'+'\033[0m'

        if self.scenario.timeout_node.timeout:
            global_result = '\033[91m'+'FAILURE'+'\033[0m'

        rp = ResultOutputProvider(self, global_result, stdout=True, filename=self.result_filename)
        rp.write()