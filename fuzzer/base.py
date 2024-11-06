from fuzzer.utils.result_recorder import ResultRecorder
from fuzzer.utils.carla_operator import CarlaOperator
from scenario.configuration import SeedConfig, ScenarioConfig
from scenario.utils.carla_data_provider import CarlaDataProvider
from scenario.utils.watchdog import Watchdog
from scenario.scenario_runner import ScenarioRunner
from typing import Tuple, Dict
from importlib import import_module
from omegaconf import DictConfig
from datetime import datetime
from loguru import logger
import pkg_resources
import numpy as np
import traceback
import signal
import carla
import os


class BaseFuzzer(object):

    scenario_id_pref = 'seed_'
    frame_rate = 20.0

    def __init__(self, save_root: str, cfg: DictConfig, server_config: DictConfig):
        dist = pkg_resources.get_distribution("carla")
        logger.info('Carla version: {}'.format(dist.version))

        self.cfg = cfg
        self.random_seed = self.cfg.random_seed
        self.resume = self.cfg.resume
        self.debug = self.cfg.debug

        # create save root path
        self.save_root = save_root
        if not os.path.exists(self.save_root):
            os.makedirs(self.save_root)
        self.result_folder = os.path.join(self.save_root, 'results')
        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)
        # save folders
        self.seed_folder = os.path.join(self.save_root, 'seeds')
        if not os.path.exists(self.seed_folder):
            os.makedirs(self.seed_folder)

        # create carla operator & start carla
        self.client = None
        self.carla_operator = CarlaOperator(server_config)
        self.server_config = server_config

        self.restart_carla()

        # connect to client & traffic manager
        # self.client = carla.Client(self.carla_operator.host, int(self.carla_operator.port))
        # self.client.set_timeout(60.0)
        # try:
        #     self.traffic_manager = self.client.get_trafficmanager(int(self.carla_operator.tm_port))
        # except Exception as e:
        #     logger.error("traffic_manager fail to init", flush=True)
        #     logger.error("> {}\033[0m\n".format(e))
        #     sys.exit(-1)

        # create result recorder
        # result recorder
        result_json = os.path.join(self.save_root, 'result_record.json')
        self.result_recorder = ResultRecorder(result_json, self.resume)
        # carla record
        self.carla_record_folder = None if not self.cfg.record else os.path.join(self.save_root, 'carla_record')
        if self.carla_record_folder is not None:
            if os.path.exists(self.carla_record_folder):
                os.makedirs(self.carla_record_folder)

        # load agent entry_point
        self.module_agent = self.load_entry_point(self.cfg.entry_point_target_agent)
        # load scenario entry_point
        self.module_scenario = self.load_entry_point(self.cfg.entry_point_scenario)

        # Create the ScenarioRunner
        self.runner = ScenarioRunner(self.debug)

        # Create the agent watchdog
        self._agent_watchdog = Watchdog(30.0)
        signal.signal(signal.SIGINT, self._signal_handler)

        # inner parameters
        self.agent_instance = None
        self._vehicle_lights = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam
        self.times = {
            'seed_select':[],
            'mutate': [],
            'feedback': [],
            'simulation': []
        }

    def restart_carla(self):
        os.system(f"kill -9 $(ps -ef|grep {self.server_config['port']}|gawk '$0 !~/grep/ {{print $2}}' |tr -s '\n' ' ')")
        self.client = self.carla_operator.start()
        self.client.set_timeout(60.0)

    def _signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        if self._agent_watchdog and not self._agent_watchdog.get_status():
            raise RuntimeError("Timeout: Agent took too long to setup")
        elif self.runner:
            self.runner.signal_handler(signum, frame)
        exit(-1)

    def destroy(self):
        """
        Cleanup and delete actors, ScenarioManager and CARLA world
        """
        self._cleanup()
        if hasattr(self, 'runner') and self.runner:
            del self.runner
        if hasattr(self, 'world') and self.world:
            del self.world
        self.carla_operator.close()

    def _cleanup(self):
        """
        Remove and destroy all actors
        """

        # Simulation still running and in synchronous mode?
        if self.runner and self.runner.get_running_status() and hasattr(self, 'world') and self.world:
            # Reset to asynchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
            # self.traffic_manager.set_synchronous_mode(False)

        if self.runner:
            self.runner.cleanup()

        CarlaDataProvider.cleanup()

        if self._agent_watchdog:
            self._agent_watchdog.stop()

        if hasattr(self, 'agent_instance') and self.agent_instance:
            self.agent_instance.destroy()
            self.agent_instance = None

        if hasattr(self, 'statistics_manager') and self.statistics_manager:
            self.statistics_manager.scenario = None

    def _load_and_wait_for_world(self, town):
        """
        Load a new CARLA world and provide data to CarlaDataProvider
        """
        # self.traffic_manager.set_synchronous_mode(False)
        try:
            self.world = self.client.load_world(town)
        except Exception as e:
            logger.error(traceback.print_exc())
            logger.error("> {}\033[0m\n".format(e))

        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1.0 / self.frame_rate
        settings.synchronous_mode = True
        self.world.apply_settings(settings)

        self.world.reset_all_traffic_lights()

        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)
        CarlaDataProvider.set_carla_host('localhost')
        CarlaDataProvider.set_carla_port(self.server_config['port'])
        # CarlaDataProvider.set_traffic_manager_port(int(self.carla_operator.tm_port))
        #
        # self.traffic_manager.set_synchronous_mode(True)
        # self.traffic_manager.set_random_device_seed(int(self.random_seed))

        # Wait for the world to be ready
        if CarlaDataProvider.is_sync_mode():
            self.world.tick()
        else:
            self.world.wait_for_tick()

        if CarlaDataProvider.get_map().name != town:
            raise Exception("The CARLA server uses the wrong map!"
                            "This scenario requires to use map {}".format(town))

    def _load_target_agent(self, config: SeedConfig):
        self._agent_watchdog.start()
        logger.info('Load agent config from {}', self.cfg.config_target_agent)
        self.agent_instance = self.module_agent(
            self.save_root,
            config.id,
            self.cfg.config_target_agent,
            config.scenario.ego_section.agents[0].route,
            require_saver=True
        )
        # config.agent = self.agent_instance
        logger.info('Loaded agent')
        self._agent_watchdog.stop()

    def load_and_run_scenario(self, config: SeedConfig) -> Tuple[bool, Dict]:
        logger.info("\n\033[1m========= Preparing {} =========".format(config.id))

        town = config.scenario.map_section.town
        logger.info(f"\033[1m> Loading the world {town} \033[0m")
        try:
            self._load_and_wait_for_world(town)
        except Exception as e:
            # The agent setup has failed -> start the next route
            logger.error("\n\033[91mCould not load the world.\033[0m:")
            logger.error("> {}\033[0m\n".format(e))
            traceback.print_exc()
            crash_message = "World couldn't be set up"
            self._cleanup()
            return False, {'message': crash_message}

        logger.info("> Setting up the agent\033[0m")
        try:
            self._load_target_agent(config)
        except Exception as e:
            # The agent setup has failed -> start the next route
            logger.error("\n\033[91mCould not set up the required agent:")
            logger.error("> {}\033[0m\n".format(e))
            traceback.print_exc()
            crash_message = "Agent couldn't be set up"
            self._cleanup()
            return False, {'message': crash_message}

        carla_record_file_name = f"{config.id}.log"
        if self.carla_record_folder:
            carla_record_file_path = self.carla_operator.get_record_path(
                self.carla_record_folder, carla_record_file_name)
        else:
            carla_record_file_path = None
        try:
            scenario = self.module_scenario(
                config=config,
                timeout=600.0,
                terminate_on_failure=True,
                debug_mode=self.debug,
                criteria_enable=True,
                fitness_enable=True
            )
            # Night mode
            if config.scenario.weather_section.sun_altitude_angle < 0.0:
                for vehicle in scenario.ego_vehicles:
                    vehicle.set_light_state(carla.VehicleLightState(self._vehicle_lights))

            # Load scenario and run it
            if carla_record_file_path is not None:
                self.client.start_recorder(carla_record_file_path, True)
        except Exception as e:
            # The scenario is wrong -> set the ejecution to crashed and stop
            logger.error("\n\033[91mThe scenario could not be loaded:")
            logger.error("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Simulation crashed"

            if carla_record_file_path is not None:
                self.client.stop_recorder()
                self.carla_operator.move_carla_record(carla_record_file_path)

            self._cleanup()
            return False, {'message': crash_message}

        try:
            logger.info("\033[1m> Running the route\033[0m")
            self.runner.run_scenario(scenario, self.agent_instance,
                                     os.path.join(self.result_folder, f"{config.id}.txt"))
            logger.info("\033[1m> Stopping the route\033[0m")
            self.runner.stop_scenario()

            runner_result = self.result_recorder.update(
                config,
                scenario.scenario,
                self.runner.scenario_duration_system,
                self.runner.scenario_duration_game
            )

            if carla_record_file_path is not None:
                self.client.stop_recorder()
                self.carla_operator.move_carla_record(carla_record_file_path)

            # Remove all actors
            scenario.remove_all_actors()

            self._cleanup()

        except Exception as e:
            print("\n\033[91mFailed to stop the scenario, the statistics might be empty:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()
            crash_message = "Simulation crashed"
            return False, {'message': crash_message}

        return True, {'message': 'success runner', 'result': runner_result}

    @staticmethod
    def get_oracle(runner_result: Dict) -> Dict:
        oracle = {
            'complete': False,
            'collision': False,
            'blocked': False
        }

        infractions = runner_result['infractions']
        if len(infractions['collisions_layout']) > 0 or len(infractions['collisions_pedestrian']) > 0 or len(infractions['collisions_vehicle']) > 0:
            oracle['collision'] = True
        if len(infractions['vehicle_blocked']) > 0 or len(infractions['route_timeout']) > 0:
            oracle['blocked'] = True

        if oracle['collision'] or oracle['blocked']:
            oracle['complete'] = False
        else:
            oracle['complete'] = True

        return oracle

    def get_fitness(self, runner_result: Dict) -> float:
        """
        minimum is better
        should be overridden by subclasses
        :param runner_result:
        :return:
        """
        fitness_result = runner_result['fitness']
        fitness_value_lst = []
        for item in fitness_result:
            item_name = item[0]
            fitness_value = item[1]

            if item_name == 'CollisionFitness':
                fitness_value = min(fitness_value, 5.0) / 5.0

            if item_name == 'BlockingFitness':
                # 0-1
                fitness_value = fitness_value

            if item_name == 'ReachDestinationFitness':
                fitness_value = 1 - min(fitness_value, 10.0) / 10.0

            fitness_value_lst.append(fitness_value)
        return np.mean(fitness_value_lst)

    @staticmethod
    def load_entry_point(name):
        mod_name, attr_name = name.split(":")
        mod = import_module(mod_name)
        fn = getattr(mod, attr_name)
        return fn

    def termination(self, start_time, run_limit: float) -> bool:
        curr_time = datetime.now()
        t_delta = (curr_time - start_time).total_seconds()
        if t_delta / 3600 > run_limit:
            return True
        return False

    @staticmethod
    def get_seed_config(idx: str, scenario_config: ScenarioConfig) -> SeedConfig:
        return SeedConfig(idx, scenario_config)

    def update_seed_config(self, seed: SeedConfig, runner_result: Dict) -> SeedConfig:
        oracle = self.get_oracle(runner_result)
        fitness = self.get_fitness(runner_result)
        seed.set_oracle(oracle)
        seed.set_fitness(fitness)
        return seed


class TimeoutException(Exception):
    pass
