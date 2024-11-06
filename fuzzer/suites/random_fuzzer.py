import os
import gc
import torch
import json
import random

from loguru import logger
from datetime import datetime
from omegaconf import DictConfig

from fuzzer.base import BaseFuzzer
from fuzzer.mutator import ScenarioMutator

from scenario.configuration import ScenarioConfig
from ..root_cause_util.root_cause_ana import StatHandler

class RandomFuzzer(BaseFuzzer):

    def __init__(self, save_root: str, cfg: DictConfig, server_config: DictConfig):

        super(RandomFuzzer, self).__init__(save_root, cfg, server_config)

        self.seed_path = self.cfg.seed_path
        self.time_limit = self.cfg.time_limit

        # load seed scenario
        with open(self.seed_path, 'r') as f:
            seed_json = json.load(f)
        seed_scenario = ScenarioConfig.from_json(seed_json)
        self.seed = self.get_seed_config('0', seed_scenario)
        self.mutator = ScenarioMutator(
            self.client,
            self.cfg.mutator_vehicle_num,
            self.cfg.mutator_walker_num,
            self.cfg.mutator_static_num,
            0.5,
            2.0
        )

        # inner parameters
        self.iteration = self.result_recorder.get_last_iteration() # note should add 1

    def run(self):
        start_time = datetime.now()
        mutated_scenario = None
        self.runner_msg_folder = os.path.join(self.save_root, 'runner_msg')
        os.makedirs(self.runner_msg_folder, exist_ok=True)
        while True:
            self.iteration += 1
            # if self.iteration % 10 == 0:
            #     self.restart_carla()
            #     self.mutator = ScenarioMutator(
            #         self.client,
            #         self.cfg.mutator_vehicle_num,
            #         self.cfg.mutator_walker_num,
            #         self.cfg.mutator_static_num,
            #         0.5,
            #         2.0
            #     )

            if self.termination(start_time, self.time_limit):
                break

            logger.info(f'==================== Run Iter {self.iteration} ====================')
            # # generate new task
            scenario_id = f"{self.scenario_id_pref}{self.iteration}"

            if random.random() > 0.6 or mutated_scenario is None:
                mutated_scenario = self.mutator.generate_new_scenario(self.seed.scenario, scenario_id)
            else:
                _, mutated_scenario = self.mutator.mutate_current_scenario(mutated_scenario, scenario_id)
            mutated_seed = self.get_seed_config(scenario_id, mutated_scenario)
            runner_pass, runner_message = self.load_and_run_scenario(mutated_seed)
            if not runner_pass:
                logger.warning('Scenario Runner has bug: {}', runner_message)
                break
            with open(os.path.join(self.runner_msg_folder, f'{mutated_seed.id}.json'), 'w') as f:
                json.dump(runner_message, f, indent=4)

            mutated_seed = self.update_seed_config(mutated_seed, runner_message['result'])
            # save scenario
            seed_file = os.path.join(self.seed_folder, f"{mutated_seed.id}.json")
            with open(seed_file, 'w') as f:
                seed_json_data = mutated_seed.json_data()
                json.dump(seed_json_data, f, indent=4)
            root_cause, feedback_score = self._get_root_cause_stat(mutated_seed.id, runner_message=runner_message)

            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                        print(type(obj), obj.size())
                except:
                    pass

    def _get_root_cause_stat(self, seed_id, runner_message):
        base_dir = os.path.join(self.save_root, 'agent_data', seed_id)
        collision = runner_message['result']['infractions']['collision_timestamp']
        stat_handler = StatHandler(base_dir, collision)
        stat_save_path = os.path.join(self.runner_msg_folder, f'{seed_id}_root_cause.json')
        stat, feedback_score = stat_handler.collect_root_cause(prefer_feedback='perception', stat_save_path=stat_save_path)
        
        # with open(os.path.join(self.runner_msg_folder, f'{seed_id}_root_cause.json'), 'w') as f:
        #     json.dump(stat, f, indent=4)
        return stat, feedback_score