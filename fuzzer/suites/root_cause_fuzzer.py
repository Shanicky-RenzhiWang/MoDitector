from omegaconf import DictConfig
from py import log
from .. import BaseFuzzer
from ..mutator import ScenarioMutator
from scenario.configuration import ScenarioConfig, SeedConfig
import json
from datetime import datetime
from loguru import logger
import os
import copy
import numpy as np
from ..utils.fileUtil import save_obj, load_obj, get_most_recent_folder
from ..root_cause_util.root_cause_ana import StatHandler
import random
from time import time

class RootCauseFuzzer(BaseFuzzer):
    def __init__(self, save_root: str, cfg: DictConfig, server_config: DictConfig):
        super().__init__(save_root, cfg, server_config)
        self.seed_path = cfg.seed_path
        self.resume = cfg.resume
        self.time_limit = cfg.time_limit
        with open(self.seed_path, 'r') as f:
            seed_json = json.load(f)
        seed_scenario = ScenarioConfig.from_json(seed_json)
        self.seed = self.get_seed_config('0', seed_scenario)
        self.scenario_mutator = ScenarioMutator(
            self.client,
            self.cfg.mutator_vehicle_num,
            self.cfg.mutator_walker_num,
            self.cfg.mutator_static_num,
            0.5,
            2.0
        )
        self.scenario_id_pref = 'seed_'
        self.frame_rate = 20.0
        self.iteration = self.result_recorder.get_last_iteration()
        self.fuzz_threshold = 0.5
        self.initial_seed_pool_size = 4

    def run(self):
        logger.info('===== Start Fuzzer (Root Cause) =====')
        st_time = datetime.now()
        restart = True
        checkpoint_file = os.path.join(self.save_root, 'checkpoint.pkl')
        self.runner_msg_folder = os.path.join(self.save_root, 'runner_msg')
        os.makedirs(self.runner_msg_folder, exist_ok=True)

        if self.resume and os.path.isfile(checkpoint_file):
            logger.info(f'===> Load checkpoint from {checkpoint_file}')
            self.seed_pool = load_obj(checkpoint_file)
            restart = len(self.seed_pool) == 0
        

        if restart:
            self.seed_pool = dict()
            while len(self.seed_pool) < self.initial_seed_pool_size:
                self.iteration += 1
                if self.termination(st_time, self.time_limit):
                    break
                logger.info(f'==================== Run Iter {self.iteration} ====================')
                # # generate new task
                scenario_id = f"{self.scenario_id_pref}{self.iteration}"
                mutated_scenario = self.scenario_mutator.generate_new_scenario(self.seed.scenario, scenario_id)
                scenario_folder = os.path.join(self.save_root, 'scenario')
                os.makedirs(scenario_folder, exist_ok=True)
                scenario_file = os.path.join(scenario_folder, f"{mutated_scenario.id}.json")
                with open(scenario_file, 'w') as f:
                    scenario_json_data = mutated_scenario.json_data()
                    json.dump(scenario_json_data, f, indent=4)
                mutated_seed = self.get_seed_config(scenario_id, mutated_scenario)
                runner_pass, runner_message = self.load_and_run_scenario(mutated_seed)
                # runner_pass = True
                # with open(os.path.join(self.save_root,'message.json'), 'r') as f:
                # runner_message = json.load(f)
                if not runner_pass:
                    logger.warning('Scenario Runner has bug: {}', runner_message)
                    break
                with open(os.path.join(self.runner_msg_folder, f'{mutated_seed.id}.json'), 'w') as f:
                    json.dump(runner_message, f, indent=4)

                # base_output_dir = '/home/erdos/workspace/ADSFuzzer/outputs'
                # most_recent_date_folder = get_most_recent_folder(base_output_dir)
                # most_recent_time_folder = get_most_recent_folder(os.path.join(base_output_dir,most_recent_date_folder))
                # log_file = os.path.join(base_output_dir, most_recent_date_folder, most_recent_time_folder, 'pylot.log')
                # os.system(f'cp {log_file} {os.path.join(self.save_root,"agent_data", mutated_seed.id)}/')
                # #print(log_file)

                mutated_seed = self.update_seed_config(mutated_seed, runner_message['result'])
                seed_file = os.path.join(self.seed_folder, f"{mutated_seed.id}.json")
                with open(seed_file, 'w') as f:
                    seed_json_data = mutated_seed.json_data()
                    json.dump(seed_json_data, f, indent=4)

                logger.info(f'Seed {mutated_seed.id} Result: {mutated_seed.oracle} Fitness: {mutated_seed.fitness}')
                # if not mutated_seed.oracle['complete']:
                #     continue
                root_cause, feedback_score = self._get_root_cause_stat(mutated_seed.id, runner_message=runner_message)
                self.seed_pool[mutated_seed.id] = {
                    'seed': copy.deepcopy(mutated_seed),
                    'fail': 0,
                    'seed_mutation': 0.0,
                    'mutate_bias': feedback_score,
                    'root_cause': root_cause,
                    'runner_message': runner_message
                }
            save_obj(self.seed_pool, checkpoint_file)
        # self.seed_pool = list()
        # print([(v['mutate_bias'], v['root_cause']) for k, v in self.seed_pool.items()])
        while True:
            self.iteration += 1
            logger.info(f'==================== Run Iter {self.iteration} ====================')
            scenario_id = f"{self.scenario_id_pref}{self.iteration}"
            if self.termination(st_time, self.time_limit):
                return
            iter_time = time()
            select_seed_id, mutate_bias = self._seed_selection()
            select_time = time()
            self.times['seed_select'].append(select_time-iter_time)
            curr_fail = self.seed_pool[select_seed_id]['fail']
            bias = self.seed_pool[select_seed_id]['mutate_bias']
            seed_mutation = self.seed_pool[select_seed_id]['seed_mutation']

            do_big_mutate = mutate_bias > self.fuzz_threshold
            # do_big_mutate = True if (random.random() > 0.6 or mutated_scenario is None) else False
            if do_big_mutate:
                mutate_seed_scenario = self.scenario_mutator.generate_new_scenario(self.seed.scenario, scenario_id)
                mutated_seed = self.get_seed_config(scenario_id, mutate_seed_scenario)
                curr_fail = 0
            else:
                mutated_seed = copy.deepcopy(self.seed_pool[select_seed_id]['seed'])
                _, mutated_seed_scenario = self.scenario_mutator.mutate_current_scenario(
                    self.seed_pool[select_seed_id]['seed'].scenario, scenario_id)
                mutated_seed.scenario = mutated_seed_scenario
                mutated_seed.id = scenario_id
                if seed_mutation == 0.0:
                    seed_mutation = 1.0
            mutate_time = time()
            self.times['mutate'].append(mutate_time-select_time)

            runner_pass, runner_message = self.load_and_run_scenario(mutated_seed)
            runner_pass = True
            # with open(os.path.join(self.runner_msg_folder,'seed_2.json'), 'r') as f:
            #     runner_message = json.load(f)

            if not runner_pass:
                logger.warning('Scenario Runner has bug: {}', runner_message)
                break
            with open(os.path.join(self.runner_msg_folder, f'{mutated_seed.id}.json'), 'w') as f:
                json.dump(runner_message, f, indent=4)
            simulation_time = time()
            self.times['simulation'].append(simulation_time-mutate_time)

            root_cause, feedback_score = self._get_root_cause_stat(mutated_seed.id, runner_message=runner_message)
            feedback_time = time()
            self.times['feedback'].append(feedback_time-simulation_time)

            # base_output_dir = '/home/erdos/workspace/ADSFuzzer/outputs'
            # most_recent_date_folder = get_most_recent_folder(base_output_dir)
            # most_recent_time_folder = get_most_recent_folder(os.path.join(base_output_dir,most_recent_date_folder))
            # log_file = os.path.join(base_output_dir, most_recent_date_folder, most_recent_time_folder, 'pylot.log')
            # os.system(f'cp {log_file} {os.path.join(self.save_root, "agent_data", mutated_seed.id)}/')
            # print(log_file)
            times_res ={'ori':self.times,
                        'average':{k:sum(v)/len(v) for k,v in self.times.items()}}
            times_file = os.path.join(self.save_root,'results','times_consume.json')
            with open(times_file,'w') as f:
                json.dump(times_res,f, indent=4)

            mutated_seed = self.update_seed_config(mutated_seed, runner_message['result'])
            seed_file = os.path.join(self.seed_folder, f"{mutated_seed.id}.json")
            with open(seed_file, 'w') as f:
                seed_json_data = mutated_seed.json_data()
                json.dump(seed_json_data, f, indent=4)

            logger.info(f'Seed {mutated_seed.id} Result: {mutated_seed.oracle} Fitness: {mutated_seed.fitness}')

            benign = True
            if mutated_seed.oracle['collision']:
                curr_fail += 1
                benign = False
            # if not do_big_mutate:
            #     if float(source_seed_mutation) == 0:
            #         source_seed_mutation = 1.0
            #     delta_fail = curr_fail / float(seed_mutation)
            #     delta_fail = delta_fail if not benign else 0.1*(delta_fail-1.0)
            #     delta_fitness = self.seed_pool[select_seed_id]['seed'].fitness - mutated_seed.fitness
            #     delta_select = -0.1
            #     mutate_bias += 0.5 * delta_fail + 0.3* np.tanh(delta_fitness)+ delta_select
            # self.seed_pool[select_seed_id]['mutate_bias'] = float(np.clip(mutate_bias, 1e-5, 4.0))
            # self.seed_pool[select_seed_id]['fail'] = curr_fail
            # self.seed_pool[select_seed_id]['seed_mutation'] = seed_mutation

            # if mutated_seed.fitness < self.seed_pool[select_seed_id]['seed'].fitness:
            if feedback_score > self.seed_pool[select_seed_id]['seed'].fitness:
                self.seed_pool[mutated_seed.id] = {
                    'seed': copy.deepcopy(mutated_seed),
                    'fail': 0,
                    'seed_mutation': 0.0,
                    'mutate_bias': feedback_score,
                    'root_cause': root_cause
                }
            save_obj(self.seed_pool, checkpoint_file)

    def _seed_selection(self):
        keys = list(self.seed_pool.keys())
        poss = [self.seed_pool[x]['mutate_bias'] for x in keys]
        min_poss = min(poss)-0.1
        sum_poss = sum([p-min_poss for p in poss])
        if sum_poss == 0:
            select_key = np.random.choice(list(keys))
        else:
            poss = [(p-min_poss)/sum_poss for p in poss]
            select_key = np.random.choice(list(keys), p=poss)
        return select_key, self.seed_pool[select_key]['mutate_bias']

    def _get_root_cause_stat(self, seed_id, runner_message):
        base_dir = os.path.join(self.save_root, 'agent_data', seed_id)
        collision = runner_message['result']['infractions']['collision_timestamp']
        stat_handler = StatHandler(base_dir, collision)
        stat_save_path = os.path.join(self.runner_msg_folder, f'{seed_id}_root_cause.json')
        stat, feedback_score = stat_handler.collect_root_cause(
            prefer_feedback='perception', stat_save_path=stat_save_path)

        # with open(os.path.join(self.runner_msg_folder, f'{seed_id}_root_cause.json'), 'w') as f:
        #     json.dump(stat, f, indent=4)
        return stat, feedback_score
