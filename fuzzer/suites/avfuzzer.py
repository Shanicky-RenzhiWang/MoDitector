import os
import gc
import torch
import json
import copy
import pickle
import random
import numpy as np

from loguru import logger
from datetime import datetime
from omegaconf import DictConfig

from fuzzer.mutator import ScenarioMutator
from scenario.configuration import ScenarioConfig, SeedConfig

from ..base import BaseFuzzer
from ..root_cause_util.root_cause_ana import StatHandler
from time import time

class AVFuzzer(BaseFuzzer):

    def __init__(self, save_root: str, cfg: DictConfig, server_config: DictConfig):
        super(AVFuzzer, self).__init__(save_root, cfg, server_config)

        self.seed_path = self.cfg.seed_path
        self.time_limit = self.cfg.time_limit

        self.resume = self.cfg.resume
        self.population_size = 4
        self.pc = 0.5
        self.pm = 0.5

        # load seed scenario
        with open(self.seed_path, 'r') as f:
            seed_json = json.load(f)
        seed_scenario = ScenarioConfig.from_json(seed_json)
        self.seed = self.get_seed_config('0', seed_scenario)

        self.fitness_file = os.path.join(self.save_root, 'fitness.csv')

        # scenario runner
        # self.scenario_runner = ScenarioRunner(cfg, self.client, self.traffic_manager, self.result_recorder)
        self.mutator = ScenarioMutator(
            self.client,
            self.cfg.mutator_vehicle_num,
            self.cfg.mutator_walker_num,
            self.cfg.mutator_static_num,
            self.pm,
            2.0
        )

        # inner parameters
        self.seed_id = self.result_recorder.get_last_iteration()  # note should add 1

    def _seed_selection(self, curr_population, prev_population):
        # fitness -> min is better
        tmp_population = curr_population + prev_population

        tmp_fitness = list()
        for i in range(len(tmp_population)):
            tmp_p_i_fitness = tmp_population[i].fitness
            tmp_fitness.append(tmp_p_i_fitness + 1e-5)

        tmp_fitness_sum = float(sum(tmp_fitness))
        tmp_probabilities = np.array([(tmp_f / tmp_fitness_sum) for tmp_f in tmp_fitness])
        tmp_probabilities = 1 - np.array(tmp_probabilities)
        tmp_probabilities /= tmp_probabilities.sum()

        next_parent = list()
        # next_parent = [copy.deepcopy(self.best_seed)]
        while len(next_parent) < self.population_size:
            select = np.random.choice(tmp_population, p=tmp_probabilities)
            select.agent = None
            next_parent.append(copy.deepcopy(select))

        return next_parent

    def crossover_mutation(self, population):
        for i in range(len(population)):
            population[i].agent = None
        mutated_population = copy.deepcopy(population)
        for i in range(int(self.population_size / 2.0)):
            # Check crossover probability
            if self.pc > random.random():
                # randomly select 2 chromosomes(scenarios) in pops
                i = 0
                j = 0
                while i == j:
                    i = random.randint(0, self.population_size - 1)
                    j = random.randint(0, self.population_size - 1)
                pop_i = mutated_population[i]
                pop_j = mutated_population[j]

                pop_i_s, pop_j_s = self.mutator.crossover(pop_i.scenario, pop_j.scenario, self.pm)

                mutated_population[i].scenario = pop_i_s
                mutated_population[j].scenario = pop_j_s

        for i in range(self.population_size):

            if self.pm > random.random():
                _, mutated_scenario = self.mutator.mutate_current_scenario(mutated_population[i].scenario, 'tmp')
                mutated_population[i].scenario = mutated_scenario

        # update seed_id
        for i in range(self.population_size):
            self.seed_id += 1
            seed_id = f"{self.scenario_id_pref}{self.seed_id}"  # TODO: modify this
            mutated_population[i].id = seed_id
            mutated_population[i].scenario.id = seed_id
        return mutated_population

    def run(self):
        self.runner_msg_folder = os.path.join(self.save_root, 'runner_msg')
        os.makedirs(self.runner_msg_folder, exist_ok=True)
        start_time = datetime.now()
        restart = False
        while True:
            if self.termination(start_time, self.time_limit):
                break
            restart = self._run_global(start_time, restart)

    def _run_global(self, start_time, restart=False):
        # minimize is better
        logger.info('===== Start Fuzzer (AVFuzzer) =====')

        checkpoint_file = os.path.join(self.save_root, 'checkpoint.pkl')

        if self.resume and (not restart) and os.path.isfile(checkpoint_file):
            logger.info(f'===> Load checkpoint from {checkpoint_file}')
            checkpoint_data = self.load_checkpoint(checkpoint_file)
            self.best_seed = checkpoint_data['best_seed']
            self.best_fitness_after_restart = checkpoint_data['best_fitness_after_restart']
            self.best_fitness_lst = checkpoint_data['best_fitness_lst']

            self.pm = checkpoint_data['pm']
            self.pc = checkpoint_data['pc']

            self.minLisGen = checkpoint_data['minLisGen']  # Min gen to start LIS
            self.curr_population = checkpoint_data['curr_population']
            self.prev_population = checkpoint_data['prev_population']
            self.curr_iteration = checkpoint_data['curr_iteration']
            self.last_restart_iteration = checkpoint_data['last_restart_iteration']
            self.population_size = checkpoint_data['population_size']
            self.noprogress = checkpoint_data['noprogress']
            if len(self.curr_population) < self.population_size:
                need_init = True
            else:
                need_init = False
        else:
            need_init = True

        if need_init:
            self.best_seed = None
            self.best_fitness_after_restart = 10
            self.best_fitness_lst = []

            self.pm = 0.6
            self.pc = 0.6

            self.minLisGen = 5  # Min gen to start LIS
            self.curr_population = list()
            self.prev_population = list()
            self.curr_iteration = 0
            self.last_restart_iteration = 0
            self.population_size = 4
            self.noprogress = False
            self.local_runner = False

            initial_scenario = self.mutator.generate_new_scenario(self.seed.scenario, 'none')
            while len(self.curr_population) < self.population_size:
                self.seed_id += 1
                seed_id = f"{self.scenario_id_pref}{self.seed_id}"
                _, mutated_scenario = self.mutator.mutate_current_scenario(initial_scenario, seed_id)
                mutated_seed = self.get_seed_config(seed_id, mutated_scenario)
                logger.info(f'==================== Run Seed {mutated_seed.id} ====================')
                runner_pass, runner_message = self.load_and_run_scenario(mutated_seed)
                if not runner_pass:
                    logger.warning('Scenario Runner has bug: {}', runner_message)
                    break
                with open(os.path.join(self.runner_msg_folder, f'{mutated_seed.id}.json'), 'w') as f:
                    json.dump(runner_message, f, indent=4)
                root_cause, feedback_score = self._get_root_cause_stat(mutated_seed.id, runner_message=runner_message)

                mutated_seed = self.update_seed_config(mutated_seed, runner_message['result'])
                # save scenario
                seed_file = os.path.join(self.seed_folder, f"{mutated_seed.id}.json")
                with open(seed_file, 'w') as f:
                    seed_json_data = mutated_seed.json_data()
                    json.dump(seed_json_data, f, indent=4)
                for obj in gc.get_objects():
                    try:
                        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                            print(type(obj), obj.size())
                    except Exception:
                        pass

                logger.info(f'Seed {mutated_seed.id} Result: {mutated_seed.oracle} Fitness: {mutated_seed.fitness}')
                self.curr_population.append(copy.deepcopy(mutated_seed))

            # save checkpoint
            checkpoint_dict = {
                'best_seed': self.best_seed,
                'best_fitness_after_restart': self.best_fitness_after_restart,
                'best_fitness_lst': self.best_fitness_lst,
                'pm': self.pm,
                'pc': self.pc,
                'minLisGen': self.minLisGen,
                'curr_population': self.curr_population,
                'prev_population': self.prev_population,
                'curr_iteration': self.curr_iteration,
                'last_restart_iteration': self.last_restart_iteration,
                'population_size': self.population_size,
                'noprogress': self.noprogress
            }
            self.save_checkpoint(checkpoint_dict, checkpoint_file)

            with open(self.fitness_file, 'w') as f:
                for item in self.best_fitness_lst:
                    f.write(f"{item},\n")

        while True:  # i th generation.

            if self.termination(start_time, self.time_limit):
                return False

            self.curr_iteration += 1
            logger.info('===== Iteration {} =====', self.curr_iteration)
            iter_time = time()
            if not self.noprogress:
                if self.curr_iteration == 1:
                    for i in range(len(self.curr_population)):
                        self.curr_population[i].agent = None
                    self.prev_population = copy.deepcopy(self.curr_population) # list of seeds
                else:
                    self.prev_population = self._seed_selection(self.curr_population, self.prev_population)
                # mutation
                self.curr_population = self.crossover_mutation(self.prev_population)
            else:
                # restart
                initial_scenario = self.mutator.generate_new_scenario(self.seed.scenario, 'none')
                for i in range(self.population_size):
                    self.seed_id += 1
                    seed_id = f"{self.scenario_id_pref}{self.seed_id}"  # TODO: modify this
                    _, mutated_scenario = self.mutator.mutate_current_scenario(initial_scenario, seed_id)
                    self.curr_population[i].id = seed_id
                    self.curr_population[i].scenario = mutated_scenario
                self.best_seed = None
            mutate_time = time()
            self.times['mutate'].append(mutate_time-iter_time)
            # run
            for i in range(self.population_size):
                if self.termination(start_time, self.time_limit):
                    return False

                curr_seed = self.curr_population[i]
                logger.info(f'==================== Run Seed {curr_seed.id} ====================')
                runner_pass, runner_message = self.load_and_run_scenario(curr_seed)
                if not runner_pass:
                    logger.warning('Scenario Runner has bug: {}', runner_message)
                    break
                with open(os.path.join(self.runner_msg_folder, f'{curr_seed.id}.json'), 'w') as f:
                    json.dump(runner_message, f, indent=4)
                simulation_time = time()
                self.times['simulation'].append(simulation_time-mutate_time)
                root_cause, feedback_score = self._get_root_cause_stat(curr_seed.id, runner_message=runner_message)

                curr_seed = self.update_seed_config(curr_seed, runner_message['result'])
                # save scenario
                seed_file = os.path.join(self.seed_folder, f"{curr_seed.id}.json")
                with open(seed_file, 'w') as f:
                    seed_json_data = curr_seed.json_data()
                    json.dump(seed_json_data, f, indent=4)

                for obj in gc.get_objects():
                    try:
                        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                            print(type(obj), obj.size())
                    except Exception:
                        pass

                logger.info(f'Seed {curr_seed.id} Result: {curr_seed.oracle} Fitness: {curr_seed.fitness}')

                # check conditions
                if curr_seed.oracle['collision']:
                    logger.info('Find violation, exit fuzzer.') # todo: restart
                    return True
                s_time = time()
                self.curr_population[i] = curr_seed
                if self.best_seed is None or curr_seed.fitness < self.best_seed.fitness:
                    curr_seed.agent = None
                    self.best_seed = copy.deepcopy(curr_seed)
                
                feedback_time = time()
                self.times['feedback'].append(feedback_time-s_time)
                times_res ={'ori':self.times,
                        'average':{k:sum(v)/len(v) for k,v in self.times.items()}}
                times_file = os.path.join(self.save_root,'results','times_consume.json')
                with open(times_file,'w') as f:
                    json.dump(times_res,f, indent=4)

            self.best_fitness_lst.append(self.best_seed.fitness)
            if self.noprogress:
                self.best_fitness_after_restart = self.best_seed.fitness
                self.noprogress = False

            # check progress with previous 5 fitness
            ave = 0
            if self.curr_iteration >= self.last_restart_iteration + 5:
                for j in range(self.curr_iteration - 5, self.curr_iteration):
                    ave += self.best_fitness_lst[j]
                ave /= 5
                if ave <= self.best_seed.fitness:
                    self.last_restart_iteration = self.curr_iteration
                    self.noprogress = True

            # save checkpoint
            checkpoint_dict = {
                'best_seed': self.best_seed,
                'best_fitness_after_restart': self.best_fitness_after_restart,
                'best_fitness_lst': self.best_fitness_lst,
                'pm': self.pm,
                'pc': self.pc,
                'minLisGen': self.minLisGen,
                'curr_population': self.curr_population,
                'prev_population': self.prev_population,
                'curr_iteration': self.curr_iteration,
                'last_restart_iteration': self.last_restart_iteration,
                'population_size': self.population_size,
                'noprogress': self.noprogress
            }
            self.save_checkpoint(checkpoint_dict, checkpoint_file)

            logger.debug(self.best_fitness_lst)
            with open(self.fitness_file, 'w') as f:
                for item in self.best_fitness_lst:
                    f.write(f"{item},\n")

    def save_checkpoint(self, checkpoint_dict, checkpoint_file):
        # best_seed
        best_seed = checkpoint_dict['best_seed']
        if best_seed is not None:
            checkpoint_dict['best_seed'] = best_seed.json_data()
        # curr_population
        curr_population = list()
        for item in checkpoint_dict['curr_population']:
            curr_population.append(item.json_data())
        checkpoint_dict['curr_population'] = curr_population

        # prev_population
        prev_population = list()
        for item in checkpoint_dict['prev_population']:
            prev_population.append(item.json_data())
        checkpoint_dict['prev_population'] = prev_population

        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_dict, f)

    def load_checkpoint(self, checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            checkpoint_dict = pickle.load(f)

        best_seed = checkpoint_dict['best_seed']
        if best_seed is not None:
            best_seed = SeedConfig.from_json(checkpoint_dict['best_seed'])
        checkpoint_dict['best_seed'] = best_seed

        curr_population = list()
        for item in checkpoint_dict['curr_population']:
            curr_population.append(SeedConfig.from_json(item))
        checkpoint_dict['curr_population'] = curr_population

        prev_population = list()
        for item in checkpoint_dict['prev_population']:
            prev_population.append(SeedConfig.from_json(item))
        checkpoint_dict['prev_population'] = prev_population

        return checkpoint_dict
    
    def _get_root_cause_stat(self, seed_id, runner_message):
        base_dir = os.path.join(self.save_root, 'agent_data', seed_id)
        collision = runner_message['result']['infractions']['collision_timestamp']
        stat_handler = StatHandler(base_dir, collision)
        stat_save_path = os.path.join(self.runner_msg_folder, f'{seed_id}_root_cause.json')
        stat, feedback_score = stat_handler.collect_root_cause(prefer_feedback='perception', stat_save_path=stat_save_path)
        
        # with open(os.path.join(self.runner_msg_folder, f'{seed_id}_root_cause.json'), 'w') as f:
        #     json.dump(stat, f, indent=4)
        return stat, feedback_score
