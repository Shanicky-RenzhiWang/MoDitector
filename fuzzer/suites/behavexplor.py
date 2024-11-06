import os
import pickle
import re
import glob
import json
import copy
import logging
import numpy as np

from typing import List
from loguru import logger
from datetime import datetime
from omegaconf import DictConfig
from sklearn.cluster import KMeans
from scipy import interpolate
from tsfresh.feature_extraction import feature_calculators

from scenario.configuration.waypoint import ScenarioConfig, SeedConfig

from .. import BaseFuzzer
# from ..mutator.mutation_space import MUTATION_SPACE
from ..mutator import ScenarioMutator
from ..root_cause_util.root_cause_ana import StatHandler
from time import time

def get_instance_logger(instance_name, log_file):
    """
    Create a unique logger for each instance.

    :param instance_name: A unique name for the instance.
    :param log_file: The file where logs for this instance should be written.
    :return: A configured logger object.
    """
    logger = logging.getLogger(instance_name)
    logger.setLevel(logging.DEBUG)  # Set your desired logging level

    # Remove all handlers associated with the logger
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create file handler which logs even debug messages
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    # Create formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(fh)

    return logger

def static_calculator(X):
    X_feature = []
    attribution_size = X.shape[1]
    for attr_i in range(attribution_size):
        attribution_i = X[:, attr_i]
        mean = feature_calculators.mean(attribution_i)
        minimum = feature_calculators.minimum(attribution_i)
        maximum = feature_calculators.maximum(attribution_i)
        mean_change = feature_calculators.mean_change(attribution_i)
        mean_abs_change = feature_calculators.mean_abs_change(attribution_i)
        variance = feature_calculators.variance(attribution_i)
        c3 = feature_calculators.c3(attribution_i, 1)
        cid_ce = feature_calculators.cid_ce(attribution_i, True)

        attribution_i_feature = [mean, variance, minimum, maximum, mean_change, mean_abs_change, c3, cid_ce]

        X_feature += attribution_i_feature

    return X_feature

class FeatureNet(object):

    def __init__(self, save_root, window_size=1):
        self.resample_frequency = 0
        self.window_size = window_size # unit is second (s)
        self.local_feature_extractor = None
        self.data_root = os.path.join(save_root, 'agent_data')

    @staticmethod
    def input_resample(xs, ts, resample='linear', sample_frequency=0.1):
        # x: [t, m], t: [t]
        x = np.array(xs)
        resample_axis = np.arange(ts[0], ts[-1], sample_frequency)
        new_x = []
        for i in range(0, x.shape[1]):
            x_i = x[:, i] # [t]
            f_i = interpolate.interp1d(ts, x_i, kind=resample)
            new_x_i = f_i(resample_axis) # [t]
            new_x_i = np.append(new_x_i, x_i[-1])
            new_x.append(new_x_i)
        new_x = np.array(new_x)
        new_x = new_x.T
        # new_x: [t, m]
        return new_x

    def forward(self, seed: SeedConfig, resample='linear'):
        # use attributes: heading, speed, acceleration
        seed_id = seed.id
        meta_folder = os.path.join(self.data_root, seed_id, 'meta_data')
        scene_file_pattern = f'{meta_folder}/*.pkl'
        scene_name_pattern = r'(\d+)\.pkl'
        scene_files = glob.glob(scene_file_pattern)
        sorted_scene_files = sorted(scene_files, key=lambda x: int(re.search(scene_name_pattern, x).group(1)))

        x = []
        for i in range(len(sorted_scene_files)):
            file_i = sorted_scene_files[i]
            with open(file_i, 'rb') as f:
                frame_data = pickle.load(f)

            ego_info = frame_data['ego']
            ego_velocity = np.array(ego_info['velocity'])
            ego_acceleration = np.array(ego_info['acceleration'])
            ego_control = np.array(ego_info['current_control'])

            ego_rotation = ego_info['rotation']
            pitch = np.deg2rad(ego_rotation[0])
            yaw = np.deg2rad(ego_rotation[1])
            ego_orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
            forward_speed = np.dot(ego_velocity, ego_orientation)
            forward_acc = np.dot(ego_acceleration, ego_orientation)

            lateral_orientation = np.array([
                -np.cos(pitch) * np.sin(yaw),
                np.cos(pitch) * np.cos(yaw),
                np.sin(pitch)
            ])
            lateral_speed = np.dot(ego_velocity, lateral_orientation)
            lateral_acceleration = np.dot(ego_acceleration, lateral_orientation)

            # speed, acceleration, heading, throttle, brake, steer
            ego_state_vector = np.array([forward_speed, lateral_speed, forward_acc, lateral_acceleration, yaw])
            frame_vector = np.concatenate((ego_state_vector, ego_control), axis=0)
            x.append(frame_vector)


        # aims to assign the time stamp for feature extraction!!!
        x_behavior_vector = np.array(x)

        time_size = x_behavior_vector.shape[0]
        if time_size < self.window_size:
            last_element = x_behavior_vector[-1:,:]
            for _ in range(self.window_size - time_size):
                x_behavior_vector = np.concatenate([x_behavior_vector, last_element], axis=0)

        y = []
        for i in range(time_size - self.window_size + 1):
            x_segment = x_behavior_vector[i:i+self.window_size]
            x_feature = static_calculator(x_segment)
            y.append(x_feature)

        return np.array(y)

class ClusterModelBehavior(object):
    def __init__(self, cluster_num):
        """
        Initial cluster number
        """
        self.cluster_model = KMeans(cluster_num)
        self.cluster_center = []
        self.cluster_data = None

    def search(self, v):
        """
        @param: v is the query feature
        """
        # v represents the behaviors of a single case
        # @format is numpy with shape (n, 64)
        # @output is numpy with shape (n, )
        cls_labels = self.cluster_model.predict(v)
        # nearest_node = self.AI.get_nns_by_vector(v, 1, include_distances=True)
        # label(node id) & distance
        return cls_labels

    def update(self, v):
        """
        Need to change to load all corpus and re-cluster
        """
        v[np.isnan(v)] = 0
        # Step1: add new behavior data @format is numpy with shape (n, 64)
        if self.cluster_data is None:
            self.cluster_data = v
        else:
            self.cluster_data = np.concatenate([self.cluster_data, v], axis=0)
        # Step2: retrain kmeans model.
        y = self.cluster_model.fit_predict(self.cluster_data) # shape (n, )
        return y

    def get_centers(self):
        return self.cluster_model.cluster_centers_

class CoverageModel(object):

    def __init__(self, save_root, window_size, cluster_num, threshold_coverage):

        self.coverage_centers = []
        self.coverage_centers_index = []
        self.coverage_centers_pointer = 0

        self.window_size = window_size
        self.cluster_num = cluster_num
        self.threshold_coverage = threshold_coverage

        self.dynamic_threshold = np.inf

        self.feature_layer = FeatureNet(save_root, window_size)
        self.cluster_layer_behavior = ClusterModelBehavior(self.cluster_num)

    def _extract_feature(self, seed: SeedConfig, resample='linear'):
        y_behavior = self.feature_layer.forward(seed, resample)
        return y_behavior

    def initialize(self, scenarios: List[SeedConfig]):
        """
        X_behavior: list [item1, item2, ..., itemn]
            itemi : array [[x1...], [x2...]]
        X_trace: list [item1, item2, ..., itemn]
            itemi: list: [(x1, y1), (x2, y2), ..., (xn, yn)]
        """
        X_behavior = []
        for item in scenarios:
            X_behavior.append(self._extract_feature(item, 'linear'))

        # behavior model
        buffer_feature = None
        for i in range(len(X_behavior)):
            x = X_behavior[i] # shape (n, 64)
            if buffer_feature is None:
                buffer_feature = x
            else:
                buffer_feature = np.concatenate([buffer_feature, x], axis=0)

            self.coverage_centers_index.append([self.coverage_centers_pointer,
                                                self.coverage_centers_pointer + x.shape[0]])
            self.coverage_centers_pointer += x.shape[0]

        # initial train
        y = self.cluster_layer_behavior.update(buffer_feature) # n x 64
        self.update(y)

    def update(self, y):
        """
        y is the class labels of all cases
         shape is (n, ), the cluster label sequence.
        """
        self.coverage_centers = []
        for item in self.coverage_centers_index:
            start_index = item[0]
            end_index = item[1]
            y_i = y[start_index:end_index]
            self.coverage_centers.append(y_i)
        self._update_threshold()

    def _update_threshold(self):
        pattern_num = len(self.coverage_centers)
        distance_matrix = np.zeros((pattern_num, pattern_num))
        for i in range(pattern_num):
            distance_matrix[i][i] = 1000
            for j in range(i + 1, pattern_num):
                tmp_distance = self._compute_distance_behavior_states(self.coverage_centers[i], self.coverage_centers[j])
                distance_matrix[i][j] = tmp_distance
                distance_matrix[j][i] = tmp_distance

        pattern_min_distance = []
        for i in range(pattern_num):
            pattern_i_min = np.min(distance_matrix[i])
            pattern_min_distance.append(pattern_i_min)
        pattern_min_distance = np.array(pattern_min_distance)
        self.dynamic_threshold = np.mean(pattern_min_distance)

    def feedback_coverage_behavior(self, seed: SeedConfig):

        x = self._extract_feature(seed, 'linear')

        y_behavior = self.cluster_layer_behavior.search(x)
        find_new_coverage = False
        min_feedback = np.inf
        for i in range(len(self.coverage_centers)):
            cov_feedback = self._compute_distance_behavior_states(y_behavior, self.coverage_centers[i])
            if cov_feedback < min_feedback:
                min_feedback = cov_feedback

        # if min_feedback > min(self.dynamic_threshold, self.threshold_coverage):
        if min_feedback > self.threshold_coverage:
            find_new_coverage = True
            # if no_pass: # ignore whether the seed is fail or pass
            self.coverage_centers_index.append([self.coverage_centers_pointer,
                                                self.coverage_centers_pointer + x.shape[0]])
            self.coverage_centers_pointer += x.shape[0]
            # update behavior model (kmeans)
            y = self.cluster_layer_behavior.update(x)
            # update existing centers
            self.update(y)

        return find_new_coverage, min_feedback, y_behavior

    @staticmethod
    def _compute_distance_behavior_states(y1, y2):
        """
        y1 is a list
        """
        # y is numpy
        y1_length = len(y1)
        y2_length = len(y2)

        coverage_score = abs(y1_length - y2_length)

        common_length = min(y1_length, y2_length)
        y1_common = y1[:common_length]
        y2_common = y2[:common_length]
        for i in range(common_length):
            y1_e = y1_common[i]
            y2_e = y2_common[i]
            if y1_e == y2_e:
                continue
            else:
                coverage_score += 1

        coverage_score /= float(max(y1_length, y2_length))

        return coverage_score

    def get_centers(self):
        return self.coverage_centers


class BehAVExplor(BaseFuzzer):

    # def __init__(
    #         self,
    #         save_root: str,
    #         tester_config: DictConfig,
    #         server_config: DictConfig,
    #         scenario_config: DictConfig,
    #         agent_config: DictConfig,
    # ):
    #     super(BehAVExplor, self).__init__(
    #         save_root,
    #         tester_config,
    #         server_config,
    #         scenario_config,
    #         agent_config
    #     )
    def __init__(self, save_root: str, cfg: DictConfig, server_config: DictConfig):
        super().__init__(save_root, cfg, server_config)

        # self.seed_path = self.tester_config.seed_path
        # self.resume = self.tester_config.resume  # todo: add this
        # self.time_limit = self.tester_config.time_limit

        # # load seed scenario
        # with open(self.seed_path, 'r') as f:
        #     seed_json = json.load(f)
        # seed_scenario = ScenarioConfig.from_json(seed_json)
        # self.seed = self.get_seed_config('0', seed_scenario)

        # self.seed_name = os.path.basename(self.seed_path)
        # self.seed_name = self.seed_name.split('.')[0]
        # self.scenario_mutator = ScenarioMutator(
        #     self.client
        # )
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

        # inner parameters
        self.iteration = self.result_recorder.get_last_iteration()  # note should add 1

        self.window_size = 10 # todo: confirm this
        self.cluster_num = 20
        self.threshold_coverage = 0.4
        self.threshold_energy = 0.8
        self.feature_resample = 'linear'
        self.initial_corpus_size = 4

        self.coverage_model = CoverageModel(self.save_root,
                                            self.window_size,
                                            self.cluster_num,
                                            self.threshold_coverage)

        self.corpus = list()  # save all elements in the fuzzing
        self.corpus_energy = list()
        self.corpus_fail = list()
        self.corpus_mutation = list()

        # config fitness
        debug_folder = os.path.join(self.save_root, 'debug')
        if not os.path.exists(debug_folder):
            os.makedirs(debug_folder)
        fitness_log_file = os.path.join(debug_folder, f"fitness.log")
        mutation_log_file = os.path.join(debug_folder, f"mutation.log")
        if os.path.isfile(fitness_log_file):
            os.remove(fitness_log_file)
        if os.path.isfile(mutation_log_file):
            os.remove(mutation_log_file)

        self.logger_fitness = get_instance_logger(f"fitness", fitness_log_file)
        self.logger_mutation = get_instance_logger("mutation", mutation_log_file)

        self.logger_fitness.info("Logger initialized for fitness")
        self.logger_mutation.info("Logger initialized for mutation")

    def _seed_selection(self):
        select_probabilities = copy.deepcopy(self.corpus_energy)
        select_probabilities = np.array(select_probabilities) + 1e-5
        select_probabilities /= (select_probabilities.sum())
        source_seed_index = np.random.choice(list(np.arange(0, len(self.corpus))), p=select_probabilities)
        return source_seed_index

    def run(self):
        # minimize is better
        logger.info('===== Start Fuzzer (BehAVExplor) =====')
        start_time = datetime.now()
        self.runner_msg_folder = os.path.join(self.save_root, 'runner_msg')
        os.makedirs(self.runner_msg_folder, exist_ok=True)

        restart = True
        checkpoint_file = os.path.join(self.save_root, 'checkpoint.pkl')
        if self.resume and os.path.isfile(checkpoint_file):
            logger.info(f'===> Load checkpoint from {checkpoint_file}')
            checkpoint_data = self.load_checkpoint(checkpoint_file)
            self.corpus = checkpoint_data['corpus']
            self.corpus_fail = checkpoint_data['corpus_fail']
            self.corpus_mutation = checkpoint_data['corpus_mutation']
            self.corpus_energy = checkpoint_data['corpus_energy']
            if self.corpus is None or len(self.corpus) == 0:
                restart = True
            else:
                restart = False

        if restart:
            self.corpus = list()  # save all elements in the fuzzing
            self.corpus_energy = list()
            self.corpus_fail = list()
            self.corpus_mutation = list()
            # generate initial scenario
            while len(self.corpus) < self.initial_corpus_size:
                self.iteration += 1
                if self.termination(start_time, self.time_limit):
                    break
                logger.info(f'==================== Run Iter {self.iteration} ====================')
                # # generate new task
                scenario_id = f"{self.scenario_id_pref}{self.iteration}"
                # mutated_scenario = self.scenario_mutator.generate_scenario(
                #     self.seed.scenario,
                #     scenario_id,
                #     npc_vehicle_num=MUTATION_SPACE[self.seed_name]['vehicle_num'],
                #     npc_walker_num=MUTATION_SPACE[self.seed_name]['walker_num'],
                #     traffic_light_green=MUTATION_SPACE[self.seed_name]['ignore_traffic_light'],
                # )
                mutated_scenario = self.scenario_mutator.generate_new_scenario(self.seed.scenario, scenario_id)
                # save scenario
                scenario_folder = os.path.join(self.save_root, 'scenario')
                if not os.path.exists(scenario_folder):
                    os.makedirs(scenario_folder)
                scenario_file = os.path.join(scenario_folder, f"{mutated_scenario.id}.json")
                with open(scenario_file, 'w') as f:
                    scenario_json_data = mutated_scenario.json_data()
                    json.dump(scenario_json_data, f, indent=4)

                mutated_seed = self.get_seed_config(scenario_id, mutated_scenario)
                runner_pass, runner_message = self.load_and_run_scenario(mutated_seed)
                if not runner_pass:
                    logger.warning('Scenario Runner has bug: {}', runner_message)
                    break
                with open(os.path.join(self.runner_msg_folder, f'{mutated_seed.id}.json'), 'w') as f:
                    json.dump(runner_message, f, indent=4)
                root_cause, feedback_score = self._get_root_cause_stat(mutated_seed.id, runner_message=runner_message)

                mutated_seed = self.update_seed_config(mutated_seed, runner_message['result'])
                seed_file = os.path.join(self.seed_folder, f"{mutated_seed.id}.json")
                with open(seed_file, 'w') as f:
                    seed_json_data = mutated_seed.json_data()
                    json.dump(seed_json_data, f, indent=4)

                # for obj in gc.get_objects():
                #     try:
                #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                #             print(type(obj), obj.size())
                #     except:
                #         pass

                logger.info(f'Seed {mutated_seed.id} Result: {mutated_seed.oracle} Fitness: {mutated_seed.fitness}')
                self.logger_fitness.info(f"{mutated_seed.id},{mutated_seed.fitness}")
                self.logger_mutation.info(f"{mutated_seed.id},{mutated_seed.id},initial,1.0")

                if not mutated_seed.oracle['complete']:
                    continue

                # check conditions
                self.corpus.append(copy.deepcopy(mutated_seed))
                # the following items are used for energy
                self.corpus_fail.append(0)
                self.corpus_mutation.append(0)
                self.corpus_energy.append(1)

            checkpoint_dict = {
                'corpus': self.corpus,
                'corpus_fail': self.corpus_fail,
                'corpus_mutation': self.corpus_mutation,
                'corpus_energy': self.corpus_energy
            }
            self.save_checkpoint(checkpoint_dict, checkpoint_file)

        # initialize the coverage model
        if len(self.corpus) > 0:
            m_start_time = datetime.now()
            self.coverage_model.initialize(self.corpus)
            m_end_time = datetime.now()
            logger.info('Coverage Spend Time: [=]{}[=]', (m_end_time - m_start_time).total_seconds())

        while True:
            self.iteration += 1
            logger.info(f'==================== Run Iter {self.iteration} ====================')
            scenario_id = f"{self.scenario_id_pref}{self.iteration}"
            if self.termination(start_time, self.time_limit):
                return
            iter_time = time()
            source_seed_index = self._seed_selection()  # select based on energy, high energy is better
            select_time = time()
            self.times['seed_select'].append(select_time-iter_time)
            source_seed = self.corpus[source_seed_index]
            source_seed_energy = self.corpus_energy[source_seed_index]
            source_seed_fail = self.corpus_fail[source_seed_index]
            source_seed_mutation = self.corpus_mutation[source_seed_index]

            if source_seed_energy > self.threshold_energy:
                mutation_stage = 'small'
            else:
                mutation_stage = 'large'

            if mutation_stage == 'small':
                # mutation
                mutated_seed = copy.deepcopy(source_seed)
                _, mutated_seed_scenario = self.scenario_mutator.mutate_current_scenario(
                    source_seed.scenario,
                    scenario_id,
                    # prob_mutation=0.6
                )
                mutated_seed.scenario = mutated_seed_scenario
                mutated_seed.id = scenario_id
            else:
                # generate new scenario
                # mutated_seed_scenario = self.scenario_mutator.generate_scenario(
                #     self.seed.scenario,
                #     scenario_id,
                #     npc_vehicle_num=MUTATION_SPACE[self.seed_name]['vehicle_num'],
                #     npc_walker_num=MUTATION_SPACE[self.seed_name]['walker_num'],
                #     traffic_light_green=MUTATION_SPACE[self.seed_name]['ignore_traffic_light'],
                # )
                mutated_scenario = self.scenario_mutator.generate_new_scenario(self.seed.scenario, scenario_id)
                mutated_seed = self.get_seed_config(scenario_id, mutated_seed_scenario)
            mutate_time = time()
            self.times['mutate'].append(mutate_time-select_time)

            runner_pass, runner_message = self.load_and_run_scenario(mutated_seed)
            if not runner_pass:
                logger.warning('Scenario Runner has bug: {}', runner_message)
                break
            with open(os.path.join(self.runner_msg_folder, f'{mutated_seed.id}.json'), 'w') as f:
                json.dump(runner_message, f, indent=4)
            simulation_time = time()
            self.times['simulation'].append(simulation_time-mutate_time)
            root_cause, feedback_score = self._get_root_cause_stat(mutated_seed.id, runner_message=runner_message)
            simulation_time = time()
            

            mutated_seed = self.update_seed_config(mutated_seed, runner_message['result'])
            # save scenario
            seed_file = os.path.join(self.seed_folder, f"{mutated_seed.id}.json")
            with open(seed_file, 'w') as f:
                seed_json_data = mutated_seed.json_data()
                json.dump(seed_json_data, f, indent=4)

            follow_up_seed_is_new, follow_up_seed_div, follow_up_seed_ab = self.coverage_model.feedback_coverage_behavior(
                mutated_seed)

            logger.info(f'Seed {mutated_seed.id} Result: {mutated_seed.oracle} Fitness: {mutated_seed.fitness} Cov: {follow_up_seed_div}')
            self.logger_fitness.info(f"{mutated_seed.id},{mutated_seed.fitness}")
            self.logger_mutation.info(f"{mutated_seed.id},{source_seed.id},{mutation_stage},{source_seed_energy}")

            # update energy & fail
            benign = True
            if mutated_seed.oracle['collision']:
                # we only need collision -> so use collision only here
                source_seed_fail += 1
                benign = False

            source_seed_mutation += 1.0
            if mutation_stage == 'large':
                source_seed_energy = source_seed_energy - 0.2
            else:
                # update energy of source_seed
                # if float(source_seed_mutation) == 0:
                #     source_seed_mutation = 1.0

                delta_fail = source_seed_fail / float(source_seed_mutation) # 1, infinite
                if benign:
                    delta_fail = min(delta_fail, 1.0)
                    delta_fail = -0.5 * (1 - delta_fail)
                else:
                    delta_fail = min(delta_fail, 1.0)

                delta_fitness = source_seed.fitness - mutated_seed.fitness  # min is better
                delta_select = -0.2

                div_score = float(np.clip(follow_up_seed_div - self.threshold_coverage, -1.0, 1.0))
                source_seed_energy = source_seed_energy + delta_fail + 0.5 * np.tanh(delta_fitness) + delta_select + 0.5 * div_score

            # update information
            self.corpus_energy[source_seed_index] = float(np.clip(source_seed_energy, 1e-5, 4.0))
            self.corpus_fail[source_seed_index] = source_seed_fail
            self.corpus_mutation[source_seed_index] = source_seed_mutation
            feedback_time = time()
            self.times['feedback'].append(feedback_time-simulation_time)
            times_res ={'ori':self.times,
                        'average':{k:sum(v)/len(v) for k,v in self.times.items()}}
            times_file = os.path.join(self.save_root,'results','times_consume.json')
            with open(times_file,'w') as f:
                json.dump(times_res,f, indent=4)

            # calculate the diversity based on the record
            m_start_time = datetime.now()
            if follow_up_seed_is_new or mutated_seed.fitness < source_seed.fitness:
                self.corpus.append(copy.deepcopy(mutated_seed))
                self.corpus_fail.append(0)
                self.corpus_mutation.append(0)
                initial_energy = 1 + min(1.0, follow_up_seed_div)
                self.corpus_energy.append(initial_energy)
            m_end_time = datetime.now()
            logger.info('Coverage Spend Time: [=]{}[=]', (m_end_time - m_start_time).total_seconds())

            # save checkpoint
            checkpoint_dict = {
                'corpus': self.corpus,
                'corpus_fail': self.corpus_fail,
                'corpus_mutation': self.corpus_mutation,
                'corpus_energy': self.corpus_energy
            }
            self.save_checkpoint(checkpoint_dict, checkpoint_file)

    def save_checkpoint(self, checkpoint_dict, checkpoint_file):
        # corpus
        corpus = list()
        for item in checkpoint_dict['corpus']:
            corpus.append(item.json_data())
        checkpoint_dict['corpus'] = corpus

        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_dict, f)

    def load_checkpoint(self, checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            checkpoint_dict = pickle.load(f)

        corpus = list()
        for item in checkpoint_dict['corpus']:
            corpus.append(SeedConfig.from_json(item))
        checkpoint_dict['corpus'] = corpus

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