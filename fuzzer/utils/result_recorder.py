import re
import copy
import json
import math
import os

from datetime import datetime
from typing import Dict

from scenario.configuration import SeedConfig
from scenario.pattern import ScenarioTreeType
from scenario.utils.traffic_events import TrafficEventType


class ResultRecorder:

    def __init__(self, save_path: str, resume: bool = True):
        """
        :param save_path: json file
        :param resume:
        """
        self.save_path = save_path
        self.resume = resume
        self.results = list()  # list of dict element
        self.finished_cases = list()

        if os.path.isfile(self.save_path):
            if resume:
                with open(self.save_path, 'r') as f:
                    json_data = json.load(f)

                self.finished_cases = json_data['overview']
                self.results = json_data['results']
            else:
                os.remove(self.save_path)

            filter_results = list()
            for i, res in enumerate(self.results):
                for item in self.finished_cases:
                    finished_id = item['scenario_id']
                    if res['id'] == finished_id:
                        filter_results.append(res)
                        break

            self.results = filter_results

    def get_last_iteration(self):
        if len(self.finished_cases) == 0:
            return 0
        case = self.finished_cases[-1]
        iteration_id = int(case['scenario_id'].split('_')[-1])
        return iteration_id

    def have_finished(self, scenario_id):
        for item in self.finished_cases:
            if item['scenario_id'] == scenario_id:
                return True
        return False

    def update(
            self,
            config: SeedConfig,
            master_scenario: ScenarioTreeType,
            duration_time_system,
            duration_time_game
    ):
        res = self.parse_runner(master_scenario, duration_time_system, duration_time_game)
        res['id'] = config.id
        now = datetime.now()
        res['timestamp'] = datetime.timestamp(now)
        self.add_item(config.id, res)
        self.write_file()
        return copy.deepcopy(res)

    def add_item(self, case_id: str, one_result: Dict):
        self.finished_cases.append({
            'scenario_id': case_id,
            'scenario_result': one_result['status']
        })
        self.results.append(copy.deepcopy(one_result))

    def write_file(self):
        with open(self.save_path, 'w') as f:
            final_data = {
                'overview': self.finished_cases,
                'results': self.results
            }
            json.dump(final_data, f, indent=4)

    def clear(self):
        """
        Clears all tracked violations
        """
        self.results = list()
        self.finished_cases = list()

    @staticmethod
    def parse_runner(
            master_scenario: ScenarioTreeType,
            duration_time_system,
            duration_time_game
    ):
        target_reached = False

        meta = dict()
        meta['duration_system'] = duration_time_system
        meta['duration_game'] = duration_time_game

        infractions = {
            'collisions_layout': list(),
            'collisions_pedestrian': list(),
            'collisions_vehicle': list(),
            'vehicle_blocked': list(),
            'route_timeout': list(),
            'collision_timestamp': list()
        }
        fitness = list()

        if master_scenario:
            if master_scenario.timeout_node.timeout:
                infractions['route_timeout'].append('Route timeout.')

            for node in master_scenario.get_criteria():
                if node.list_traffic_events:
                    # analyze all traffic events
                    for event in node.list_traffic_events:
                        if event.get_type() == TrafficEventType.COLLISION_STATIC:
                            infractions['collisions_layout'].append(event.get_message())
                            infractions['collision_timestamp'].append(get_timestamp_from_msg(event.get_message()))
                        elif event.get_type() == TrafficEventType.COLLISION_PEDESTRIAN:
                            infractions['collisions_pedestrian'].append(event.get_message())
                            infractions['collision_timestamp'].append(get_timestamp_from_msg(event.get_message()))
                        elif event.get_type() == TrafficEventType.COLLISION_VEHICLE:
                            infractions['collisions_vehicle'].append(event.get_message())
                            infractions['collision_timestamp'].append(get_timestamp_from_msg(event.get_message()))
                        elif event.get_type() == TrafficEventType.VEHICLE_BLOCKED:
                            infractions['vehicle_blocked'].append(event.get_message())
                        elif event.get_type() == TrafficEventType.ROUTE_COMPLETED:
                            target_reached = True

            for node in master_scenario.get_fitness():
                fitness.append((node.name, node.actual_value))

        # update status
        if target_reached:
            status = 'Completed'
        else:
            status = 'Failed'

        res_record = {
            'status': status,
            'meta': meta,
            'infractions': infractions,
            'fitness': fitness
        }
        return res_record


def get_timestamp_from_msg(msg):
    pattern = r't=([\d.]+)'
    matches = re.findall(pattern, msg)
    if matches:
        return int(float(matches[0]))
