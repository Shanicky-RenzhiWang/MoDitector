import glob
from ..utils.fileUtil import load_json
from collections import OrderedDict
import json
import numpy as np
import pathlib
import math


class StatHandler:
    def __init__(self, base_dir, collision_game_time=None):
        self.base_dir = pathlib.Path(base_dir)
        self.collision = [int(t)*50 for t in collision_game_time]
        self.timestamps = self.__get_timestamps()
        self.profile_categories = ['actors', 'pose', 'prediction_w_gt',
                                   'predictions', 'waypoint', 'prediction_with_all_gt', 'detect_eval']
        self.data = {}
        for pc in self.profile_categories:
            self.data[pc] = {}
            for data_file in glob.glob(self.base_dir.joinpath(pc).joinpath('*.json').as_posix()):
                timestamp = int(data_file.split('-')[-1].split('.')[0])
                self.data[pc][timestamp] = load_json(data_file)
        self.ego_id = None
        self._get_actual_traj_actor_view()
        self.configs = {
            'camera_loc': [1.3, 0.0, 1.8],
            'prediction_window': 10,
            'pred_score_threshold': 0.1,
        }
        self.TIMESTAP_PER_STEP = 50
        self.categories = {
            0: 'no failure',
            1: 'perception',
            2: 'prediction',
            3: 'perception and prediction',
            4: 'planning',
            5: 'perception and planning',
            6: 'prediction and planning',
            7: 'perception, prediction and planning',
            8: 'controller'
        }

    def __get_timestamps(self):
        timestamps = []
        for data_file in glob.glob(self.base_dir.joinpath('actors').joinpath('*.json').as_posix()):
            timestamp = int(data_file.split('-')[-1].split('.')[0])
            timestamps.append(timestamp)
        return sorted(timestamps)

    def _get_actual_traj_actor_view(self):
        self.actual_traj_actor_view = {}
        pose_timestamps = sorted(self.data['actors'].keys())
        for pt in pose_timestamps:
            for actor_id, actor_data in self.data['actors'][pt].items():
                if actor_id not in self.actual_traj_actor_view:
                    self.actual_traj_actor_view[actor_id] = {}
                    self.actual_traj_actor_view[actor_id]['extent'] = actor_data['extent']
                    self.actual_traj_actor_view[actor_id]['location'] = OrderedDict(
                    )
                    self.actual_traj_actor_view[actor_id]['location'][pt] = actor_data['location']
                else:
                    self.actual_traj_actor_view[actor_id]['location'][pt] = actor_data['location']
        for actor_id, actor_data in self.data['actors'][50].items():
            x, y = actor_data['location']['x'], actor_data['location']['y']
            ego_x, ego_y = float(self.data['pose'][50]['x']), float(
                self.data['pose'][50]['y'])
            if abs(x-ego_x) < 0.5 and abs(y-ego_y) < 0.5:
                self.ego_id = actor_id
                break

    def collect_root_cause(self, prefer_feedback=None, stat_save_path=None):
        '''
           root cause stats in 3 numbers, 1 means failure happens
           first number: perecption
           second number: prediction
           third number: planning
           000: no failure
           001: perception
           010: prediction
           011: perception and prediction
           100: planning
           101: perception and planning
           110: prediction and planning
           111: perception, prediction and planning
        '''
        self.root_cause_stats = {self.categories[i]: {
            'collision': 0, 'safe': 0} for i in range(len(self.categories))}
        factor = {
            'perception': 1,
            'prediction': 1,
            'planning': 1,
            'controller': 1
        }
        count = {
            'perception': 0,
            'prediction': 0,
            'planning': 0,
            'controller': 0
        }
        max_danger_score = -4
        if prefer_feedback:
            factor = {f: v*(1 if prefer_feedback == f else -1)
                      for f, v in factor.items()}
        for ts in self.timestamps:
            danger_score = 0
            fail_status = 0
            if prefer_feedback == 'perception':
                perception_failure, perception_danger_score = self._get_perception_stats(
                    ts)
                if perception_failure:
                    fail_status += 1
                    count['perception'] += 1
                danger_score += perception_danger_score * factor['perception']
            prediction_failure, prediction_danger_score = self._get_prediction_stats(
                ts)
            if prediction_failure:
                fail_status += 2
                count['prediction'] += 1
            danger_score += prediction_danger_score * factor['prediction']
            planning_failure, planning_danger_score = self._get_planning_stats(
                ts)
            if planning_failure:
                fail_status += 4
                count['planning'] += 1
            danger_score += planning_danger_score * factor['planning']
            control_failure, control_danger_score = self._get_control_stats(ts)
            danger_score += control_danger_score * factor['controller']
            collision_stats = self._check_timewindow_stats(ts)
            cs = 'collision' if collision_stats else 'safe'
            self.root_cause_stats[self.categories[fail_status]][cs] += 1
            if control_failure:
                self.root_cause_stats['controller'][cs] += 1
            if cs == 'collision':
                danger_score = 1.0
            if danger_score > max_danger_score:
                max_danger_score = danger_score
        # return self.root_cause_stats, max_danger_score
        # tmp_score = count[prefer_feedback]/sum_count if (sum_count:=sum(list(count.values()))) else 0.0
        if self.collision.__len__() > 0:
            prefer_cause_collision = False
            for k, v in self.root_cause_stats.items():
                if prefer_feedback in k and v['collision'] > 0:
                    prefer_cause_collision = True
                    break
            max_danger_score = 1.0 if prefer_cause_collision else max_danger_score - 0.5
        if stat_save_path:
            with open(stat_save_path, 'w') as f:
                json.dump(self.root_cause_stats, f, indent=4)
        return self.root_cause_stats, max_danger_score

    def _check_timewindow_stats(self, timestamp):
        if not self.collision:
            return False
        for col in self.collision:
            if timestamp < col <= timestamp + self.configs['prediction_window'] * self.TIMESTAP_PER_STEP:
                return True
        return False
    
    def _get_control_stats(self, timestamp):
        planning_data = self.data['waypoint'][timestamp]
        if not planning_data:
            return False, 0
        next_timestamp = timestamp + self.TIMESTAP_PER_STEP
        if next_timestamp > self.timestamps[-1]:
            return False, 0
        next_plan_loc = planning_data['1']
        next_actual_loc = self.data['pose'][next_timestamp]
        ego_x, ego_y = float(self.data['pose'][timestamp]['x']), float(self.data['pose'][timestamp]['y'])
        plan_dis_x, plan_dis_y = float(next_plan_loc['x']) - ego_x, float(next_plan_loc['y'])-ego_y
        actual_dis_x, actual_dis_y = float(next_actual_loc['x']) - ego_x, float(next_actual_loc['y']) - ego_y
        if abs(actual_dis_x) <0.6 and abs(actual_dis_y) <0.6:
            return False,0
        def judge_val(a,b):
            if abs(b) < 0.5:
                return 1
            diff = abs(a-b)
            if diff < 0.5:
                return 1
            return abs(a-b)/abs(b)
        x_score = judge_val(actual_dis_x, plan_dis_x)
        y_score = judge_val(actual_dis_y, plan_dis_y)
        if x_score < 0.1 and y_score<0.1:
            res = False
        else:
            res = True
        score = 1 - max(x_score, y_score)
        return res, score

    def _get_perception_stats(self, timestamp):
        if timestamp not in self.data['detect_eval']:
            return False, 0
        data = self.data['detect_eval'][timestamp].get('iou', None)
        if not data:
            return False, 0
        data = [dm for d in data if (dm := max(d)) > 0.05]
        if len(data) == 0:
            return False, 0
        mean_iou = np.mean(data)
        is_outlier = mean_iou < 0.5
        return is_outlier, 1-min(0.5, mean_iou)/0.5
    
    

    def _get_prediction_stats(self, timestamp):
        data = self.data['prediction_with_all_gt'].get(timestamp, None)
        if not data:
            return False, 0
        max_error = 0
        for pred_id, pred_data in data.items():
            if pred_id == self.ego_id:
                continue
            for future_ts, pred_loc in pred_data.items():
                future_timestamp = timestamp + \
                    int(future_ts)*self.TIMESTAP_PER_STEP
                if future_timestamp >= self.timestamps[-1]:
                    continue
                ego_x, ego_y = float(self.data['pose'][future_timestamp]['x']), float(
                    self.data['pose'][future_timestamp]['y'])
                yaw = float(self.data['pose'][timestamp]['yaw'])
                theta = math.radians(yaw)
                x = ego_x + pred_loc['x'] * \
                    math.cos(theta) - pred_loc['y'] * math.sin(theta)
                y = ego_y + pred_loc['x'] * \
                    math.sin(theta) + pred_loc['y'] * math.cos(theta)
                actual_x, actual_y = self.actual_traj_actor_view[pred_id]['location'][future_timestamp][
                    'x'], self.actual_traj_actor_view[pred_id]['location'][future_timestamp]['y']
                error = np.sqrt((x - actual_x)**2 + (y - actual_y)**2)
                dis = np.sqrt((ego_x - actual_x)**2 + (ego_y - actual_y)**2)
                score = error/dis
                if score > max_error:
                    max_error = score
                if dis < 4 and error/dis >= self.configs['pred_score_threshold']:
                    # if error/dis >= self.configs['pred_score_threshold']:
                    return True, max_error
        return False, max_error

    def _get_planning_stats(self, timestamp):
        planning_data = self.data['waypoint'][timestamp]
        prediction_data = self.data['predictions'][timestamp]
        yaw = float(self.data['pose'][timestamp]['yaw'])
        theta = math.radians(yaw)
        ego_x, ego_y = planning_data['0']['x'], planning_data['0']['y']
        if not prediction_data or not planning_data:
            return False, 0
        pred_id_in_gt = {}

        for pred_id, pred_data in prediction_data.items():
            x = float(pred_data['0']['x'])
            y = float(pred_data['0']['y'])
            min_dis = 100
            for pred_gt_id, pred_gt_data in self.data['prediction_with_all_gt'][timestamp].items():
                if pred_gt_id == self.ego_id:
                    continue
                gx = float(pred_gt_data['0']['x'])
                gy = float(pred_gt_data['0']['y'])
                dis = np.sqrt((gx - x)**2 + (gy - y)**2)
                if dis < min_dis:
                    min_dis = dis
                    pred_id_in_gt[pred_id] = pred_gt_id

        min_score = 1
        for future_ts, plan_loc in planning_data.items():
            if future_ts == '0':
                continue
            future_timestamp = timestamp + \
                int(future_ts)*self.TIMESTAP_PER_STEP
            if future_timestamp > self.timestamps[-1]:
                continue

            for pred_id, pred_data in prediction_data.items():
                pred_loc = prediction_data[pred_id].get(future_ts, 'None')
                if pred_loc == 'None' or pred_id not in pred_id_in_gt:
                    continue
                else:
                    x = ego_x + pred_data[future_ts]['x'] * \
                        math.cos(theta) - \
                        pred_data[future_ts]['y'] * math.sin(theta)
                    y = ego_y + pred_data[future_ts]['x'] * \
                        math.sin(theta) + \
                        pred_data[future_ts]['y'] * math.cos(theta)
                stats, score = self.check_collision(
                    plan_loc, self.ego_id,
                    (x, y), pred_id_in_gt[pred_id])
                if stats:
                    return True, score
                if score < min_score:
                    min_score = score
        return False, 1 - min(1, min_score)

    def check_collision(self, ego_location, ego_id, actor_location, actor_id):
        actor_extent = self.actual_traj_actor_view[actor_id]['extent']
        ego_extent = self.actual_traj_actor_view[ego_id]['extent']

        diff_x = abs(ego_location['x'] - actor_location[0]) - \
            (actor_extent['x'] + ego_extent['x'] -
             self.configs['camera_loc'][0])
        diff_y = abs(ego_location['y'] - actor_location[1]) - \
            (actor_extent['y'] + ego_extent['y'] -
             self.configs['camera_loc'][1])

        if diff_x < 0 and diff_y < 0:
            return True, 0
        return False, math.sqrt(diff_x**2 + diff_y**2)
