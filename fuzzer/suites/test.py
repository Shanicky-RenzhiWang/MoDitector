
import os
import sys
sys.path.append("..")
from root_cause_util.root_cause_ana import StatHandler
import json

seed_id = 5

base_dir = os.path.join(
    '/home/wangrenzhi/code/pylot_workspace/results/MoDitector/data_collection/pylot/highway_enter_bak/RootCauseFuzzer/testing', 'agent_data', f'seed_{seed_id}')
runner_msg_path = os.path.join('/home/wangrenzhi/code/pylot_workspace/results/MoDitector/data_collection/pylot/highway_enter_bak/RootCauseFuzzer/testing', 'runner_msg')

with open(os.path.join(runner_msg_path, f'seed_{seed_id}.json'),'r') as f:
    runner_message = json.load(f)
stat_handler = StatHandler(base_dir, runner_message['result']['infractions']['collision_timestamp'])
stat, feedback_score = stat_handler.collect_root_cause(prefer_feedback='perception')
print(stat)
print(feedback_score)
with open(os.path.join(runner_msg_path, f'seed_{seed_id}_root_cause.json'), 'w') as f:
    json.dump(stat, f, indent=4)
