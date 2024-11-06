file_path = '/home/erdos/workspace/results/ADSFuzzer/debug/agent_data/seed_126/meta/0080.pkl'

import pickle

with open(file_path, 'rb') as f:
    data = pickle.load(f)

print(data['ego'])