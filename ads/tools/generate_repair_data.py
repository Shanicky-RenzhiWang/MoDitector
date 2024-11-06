import os

monitor_folder = '/home/erdos/workspace/ADSRepair/data/monitor/roach'

repair_folder = '/home/erdos/workspace/ADSRepair/data/repair/roach'

if not os.path.exists(repair_folder):
    os.makedirs(repair_folder)

def process_one(monitor_file, save_file):
    save_records = []
    with open(monitor_file, 'r') as f:
        line = f.readline()
        while line:
            line = line.rstrip()
            line = line.replace('agent_data', 'seeds')
            line = line.replace(',', '.json,')
            save_records.append(line)
            line = f.readline()

    with open(save_file, 'w') as f:
        for line in save_records:
            f.write(line + '\n')

files = os.listdir(monitor_folder)
for file_name in files:
    monitor_file = os.path.join(monitor_folder, file_name)
    repair_file = os.path.join(repair_folder, file_name)
    process_one(monitor_file, repair_file)
