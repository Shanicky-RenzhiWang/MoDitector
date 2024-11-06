import os

scenario_lst = ['highway_enter', 'highway_exit', 'intersection_left', 'intersection_right', 'intersection_straight']

original_data_folder = '/home/erdos/workspace/ADSRepair/data/monitor/roach_collector'
save_folder = '/home/erdos/workspace/ADSRepair/data/monitor/roach'
version = 'v1'

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

data_files = os.listdir(original_data_folder)

for scenario in scenario_lst:
    data_records_train = {
        0: [],
        1: []
    }

    data_records_test = {
        0: [],
        1: []
    }
    for data_file_name in data_files:
        if scenario not in data_file_name:
            continue

        data_file_records = {
            0: [],
            1: []
        }
        data_file_path = os.path.join(original_data_folder, data_file_name)
        with open(data_file_path, 'r') as f:
            line = f.readline()
            while line:
                line = line.rstrip()
                data_path = line.split(',')[0]
                data_label = int(line.split(',')[1])

                if data_label == 0:
                    data_file_records[0].append([data_path, data_label])
                else:
                    data_file_records[1].append([data_path, data_label])
                line = f.readline()

        # fail
        run_fail_test = data_file_records[1][:int(len(data_file_records[1]) / 2)]
        run_fail_train = data_file_records[1][int(len(data_file_records[1]) / 2):]

        # success
        run_success_test = data_file_records[0][:int(len(data_file_records[0]) / 2)]
        run_success_train = data_file_records[0][int(len(data_file_records[0]) / 2):]

        print(f'run_fail_test: {len(run_fail_test)}, run_success_test: {len(run_success_test)}')

        # complete
        data_records_test[1] += run_fail_test
        data_records_train[1] += run_fail_train

        # normal
        data_records_test[0] += run_success_test
        data_records_train[0] += run_success_train

        save_path_train = f"{save_folder}/{scenario}_train_{version}.txt"

        with open(save_path_train, 'w') as f:
            for k, v in data_records_train.items():
                for item in v:
                    f.write(str(item[0]) + ',' + str(item[1]) + '\n')

        save_path_test = f"{save_folder}/{scenario}_test_{version}.txt"
        with open(save_path_test, 'w') as f:
            for k, v in data_records_test.items():
                for item in v:
                    f.write(str(item[0]) + ',' + str(item[1]) + '\n')

