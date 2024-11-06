import os
import json
import argparse

from tqdm import tqdm

def process_one_run(run_folder):
    agent_data_folder = os.path.join(run_folder, 'agent_data')
    seeds_folder = os.path.join(run_folder, 'seeds')

    existing_cases = os.listdir(agent_data_folder)

    run_data_records = []
    for case_id in tqdm(existing_cases):
        case_folder = os.path.join(agent_data_folder, case_id)
        # check running
        mp4_file = os.path.join(case_folder, 'view_video.mp4')
        if not os.path.isfile(mp4_file):
            continue

        seed_file = os.path.join(seeds_folder, case_id + '.json')
        if not os.path.isfile(seed_file):
            continue

        with open(seed_file, 'r') as f:
            seed_data = json.load(f)

        case_oracle = seed_data['oracle']
        case_label = []
        if case_oracle['complete']:
            case_label.append(0)
        else:
            if case_oracle['collision']:
                case_label.append(1)
            if case_oracle['blocked']:
                case_label.append(2)
            if case_oracle['traffic_rule']:
                case_label.append(3)

        run_data_records.append([case_folder, case_label])
    return run_data_records

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="generate data annotation for surrogate models")
    parser.add_argument('--run_folder', type=str, required=True,
                        help='Data folder')
    parser.add_argument('--save_folder', required=True, type=str,
                        help='save folder')
    parser.add_argument('--file_name', type=str, required=True,)
    opt = parser.parse_args()

    if not os.path.exists(opt.run_folder):
        exit()

    run_name = opt.file_name #os.path.basename(opt.run_folder)

    if not os.path.exists(opt.save_folder):
        os.makedirs(opt.save_folder)

    run_data_records = process_one_run(opt.run_folder)

    save_file = os.path.join(opt.save_folder, run_name + '.txt')
    with open(save_file, 'w') as f:
        for case_record in run_data_records:
            case_folder, case_label = case_record
            line_str = f"{case_folder}"
            for case_label in case_label:
                line_str += f",{case_label}"
            line_str += "\n"
            f.write(line_str)
