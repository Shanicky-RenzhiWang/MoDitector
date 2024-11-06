from importlib.metadata import entry_points
import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger
import os
import sys
import glob
import json
from scenario.configuration import ScenarioConfig, SeedConfig
from fuzzer.base import BaseFuzzer
from fuzzer.root_cause_util.root_cause_ana import StatHandler


def _load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


class ReplayRunner(BaseFuzzer):
    def __init__(self, save_root, cfg, server_config):
        self.save_root = save_root + '_replay'
        if not os.path.exists(self.save_root):
            os.makedirs(self.save_root)
        os.makedirs(f'{self.save_root}/runner_msg', exist_ok=True)
        os.system(
            f'cp {save_root}/runner_msg/*.json {self.save_root}/runner_msg/')
        super(ReplayRunner, self).__init__(save_root, cfg, server_config)
        self.ori_res = _load_json(os.path.join(
            self.save_root, 'result_record1.json'))

    def run(self):
        self.runner_msg_folder = os.path.join(self.save_root, 'runner_msg')
        # for seed_file in glob.glob(os.path.join(self.seed_folder, '*.json')):
        # seed_id = seed_file.split('/')[-1].split('.')[0]
        for res in self.ori_res['overview']:
            if res['scenario_result'] == "Failed":
                seed_id = res['scenario_id']
            else:
                continue
            if os.path.exists(os.path.join(self.save_root, 'results', seed_id+'.txt')):
                continue
            seed_file = os.path.join(self.seed_folder, seed_id+'.json')
            logger.info(f'running seed {seed_id}')
            seed_json_data = _load_json(seed_file)
            seed_scenairo = ScenarioConfig.from_json(
                seed_json_data['scenario'])
            seed = SeedConfig(seed_id, seed_scenairo)
            runner_pass, runner_msg = self.load_and_run_scenario(seed)
            root_cause, feedback_score = self._get_root_cause_stat(
                seed.id, runner_message=runner_msg)

    def _get_root_cause_stat(self, seed_id, runner_message):
        base_dir = os.path.join(self.save_root, 'agent_data', seed_id)
        collision = runner_message['result']['infractions']['collision_timestamp']
        stat_handler = StatHandler(base_dir, collision)
        stat_save_path = os.path.join(
            self.runner_msg_folder, f'{seed_id}_root_cause_replay.json')
        stat, feedback_score = stat_handler.collect_root_cause(
            prefer_feedback='prediction', stat_save_path=stat_save_path)

        with open(os.path.join(self.runner_msg_folder, f'{seed_id}_root_cause.json'), 'w') as f:
            json.dump(stat, f, indent=4)
        return stat, feedback_score


@hydra.main(config_path='configs', config_name='main_fuzzer', version_base=None)
def main(cfg: DictConfig):
    level = cfg.log_level
    logger.configure(handlers=[{"sink": sys.stderr, "level": level}])

    logger.debug(cfg)
    save_root = cfg.save_root
    assert os.path.exists(save_root)

    logger_file = os.path.join(save_root, 'system_run.log')
    if os.path.exists(logger_file):
        os.remove(logger_file)
    _ = logger.add(logger_file, level=level, rotation=None)
    replay_runner = ReplayRunner(
        cfg.save_root_testing, cfg, cfg.server_configs.testing)
    replay_runner.run()


if __name__ == '__main__':
    main()
    sys.exit(0)
