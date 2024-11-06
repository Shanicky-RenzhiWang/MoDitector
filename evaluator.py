import copy
import os
import sys
import hydra
import json

from typing import List
from loguru import logger
from importlib import import_module
from omegaconf import DictConfig, OmegaConf

from scenario.configuration import SeedConfig
from ads.systems.RLRepair.fuzzer.suites import SeparateEvaluator

def load_entry_point(name):
    mod_name, attr_name = name.split(":")
    mod = import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn

def load_data(data_file: str) -> List[SeedConfig]:
    # case_id, label, seed_path
    with open(data_file, 'r') as f:
        data_lines = f.readlines()

    data = []
    for line in data_lines:
        line = line.strip()
        data_case_id = line.split(',')[0]
        data_label = line.split(',')[1]
        data_file_path = line.split(',')[2]
        with open(data_file_path, 'r') as f:
            data_json = json.load(f)
        seed_config = SeedConfig.from_json(data_json)
        original_seed_id = seed_config.id
        seed_config.id = f"{data_case_id}_{original_seed_id}_label_{data_label}"
        data.append(copy.deepcopy(seed_config))
    return data

@hydra.main(config_path='configs', config_name='evaluator', version_base=None)
def main(cfg: DictConfig):
    # setup logger
    level = cfg.log_level
    logger.configure(handlers=[{"sink": sys.stderr, "level": level}])

    logger.debug(cfg)

    # save folder
    save_root = cfg.save_root
    save_root_repair = cfg.save_root_repair
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    if not os.path.exists(save_root_repair):
        os.makedirs(save_root_repair)

    resume = cfg.resume
    logger_file = os.path.join(save_root, 'system_run.log')
    if (not resume) and os.path.exists(logger_file):
        os.remove(logger_file)
    _ = logger.add(logger_file, level=level, rotation=None)

    OmegaConf.save(config=cfg, f=os.path.join(save_root, 'config_overall.yaml'))
    logger.info('Save result and log to {}', save_root)

    logger.info("TRAIN RL FINISHED! START EVALUATION!")
    # create evaluator
    # for test data
    test_data = load_data(cfg.test_data_file)
    cfg.server_configs.evaluator.gpu = cfg.gpu
    evaluator = SeparateEvaluator(os.path.join(cfg.save_root_evaluator, 'test'), cfg, cfg.server_configs.evaluator)
    evaluator.run(test_data, 0)
    evaluator.destroy()

    # for train data
    train_data = load_data(cfg.train_data_file)
    cfg.server_configs.evaluator.gpu = cfg.gpu
    evaluator = SeparateEvaluator(os.path.join(cfg.save_root_evaluator, 'train'), cfg, cfg.server_configs.evaluator)
    evaluator.run(train_data, 0)
    evaluator.destroy()


if __name__ == '__main__':
    main()
    logger.info("$=$ train_rl_repair.py DONE!")
    sys.exit(0)