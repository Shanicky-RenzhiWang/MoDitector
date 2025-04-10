import os
import sys
import hydra

from importlib import import_module
from loguru import logger
from omegaconf import DictConfig, OmegaConf


def load_entry_point(name):
    mod_name, attr_name = name.split(":")
    mod = import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn

@hydra.main(config_path='configs', config_name='main_fuzzer', version_base=None)
def main(cfg: DictConfig):
    # setup logger
    level = cfg.log_level
    logger.configure(handlers=[{"sink": sys.stderr, "level": level}])

    # save folder
    save_root = cfg.save_root
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    resume = cfg.resume
    logger_file = os.path.join(save_root, 'system_run.log')
    if (not resume) and os.path.exists(logger_file):
        os.remove(logger_file)
    _ = logger.add(logger_file, level=level, rotation=None)

    OmegaConf.save(config=cfg, f=os.path.join(save_root, 'config_overall.yaml'))
    logger.info('Save result and log to {}', save_root)

    entry_point_fuzzer = cfg.entry_point_fuzzer
    fuzz_class = load_entry_point(entry_point_fuzzer)

    server_config = cfg.server_configs.testing
    server_config.gpu = cfg.gpu
    fuzz_instance = fuzz_class(cfg.save_root_testing, cfg, server_config)
    fuzz_instance.run()
    fuzz_instance.destroy()

if __name__ == '__main__':
    main()
    logger.info('^=^ Fuzzing DONE!')
    sys.exit(0)