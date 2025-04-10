import os
import sys
import time
import carla
import subprocess
import traceback

from loguru import logger
from omegaconf import DictConfig, OmegaConf
from scenario.utils.timer import GameTime


class CarlaOperator:

    def __init__(self, server_config: DictConfig):
        self.cfg = OmegaConf.to_container(server_config)
        self.port = int(self.cfg['port'])
        self.tm_port = self.port + 2
        self.gpu = self.cfg['gpu']
        self.carla_path = self.cfg['path']
        self.timeout = 60.0

        self.host = 'localhost'
        self.process_name = f'carla-rpc-port={self.port}'

    @property
    def is_running(self):
        try:
            # Run pgrep command to search for the process
            subprocess.check_output(["pgrep", "-f", self.process_name])
            return True
        except subprocess.CalledProcessError:
            return False

    def _start_operation(self, wait_time=5.0):
        cmd = (f"CUDA_VISIBLE_DEVICES={self.gpu} SDL_VIDEODRIVER=offscreen "
               "bash ${PYLOT_CARLA_HOME}/CarlaUE4.sh "
               f"-nosound -prefernvidia -carla-rpc-port={self.port} -fps=20")
        server_process = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
        logger.info(cmd)
        logger.info("Process ID: {}", server_process.pid)
        time.sleep(wait_time)

    def start(self, town_map: str = 'Town01'):
        GameTime.restart_with_server()
        wait_time = 2.0
        tries = 0
        client = None
        self._start_operation(wait_time)
        while True:
            tries += 1
            if tries > 20:
                logger.error('Connect carla has error, exit')
                sys.exit(-1)

            # self.stop()

            try:
                logger.info('Carla is connecting to {}:{}', self.host, self.port)
                client = carla.Client(self.host, int(self.port))
                client.set_timeout(10.0)
                world = client.load_world(town_map)  # for test
                logger.info('Carla connected to {}:{}', self.host, self.port)
            except Exception as e:
                logger.warning('Restart carla due to connection error.')
                logger.warning(traceback.print_exc())
                wait_time += 3
                continue
            else:
                break
        return client

    def stop(self):
        # kill_process = subprocess.Popen('killall -9 -r CarlaUE4-Linux', shell=True)
        # cmd = f"kill -9 $(ps -ef|grep carla-rpc-port={self.port}"
        # cmd += "|gawk '$0 !~/grep/ {print $2}' |tr -s '\n' ' ') "
        cmd = f"ps -ef | grep 'carla-rpc-port={self.port}'" + " | grep -v grep | awk '{print $2}' | xargs -r kill -9"

        logger.info(cmd)
        kill_process = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
        # kill_process.wait()
        time.sleep(5.0)

    def get_record_path(self, carla_record_folder, record_file_name):
        return os.path.join(carla_record_folder, record_file_name)

    def move_carla_record(self, record_file):
        pass

    def close(self):
        self.stop()
