import os
import sys
import time
import carla
import subprocess
import traceback

from loguru import logger
from omegaconf import DictConfig, OmegaConf

class CarlaOperator:

    def __init__(self, port, gpu, carla_path):
        self.port = int(port)
        self.tm_port = self.port + 2
        self.gpu = gpu
        self.carla_path = carla_path
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

    def _start_operation(self, wait_time = 5.0):
        cmd = (f"CUDA_VISIBLE_DEVICES={self.gpu} SDL_VIDEODRIVER=offscreen "
               f"bash {os.path.join(self.carla_path, 'CarlaUE4.sh')} "
               f"-nosound -windowed -opengl -carla-rpc-port={self.port}")
        server_process = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
        logger.info(cmd)
        logger.info("Process ID: {}", server_process.pid)
        time.sleep(wait_time)

    def start(self):
        wait_time = 2.0
        tries = 0
        client = None
        while True:
            tries += 1
            if tries > 20:
                logger.error('Connect carla has error, exit')
                sys.exit(-1)

            self.stop()
            self._start_operation(wait_time)
            try:
                logger.info('Carla is connecting to {}:{}', self.host, self.port)
                client = carla.Client(self.host, int(self.port))
                client.set_timeout(10.0)
                world = client.load_world('Town01')  # for test
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
        cmd = f"ps -ef | grep 'carla-rpc-port={self.port}'" +" | grep -v grep | awk '{print $2}' | xargs -r kill -9"

        logger.info(cmd)
        kill_process = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
        kill_process.wait()

        # cmd = f"kill -9 $(lsof -i :{self.tm_port}" + " | grep LISTEN | awk '{print $2}')"
        # logger.info(cmd)
        # kill_process = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
        # kill_process.wait()

        time.sleep(5.0)

    def get_record_path(self, carla_record_folder, record_file_name):
        return os.path.join(carla_record_folder, record_file_name)

    def move_carla_record(self, record_file):
        pass

    def close(self):
        self.stop()

if __name__ == '__main__':
    carla_operator = CarlaOperator('10030', 2, '/home/erdos/workspace/pylot/dependencies/CARLA_0.9.10.1')
    for i in range(10):
        logger.debug(i)
        client = carla_operator.start()

        # connect to client & traffic manager
        # self.client = carla.Client(self.carla_operator.host, int(self.carla_operator.port))
        # self.client.set_timeout(60.0)
        try:
            traffic_manager = client.get_trafficmanager(int(carla_operator.tm_port) + i)
        except Exception as e:
            logger.error("traffic_manager fail to init", flush=True)
            logger.error("> {}\033[0m\n".format(e))
            sys.exit(-1)

        logger.debug(traffic_manager.get_port())
        # traffic_manager.shut_down()
