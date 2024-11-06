import numpy as np
import py_trees

from scenario.utils.carla_data_provider import CarlaDataProvider
from scenario.utils.timer import GameTime
from .base import Fitness

class BlockingFitness(Fitness):
    def __init__(
            self,
            actor,
            speed_threshold,
            below_threshold_max_time,
            name="BlockingFitness"
    ):
        """
        Class constructor.
        """
        super(Fitness, self).__init__(name, actor)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._actor = actor
        self._speed_threshold = speed_threshold
        self._below_threshold_max_time = below_threshold_max_time
        self._time_last_valid_state = None

        self.actual_value = np.inf

    def update(self):
        """
        Check if the actor speed is above the speed_threshold
        """
        new_status = py_trees.common.Status.RUNNING

        linear_speed = CarlaDataProvider.get_velocity(self._actor)
        if linear_speed is not None:
            if linear_speed < self._speed_threshold and self._time_last_valid_state:
                stop_time = GameTime.get_time() - self._time_last_valid_state
                stop_rate = stop_time / self._below_threshold_max_time
                stop_rate = 1 - stop_rate
                if stop_rate < self.actual_value:
                    self.actual_value = stop_rate
            else:
                self._time_last_valid_state = GameTime.get_time()

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        return new_status

    def terminate(self, new_status):
        """
        Cleanup sensor
        """
        super(BlockingFitness, self).terminate(new_status)