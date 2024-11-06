import py_trees

from scenario.utils.carla_data_provider import CarlaDataProvider
from scenario.utils.timer import GameTime
from .base import AtomicCondition

EPSILON = 0.001

class StandStill(AtomicCondition):

    """
    This class contains a standstill behavior of a scenario

    Important parameters:
    - actor: CARLA actor to execute the behavior
    - name: Name of the condition
    - duration: Duration of the behavior in seconds

    The condition terminates with SUCCESS, when the actor does not move
    """

    def __init__(self, actor, name, duration=float("inf")):
        """
        Setup actor
        """
        super(StandStill, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._actor = actor

        self._duration = duration
        self._start_time = 0

    def initialise(self):
        """
        Initialize the start time of this condition
        """
        self._start_time = GameTime.get_time()
        super(StandStill, self).initialise()

    def update(self):
        """
        Check if the _actor stands still (v=0)
        """
        new_status = py_trees.common.Status.RUNNING

        velocity = CarlaDataProvider.get_velocity(self._actor)

        if velocity > EPSILON:
            self._start_time = GameTime.get_time()

        if GameTime.get_time() - self._start_time > self._duration:
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status