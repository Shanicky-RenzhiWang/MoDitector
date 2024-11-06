import py_trees

from scenario.utils.timer import GameTime
from .base import AtomicCondition

class TriggerTimer(AtomicCondition):

    def __init__(self, actor, name, duration=float("inf")):
        """
        Setup actor
        """
        super(TriggerTimer, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._actor = actor

        self._duration = duration
        self._start_time = 0

    def initialise(self):
        """
        Initialize the start time of this condition
        """
        self._start_time = GameTime.get_time()
        super(TriggerTimer, self).initialise()

    def update(self):
        """
        Check if the _actor stands still (v=0)
        """
        new_status = py_trees.common.Status.RUNNING

        if GameTime.get_time() - self._start_time > self._duration:
            new_status = py_trees.common.Status.SUCCESS
            # logger.debug('{}: duration: {} current: {}', self._actor.id, self._duration, GameTime.get_time())

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        return new_status