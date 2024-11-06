import carla
import py_trees
import numpy as np

from shapely.geometry import Point
from scenario.atomic.fitness import Fitness
from scenario.configuration import WaypointUnit

class ReachDestinationFitness(Fitness):

    def __init__(self, actor, destination: WaypointUnit, name="ReachDestinationFitness"):
        """
        Construction with sensor setup
        """
        super(ReachDestinationFitness, self).__init__(name, actor)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

        self.destination = destination
        self._world = self.actor.get_world()
        self._map = self._world.get_map()
        target_destination = self._map.get_waypoint(
            carla.Location(
                x = destination.x,
                y = destination.y,
                z = destination.z
            )
        )
        self.target_point = Point(target_destination.transform.location.x, target_destination.transform.location.y)
        self.actual_value = np.inf

    def update(self):
        """
        Check collision distance with vehicles and walkers
        """
        new_status = py_trees.common.Status.RUNNING

        ego_location = self.actor.get_location()
        ego_point = Point(ego_location.x, ego_location.y)

        dist = ego_point.distance(self.target_point)
        if dist < self.actual_value:
            self.actual_value = dist

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        return new_status

    def terminate(self, new_status):
        """
        Cleanup sensor
        """
        super(ReachDestinationFitness, self).terminate(new_status)
