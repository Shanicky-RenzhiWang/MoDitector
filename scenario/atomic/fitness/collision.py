import py_trees
import numpy as np

from shapely.geometry import Polygon
from scenario.atomic.fitness import Fitness

class CollisionFitness(Fitness):

    def __init__(self, actor, name="CollisionFitness"):
        """
        Construction with sensor setup
        """
        super(CollisionFitness, self).__init__(name, actor)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

        self._world = self.actor.get_world()
        self.actual_value = np.inf

    def update(self):
        """
        Check collision distance with vehicles and walkers
        """
        new_status = py_trees.common.Status.RUNNING

        ego_bb = self.actor.bounding_box
        ego_vertices = ego_bb.get_world_vertices(self.actor.get_transform())
        ego_list = [[v.x, v.y, v.z] for v in ego_vertices]
        ego_polygon = Polygon(ego_list)

        actors = self._world.get_actors()
        vehicle = actors.filter('*vehicle*')
        for target_actor in vehicle:
            if target_actor.id == self.actor.id:
                continue
            target_bb = target_actor.bounding_box
            target_vertices = target_bb.get_world_vertices(target_actor.get_transform())
            target_list = [[v.x, v.y, v.z] for v in target_vertices]
            target_polygon = Polygon(target_list)
            dist = target_polygon.distance(ego_polygon)
            if dist < self.actual_value:
                self.actual_value = dist

        walker = actors.filter('*walker*')
        for target_actor in walker:
            if target_actor.id == self.actor.id:
                continue
            target_bb = target_actor.bounding_box
            target_vertices = target_bb.get_world_vertices(target_actor.get_transform())
            target_list = [[v.x, v.y, v.z] for v in target_vertices]
            target_polygon = Polygon(target_list)
            dist = target_polygon.distance(ego_polygon)
            if dist < self.actual_value:
                self.actual_value = dist
        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        return new_status

    def terminate(self, new_status):
        """
        Cleanup sensor
        """
        super(CollisionFitness, self).terminate(new_status)
