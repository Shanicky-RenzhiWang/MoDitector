import carla
import py_trees

from shapely.geometry import Point

from scenario.configuration import WaypointUnit
from scenario.utils.carla_data_provider import CarlaDataProvider
from scenario.utils.traffic_events import TrafficEvent, TrafficEventType
from .base import Criterion

class ReachDestinationTest(Criterion):

    """
    TODO: change to last destination
    Check at which stage of the route is the actor at each tick

    Important parameters:
    - actor: CARLA actor to be used for this test
    - route: Route to be checked
    - terminate_on_failure [optional]: If True, the complete scenario will terminate upon failure of this test
    """
    DISTANCE_THRESHOLD = 10.0  # meters
    WINDOWS_SIZE = 2

    def __init__(self, actor, destination: WaypointUnit, name="ReachDestinationTest", terminate_on_failure=False):
        """
        """
        super(ReachDestinationTest, self).__init__(name, actor, 100, terminate_on_failure=terminate_on_failure)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._actor = actor
        self._world = self.actor.get_world()
        self._map = self._world.get_map()
        target_destination = self._map.get_waypoint(
            carla.Location(
                x=destination.x,
                y=destination.y,
                z=destination.z
            )
        )
        self.target = target_destination.transform.location

    def update(self):
        """
        Check if the actor location is within trigger region
        """
        new_status = py_trees.common.Status.RUNNING

        location = CarlaDataProvider.get_location(self._actor)
        if location is None:
            return new_status

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE

        elif self.test_status == "RUNNING" or self.test_status == "INIT":
            if location.distance(self.target) < self.DISTANCE_THRESHOLD:
                route_completion_event = TrafficEvent(event_type=TrafficEventType.ROUTE_COMPLETED)
                route_completion_event.set_message("Destination was successfully reached")
                self.list_traffic_events.append(route_completion_event)
                self.test_status = "SUCCESS"

        elif self.test_status == "SUCCESS":
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status

    def terminate(self, new_status):
        """
        Set test status to failure if not successful and terminate
        """
        if self.test_status == "INIT":
            self.test_status = "FAILURE"
        super(ReachDestinationTest, self).terminate(new_status)

