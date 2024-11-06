import time
import copy
import carla
import py_trees
import operator

from scenario.atomic.behavior import AtomicBehavior
from scenario.configuration import BasicAgent
from .walker_agent import WalkerAgent

class WalkerWaypointFollower(AtomicBehavior):

    def __init__(
            self,
            actor: carla.Actor,
            actor_config: BasicAgent,
            name="WalkerWaypointFollower"
    ):
        """
        Set up actor and local planner
        """
        super(WalkerWaypointFollower, self).__init__(name, actor)

        self._actor_config = actor_config
        self._route = self._actor_config.route

        self._local_planner = None
        self._unique_id = 0

    def initialise(self):
        super(WalkerWaypointFollower, self).initialise()
        self._unique_id = int(round(time.time() * 1e9))
        try:
            # running_WF_actor_ -> update for terminate
            # check whether WF for this actor is already running and add new WF to running_WF list
            check_attr = operator.attrgetter("running_WF_actor_{}".format(self._actor.id))
            running = check_attr(py_trees.blackboard.Blackboard())
            active_wf = copy.copy(running)
            active_wf.append(self._unique_id)
            py_trees.blackboard.Blackboard().set("running_WF_actor_{}".format(self._actor.id), active_wf, overwrite=True)
        except AttributeError:
            # no WF is active for this actor
            py_trees.blackboard.Blackboard().set("terminate_WF_actor_{}".format(self._actor.id), [], overwrite=True)
            py_trees.blackboard.Blackboard().set("running_WF_actor_{}".format(self._actor.id), [self._unique_id], overwrite=True)

        self._apply_local_planner(self._actor)
        return True

    def _apply_local_planner(self, actor: carla.Actor):

        local_planner = WalkerAgent(
            actor,
            opt_dict={
                'sampling_radius': 1.0
            }
        )
        local_planner.set_global_route(self._route)
        self._local_planner = local_planner

    def update(self):
        """
        Compute next control step for the given waypoint plan, obtain and apply control to actor
        """
        new_status = py_trees.common.Status.RUNNING

        # check thread conflicts.
        check_term = operator.attrgetter("terminate_WF_actor_{}".format(self._actor.id))
        terminate_wf = check_term(py_trees.blackboard.Blackboard())
        check_run = operator.attrgetter("running_WF_actor_{}".format(self._actor.id))
        active_wf = check_run(py_trees.blackboard.Blackboard())
        # Termination of WF if the WFs unique_id is listed in terminate_wf
        # only one WF should be active, therefore all previous WF have to be terminated
        # this is to make sure that new added agent overwrite previous ones, so, skip this
        if self._unique_id in terminate_wf:
            terminate_wf.remove(self._unique_id)
            if self._unique_id in active_wf:
                active_wf.remove(self._unique_id)
            py_trees.blackboard.Blackboard().set("terminate_WF_actor_{}".format(self._actor.id), terminate_wf, overwrite=True)
            py_trees.blackboard.Blackboard().set("running_WF_actor_{}".format(self._actor.id), active_wf, overwrite=True)
            new_status = py_trees.common.Status.SUCCESS
            return new_status

        # start update action - replace by localplanner output
        success = False
        if self._actor is not None and self._actor.is_alive and self._local_planner is not None:
            control = self._local_planner.run_step(debug=False)
            self._actor.apply_control(control)
            # Check if the actor reached the end of the plan
            #  replace access to private _waypoints_queue with public getter
            if not self._local_planner.is_task_finished():  # pylint: disable=protected-access
                success = False
            else:
                tmp_route = copy.deepcopy(self._route)
                tmp_route = tmp_route[::-1]
                self._local_planner.set_global_route(tmp_route)
                success = False

        if success:
            new_status = py_trees.common.Status.SUCCESS

        return new_status

    def terminate(self, new_status):
        """
        On termination of this behavior,
        the controls should be set back to 0.
        """
        if self._actor is not None and self._actor.is_alive:
            control = self._actor.get_control()
            control.speed = 0
            self._actor.apply_control(control)

        super(WalkerWaypointFollower, self).terminate(new_status)