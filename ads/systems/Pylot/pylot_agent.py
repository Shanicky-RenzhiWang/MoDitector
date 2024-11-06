import os
import glob
import random
import time

import erdos
import carla
import logging
import pathlib
import shutil
import pickle
import traceback
import moviepy.video.io.ImageSequenceClip

import pylot.flags
import pylot.component_creator  # noqa: I100
import pylot.operator_creator
import pylot.perception.messages
import pylot.utils

from PIL import Image
from collections import deque
from absl import flags
from loguru import logger

from pylot.map.hd_map import HDMap
from pylot.perception.messages import ObstaclesMessage, TrafficLightsMessage, LanesMessage
from pylot.planning.messages import WaypointsMessage
from pylot.planning.waypoints import Waypoints

from agents.navigation.autonomous_agent import AutonomousAgent
from agents.pylot_agent.misc import *
from agents.langad.lang_repairer import LangRepairer
from agents.langad.recorder_tools import DisplayInterface, convert_global_route, cut_waypoints
from agents.langad.misc import get_forward_value

from common_tools.carla_data_provider import CarlaDataProvider
from common_tools.global_config import GlobalConfig
from common_tools.route_manipulation import downsample_route

FLAGS = flags.FLAGS

def enable_logging():
    """Overwrites logging config so that loggers can control verbosity.

    This method is required because the challenge evaluator overwrites
    verbosity, which causes Pylot log messages to be discarded.
    """
    import logging
    logging.root.setLevel(logging.INFO)

def get_entry_point():
    return 'ERDOSAgent'

class ERDOSAgent(AutonomousAgent):
    """Agent class that interacts with the challenge leaderboard.
    TODO: update the planning lane width config

    Attributes:
        _camera_setups: Mapping between camera names and
            :py:class:`~pylot.drivers.sensor_setup.CameraSetup`.
        _lidar_setup (:py:class:`~pylot.drivers.sensor_setup.LidarSetup`):
            Setup of the Lidar sensor.
    """
    def set_global_plan(self, global_plan_gps, global_plan_world_coord, wp=None):
        """
        Set the plan (route) for the agent
        global_plan_world_coord: route.append((wp_tuple[0].transform, wp_tuple[1]))
        """
        # get more waypoints in case of bugs
        dest_wp_tuple = global_plan_world_coord[-1]
        dest_wp_transform = dest_wp_tuple[0]
        _map = CarlaDataProvider.get_map()
        dest_wp = _map.get_waypoint(dest_wp_transform.location)
        for i in range(10):
            next_wp_lst = dest_wp.next(1.0)
            if len(next_wp_lst) > 0:
                next_wp = random.choice(next_wp_lst)
                next_tuple = (next_wp.transform, dest_wp_tuple[1])
                global_plan_world_coord.append(next_tuple)
                dest_wp = next_wp

        ds_ids = downsample_route(global_plan_world_coord, 50)
        # logger.debug(ds_ids)
        self._global_plan_world_coord = [(global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in ds_ids]
        # self._global_plan = [global_plan_gps[x] for x in ds_ids]

    def setup(self, path_to_conf_file):

        ##### added for recording #####
        self.initialized = False
        self.step = -1
        self.hist_waypoint_buffer = list()
        self.recorder_waypoints = list()
        self.last_state = -1
        self.tries = 0

        self._hic = DisplayInterface()
        # load some global config. i.e. save_root
        gc = GlobalConfig.get_instance()
        record_agent = gc.record_agent
        if record_agent:
            save_root = gc.save_root
            self.save_folder = os.path.join(save_root, 'agent_data')
            if not os.path.exists(self.save_folder):
                os.makedirs(self.save_folder)
        else:
            self.save_folder = None

        if GlobalConfig.repair_agent is not None:
            self.lang_repair = GlobalConfig.repair_agent
        else:
            self.lang_repair = LangRepairer()
            # if FLAGS.training_file is not None and os.path.isfile(FLAGS.training_file):
            #     self.lang_repair.load_training_data(FLAGS.training_file)
            # else:
            #     logger.warning(f'No training file {FLAGS.training_file} for repairer..')
            GlobalConfig.repair_agent = self.lang_repair

        self.pylot_port = gc.cfg.pylot_port

        # active if saving info
        if self.save_folder is not None:
            string = f"{gc.attributes['route_index']}"

            self.save_path = pathlib.Path(self.save_folder) / string

            if os.path.exists(self.save_path):
                shutil.rmtree(self.save_path)

            self.save_path.mkdir(parents=True, exist_ok=False)
            (self.save_path / 'monitor_view').mkdir(parents=True, exist_ok=True)
            (self.save_path / 'key_frames').mkdir(parents=True, exist_ok=True)
            (self.save_path / 'meta').mkdir(parents=True, exist_ok=True)
            (self.save_path / 'repair').mkdir(parents=True, exist_ok=True)
        ##### added for recording #####

        pylot.utils.set_tf_loglevel(logging.INFO)
        # Parse the flag file. Users can use the different flags defined
        # across the Pylot directory.
        flags.FLAGS([__file__, '--flagfile={}'.format(path_to_conf_file)])
        # self.logger = erdos.utils.setup_logging('erdos_agent',
        #                                         os.path.join(self.save_path, FLAGS.log_file_name))
        enable_logging()

        self._town_name = None
        self._ego_vehicle = None
        self.start_pylot()

    def start_pylot(self):
        # Create the dataflow of AV components. Change the method
        # to add your operators.
        (self._pose_stream,
         self._route_stream,
         self._lanes_stream,
         self._open_drive_stream,
         self._control_stream,
         self._obstacles_stream,
         self._static_obstacles_stream,
         self._vehicle_id_stream,
         streams_to_send_top_on,
         self._behavior_stream,
         self._planning_stream,
         self.streams_to_be_monitored
         ) = create_data_flow()
        # Execute the dataflow.
        find_port = self.pylot_port
        self._node_handle = erdos.run_async(graph_filename=f'pylot_{find_port}', start_port=find_port)
        # Close the streams that are not used (i.e., send top watermark).
        for stream in streams_to_send_top_on:
            stream.send(erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))

    def destroy_pylot(self):
        self.initialized = False

        # self.logger.info('ERDOSAgent destroy method invoked')
        # Stop the ERDOS node.
        self._node_handle.shutdown()
        # Reset the ERDOS dataflow graph.
        erdos.reset()
        time.sleep(1.0)

    def _init(self):
        self._world = CarlaDataProvider.get_world()
        self._open_drive_map = CarlaDataProvider.get_map().to_opendrive()
        self._ego_vehicle = CarlaDataProvider.get_ego()
        self._od_map = HDMap(carla.Map('map', self._open_drive_map), None)
        self._carla_map = self._world.get_map()
        self.recorder_waypoints = convert_global_route(self._global_plan_world_coord)

        self.last_control = None
        self.hist_waypoint_buffer = list()
        self.initialized = True
        self.last_state = 0
        self.tries = 0

        # 1. send ego vehicle id
        self.send_vehicle_id(self._vehicle_id_stream)
        # 2. send hd_map
        self.send_hd_map(self._open_drive_stream)
        # 3. send route here
        self.send_route(self._route_stream, None)

        logger.info("initialized")

    def destroy(self):
        """Clean-up the agent. Invoked between different runs."""
        # self.logger.info('ERDOSAgent destroy method invoked')
        # Stop the ERDOS node.
        self._node_handle.shutdown()
        # Reset the ERDOS dataflow graph.
        erdos.reset()

        logger.info('Start save monitor rgb...')
        fps = 30
        image_folder = os.path.join(self.save_path, 'monitor_view')
        save_file = os.path.join(self.save_path, 'monitor_view.mp4')
        image_files = [file_path for file_path in sorted(glob.glob(image_folder + '/*.png'), key=os.path.getmtime)]
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
        clip.write_videofile(save_file)
        shutil.rmtree(image_folder)

        time.sleep(1.0)

    def send_vehicle_id(self, vehicle_id_stream):
        # confirm the ego vehicle id, only send once at the start
        vehicle_id_stream.send(erdos.Message(erdos.Timestamp(coordinates=[0]), self._ego_vehicle.id))
        vehicle_id_stream.send(erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))

    def send_hd_map(self, open_drive_stream):
        open_drive_stream.send(erdos.Message(erdos.Timestamp(coordinates=[0]), self._open_drive_map))
        open_drive_stream.send(erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))

    def send_perfect_object_detection(self, obstacles_stream, static_obstacles_stream, timestamp):
        # send obstacles
        actor_list = self._world.get_actors()
        (vehicles, people, traffic_lights, _, _, traffic_cones) = extract_data_in_pylot_format(actor_list)
        traffic_lights = []

        obstacles_stream.send(ObstaclesMessage(timestamp, vehicles + people + traffic_cones))
        obstacles_stream.send(erdos.WatermarkMessage(timestamp))

        # send traffic lights
        static_obstacles_stream.send(TrafficLightsMessage(timestamp, traffic_lights))
        static_obstacles_stream.send(erdos.WatermarkMessage(timestamp))

    def send_perfect_pose(self, pose_stream, timestamp):
        vec_transform = pylot.utils.Transform.from_simulator_transform(self._ego_vehicle.get_transform())
        velocity_vector = pylot.utils.Vector3D.from_simulator_vector(self._ego_vehicle.get_velocity())
        forward_speed = velocity_vector.magnitude()
        pose = pylot.utils.Pose(vec_transform, forward_speed, velocity_vector, timestamp.coordinates[0])

        pose_stream.send(erdos.Message(timestamp, pose))
        pose_stream.send(erdos.WatermarkMessage(timestamp))

    def send_perfect_lanes(self, lanes_stream, timestamp):
        # todo: this is important for change lane
        vehicle_location = self._ego_vehicle.get_location()
        w = self._carla_map.get_waypoint(vehicle_location)

        if w.is_junction:
            road_width_l = 6.0
            road_width_r = 6.0
        else:
            lane_width = [w.lane_width * 0.5]
            # print(target_point)
            found_lanes = [f"{w.road_id}_{w.section_id}_{w.lane_id}"]
            l = w.get_left_lane()
            while l and l.lane_type == carla.LaneType.Driving:
                l_id = f"{l.road_id}_{l.section_id}_{l.lane_id}"
                if l_id in found_lanes:
                    break
                found_lanes.append(l_id)
                lane_width.append(l.lane_width)
                # print(target_point)
                l = l.get_left_lane()
            road_width_l = sum(lane_width)

            lane_width = [w.lane_width * 0.5]
            found_lanes = [f"{w.road_id}_{w.section_id}_{w.lane_id}"]
            r = w.get_right_lane()
            while r and r.lane_type == carla.LaneType.Driving:
                r_id = f"{r.road_id}_{r.section_id}_{r.lane_id}"
                if r_id in found_lanes:
                    break
                found_lanes.append(r_id)
                lane_width.append(r.lane_width)
                r = r.get_right_lane()
            road_width_r = sum(lane_width)

        lanes = [get_lane(self._carla_map, vehicle_location, road_width_l, road_width_r)]

        lane_message = LanesMessage(timestamp, lanes)
        lane_message.max_distance_l = road_width_l
        lane_message.max_distance_r = road_width_r
        lanes_stream.send(lane_message)
        lanes_stream.send(erdos.WatermarkMessage(timestamp))

    def send_route(self, route_stream, timestamp = None):
        if timestamp is None:
            timestamp = erdos.Timestamp(coordinates=[0])
        # Gets global waypoints from the agent.
        # todo: maybe need reroute
        waypoints = deque([])
        road_options = deque([])
        for (transform, road_option) in self._global_plan_world_coord:
            waypoints.append(pylot.utils.Transform.from_simulator_transform(transform))
            road_options.append(pylot.utils.RoadOption(road_option.value))
        waypoints = Waypoints(waypoints, road_options=road_options)

        route_stream.send(WaypointsMessage(timestamp, waypoints))
        route_stream.send(erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))

    def run_step(self, input_data, timestamp):
        # command = self._run_step(input_data, timestamp)
        # self.step += 1
        # return command
        #
        while True:
            if self.tries > 5:
                control = carla.VehicleControl()
                control.steer = 0.0
                control.throttle = 0.0
                control.brake = 0.0
                command = control
            else:
                try:
                    command = self._run_step(input_data, timestamp)
                except Exception as e:
                    self.tries += 1
                    logger.warning('Pylot error, should be restart')
                    logger.error(traceback.print_exc())
                    self.destroy_pylot()
                    self.start_pylot()
                    continue

            self.step += 1

            ##### add recorder #####
            ego_location = self._ego_vehicle.get_location()
            ego_transform = self._ego_vehicle.get_transform()
            ego_velocity = self._ego_vehicle.get_velocity()
            ego_acceleration = self._ego_vehicle.get_acceleration()
            ego_speed = get_forward_value(transform=ego_transform, velocity=ego_velocity)  # In m/s
            ego_head_acc = get_forward_value(transform=ego_transform, velocity=ego_acceleration)
            vis_data = {
                "basic_info": f"(Basic) x: {ego_location.x:.2f} y: {ego_location.y:.2f} speed: {ego_speed:.2f} acc: {ego_head_acc:.2f}",
                "control": f"(Control) throttle: {command.throttle:.2f} steer: {command.steer:.2f} brake: {command.brake:.2f}"
            }
            input_data['vis_text'] = vis_data
            self.save_recorder(input_data, command)
            return command
    def _run_step(self, input_data, timestamp):
        """
        TODO: add try to catch IOError in erdos
        TODO: modify the destination controller
        Tracking:
            1. vehicle_id_stream
            2. ground_obstacles_stream
            3. pose_stream: current ego pose

        Prediction:
            1. tracking_stream

        Behavior Planning:
            1. pose_stream
            2. open_drive_stream: this should be only sent once
            3. route_stream: todo: should be confirmed, I think, this should be only sent once (reroute)???
            4. goal_location: None

        Planning:
            1. pose_stream: pose
            2. prediction_stream: prediction output
            3. static_obstacles_stream: traffic_light_stream
            4. lanes_stream: todo: should be confirmed, send perfect lanes
            5. route_stream: Behavior_Planner output
            6. open_drive_stream: map

        Control:
            1. pose_stream: pose
            2. waypoints_stream: planning output

        :param input_data:
        :param timestamp:
        :return:
        """
        if not self.initialized:
            self._init()

        game_time = int(timestamp * 1000)
        erdos_timestamp = erdos.Timestamp(coordinates=[game_time])

        # 1. send pose
        self.send_perfect_pose(self._pose_stream, erdos_timestamp)
        # 2. send perfect detection
        self.send_perfect_object_detection(self._obstacles_stream, self._static_obstacles_stream, erdos_timestamp)
        # 3. send perfect lanes
        self.send_perfect_lanes(self._lanes_stream, erdos_timestamp)

        command = read_control_command(self._control_stream)
        # logger.debug(command)
        if command is None:
            if self.last_control is None:
                control = carla.VehicleControl()
                control.steer = 0.0
                control.throttle = 0.0
                control.brake = 0.0
                self.last_control = control
            command = self.last_control
        else:
            self.last_control = command

        # logger.debug(command)
        return command

    ##### add recorder #####
    def save_recorder(self, tick_data, control):

        ego_location = self._ego_vehicle.get_location()
        ego_transform = self._ego_vehicle.get_transform()
        ego_velocity = self._ego_vehicle.get_velocity()
        ego_acceleration = self._ego_vehicle.get_acceleration()
        ego_speed = get_forward_value(transform=ego_transform, velocity=ego_velocity)  # In m/s
        ego_head_acc = get_forward_value(transform=ego_transform, velocity=ego_acceleration)
        original_control = [control.throttle, control.steer, control.brake]
        vis_data = {
            "basic_info": f"(Basic) x: {ego_location.x:.2f} y: {ego_location.y:.2f} speed: {ego_speed:.2f} acc: {ego_head_acc:.2f}",
            "control": f"(Control) throttle: {original_control[0]:.2f} steer: {original_control[1]:.2f} brake: {original_control[2]:.2f}"
        }
        tick_data['vis_text'] = vis_data

        frame = self.step
        image = self._hic.run_interface(tick_data)
        Image.fromarray(image).save(self.save_path / 'monitor_view' / ('%04d.png' % frame))

        if self.step % 10 == 0:
            # todo: add route waypoint filter
            self.recorder_waypoints = cut_waypoints(ego_transform, self.recorder_waypoints)
            meta_data = self.lang_repair.get_meta_data(self._world, self._ego_vehicle, self.recorder_waypoints, self.hist_waypoint_buffer, control)
            state_signal = read_behavior_command(self._behavior_stream, self.last_state)
            meta_data['behavior_decision'] = state_signal
            self.last_state = state_signal

            data_path = (self.save_path / 'meta' / ('%04d.pkl' % frame))
            with open(data_path, 'wb') as f:
                pickle.dump(meta_data, f)

            Image.fromarray(image).save(self.save_path / 'key_frames' / ('%04d.png' % frame))

            # update hist
            carla_map = CarlaDataProvider.get_map()
            ego_location = self._ego_vehicle.get_location()
            current_waypoint = carla_map.get_waypoint(ego_location)
            if len(self.hist_waypoint_buffer) == 0:
                self.hist_waypoint_buffer.append(current_waypoint)
            else:
                last_waypoint = self.hist_waypoint_buffer[-1]
                if last_waypoint.transform.location.distance(current_waypoint.transform.location) > 1.99:
                    self.hist_waypoint_buffer.append(current_waypoint)

    def sensors(self):

        ##### monitor sensors #####
        sensors = [
                {
                    'type': 'sensor.camera.rgb',
                    'x': - 6.0, 'y': 0.0, 'z': 2.0,
                    'roll': 0.0, 'pitch': -15.0, 'yaw': 0.0,
                    'width': 1200, 'height': 400, 'fov': 100,
                    'id': 'recorder_rgb_front'
                    }
        ]
        # recorder_sensors = []
        return sensors

def create_data_flow(goal_destination = None):
    """Creates a dataflow graph of operators.

    This is the place to add other operators (e.g., other detectors,
    behavior planning).
    """
    streams_to_send_top_on = []
    streams_to_be_monitored = []

    time_to_decision_loop_stream = erdos.LoopStream()

    # Creates a stream on which the agent sends the high-level route the
    # agent must follow in the challenge.
    route_stream = erdos.IngestStream(_name='route')
    streams_to_be_monitored.append(route_stream)
    # Creates a stream on which the agent sends the open drive stream it
    # receives when it executes in the MAP track.
    open_drive_stream = erdos.IngestStream(_name='open_drive_map')
    streams_to_be_monitored.append(open_drive_stream)

    vehicle_id_stream = erdos.IngestStream(_name='vehicle_id')
    streams_to_be_monitored.append(vehicle_id_stream)

    pose_stream = erdos.IngestStream(_name='pose')
    streams_to_be_monitored.append(pose_stream)
    # Stream on which the obstacles are sent when the agent is using perfect
    # detection.
    obstacles_stream = erdos.IngestStream(_name='obstacles')
    streams_to_be_monitored.append(obstacles_stream)
    # Stream on which the traffic lights are sent when the agent is
    # using perfect traffic light detection.
    static_obstacles_stream = erdos.IngestStream(_name='static_obstacles')
    streams_to_be_monitored.append(static_obstacles_stream)
    lanes_stream = erdos.IngestStream(_name='lanes')
    streams_to_be_monitored.append(lanes_stream)

    # Adds an operator for tracking detected agents. The operator uses the
    # frames from the center camera, and the bounding boxes found by the
    # obstacle detector operator.
    obstacles_tracking_stream = add_perfect_obstacle_tracking(
        vehicle_id_stream,
        obstacles_stream,
        pose_stream
    )
    streams_to_be_monitored.append(obstacles_tracking_stream)

    # The agent uses a linear predictor to compute future trajectories
    # of the other agents.
    prediction_stream = pylot.operator_creator.add_linear_prediction(
        obstacles_tracking_stream
    )
    streams_to_be_monitored.append(prediction_stream)

    # Adds a planner to the agent. The planner receives the pose of
    # the ego-vehicle, detected traffic lights, predictions for other
    # agents, the route the agent must follow, and the open drive data if
    # the agent is executing in the Map track, or detected lanes if it is
    # executing in the Sensors-only track.
    waypoints_stream, extract_behavior_stream, extract_planning_stream = add_planning(
        goal_destination,
        pose_stream,
        prediction_stream,
        static_obstacles_stream,
        lanes_stream,
        open_drive_stream,
        route_stream,
        time_to_decision_loop_stream
    )
    streams_to_be_monitored.append(waypoints_stream)

    # Adds a PID controller which tries to follow the waypoints computed
    # by the planner.
    control_stream = pylot.operator_creator.add_pid_control(
        pose_stream,
        waypoints_stream
    )
    streams_to_be_monitored.append(control_stream)
    # The PID planner returns a stream of commands (i.e., throttle, steer)
    # from which the agent can read the command it must return to the
    # challenge.
    extract_control_stream = erdos.ExtractStream(
        control_stream
    )

    # Operator that computes how much time each component gets to execute.
    # This is needed in Pylot, but can be ignored when running in challenge
    # mode.
    time_to_decision_stream = pylot.operator_creator.add_time_to_decision(pose_stream, obstacles_stream)
    time_to_decision_loop_stream.set(time_to_decision_stream)

    return (
        pose_stream,
        route_stream,
        lanes_stream,
        open_drive_stream,
        extract_control_stream,
        obstacles_stream,
        static_obstacles_stream,
        vehicle_id_stream,
        streams_to_send_top_on,
        extract_behavior_stream,
        extract_planning_stream,
        streams_to_be_monitored
    )

def add_perfect_obstacle_tracking(vehicle_id_stream, ground_obstacles_stream, pose_stream):
    assert (pose_stream is not None and ground_obstacles_stream is not None)
    obstacles_tracking_stream = pylot.operator_creator.add_perfect_tracking(vehicle_id_stream, ground_obstacles_stream, pose_stream)
    return obstacles_tracking_stream

def add_planning(
        goal_location,
        pose_stream,
        prediction_stream,
        static_obstacles_stream,
        lanes_stream,
        open_drive_stream,
        global_trajectory_stream,
        time_to_decision_stream
):
    trajectory_stream = pylot.operator_creator.add_behavior_planning(
        pose_stream, open_drive_stream, global_trajectory_stream,
        goal_location)

    waypoints_stream = pylot.operator_creator.add_planning(
        pose_stream, prediction_stream, static_obstacles_stream, lanes_stream,
        trajectory_stream, open_drive_stream, time_to_decision_stream)

    behavior_planning_reader = erdos.ExtractStream(
        trajectory_stream
    )

    planning_reader = erdos.ExtractStream(
        waypoints_stream
    )

    return waypoints_stream, behavior_planning_reader, planning_reader

def read_control_command(control_stream):
    # Wait until the control is set.
    tries = 0
    while True:
        tries += 1
        if tries > 50:
            return None
        # Read the control command from the control stream.
        control_msg = control_stream.try_read()
        if control_msg is not None and (not isinstance(control_msg, erdos.WatermarkMessage)):
            # We have read a control message. Return the command
            # so that the leaderboard can tick the simulator.
            output_control = carla.VehicleControl()
            output_control.throttle = control_msg.throttle
            output_control.brake = control_msg.brake
            output_control.steer = control_msg.steer
            output_control.reverse = control_msg.reverse
            output_control.hand_brake = control_msg.hand_brake
            output_control.manual_gear_shift = False
            return output_control

def read_behavior_command(behavior_stream, last_state):
    tries = 0
    while True:
        tries += 1
        if tries > 20:
            return last_state

        msg = behavior_stream.try_read()
        if msg is None or isinstance(msg, erdos.WatermarkMessage):
            continue

        if msg.agent_state:
            agent_state = int(msg.agent_state.value)
            return agent_state