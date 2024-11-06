import cv2
import time
import erdos
import carla
import logging
import numpy as np
import pylot.flags
import pylot.component_creator  # noqa: I100
import pylot.operator_creator
import pylot.perception.messages
import pylot.utils

from collections import deque
from absl import flags
from loguru import logger

from pylot.map.hd_map import HDMap
from pylot.perception.messages import ObstaclesMessage, TrafficLightsMessage, LanesMessage
from pylot.planning.messages import WaypointsMessage
from pylot.planning.waypoints import Waypoints
from pylot.drivers.sensor_setup import RGBCameraSetup
from pylot.simulation.utils import extract_data_in_pylot_format
from pylot.localization.messages import GNSSMessage, IMUMessage

from ads.systems.Pylot.perception_gt import *

from scenario.utils.carla_data_provider import CarlaDataProvider
from scenario.utils.route_manipulation import downsample_route
from ..base_collection import AutonomousAgent, interpolate_trajectory, convert_transform_to_location

CENTER_CAMERA_LOCATION = pylot.utils.Location(0.0, 0.0, 2.0)
TL_CAMERA_NAME = 'traffic_lights_camera'

FLAGS = flags.FLAGS


def enable_logging():
    """Overwrites logging config so that loggers can control verbosity.

    This method is required because the challenge evaluator overwrites
    verbosity, which causes Pylot log messages to be discarded.
    """
    import logging
    logging.root.setLevel(logging.INFO)


class PylotAgent(AutonomousAgent):
    """Agent class that interacts with the challenge leaderboard.
    TODO: update the planning lane width config

    Attributes:
        _camera_setups: Mapping between camera names and
            :py:class:`~pylot.drivers.sensor_setup.CameraSetup`.
        _lidar_setup (:py:class:`~pylot.drivers.sensor_setup.LidarSetup`):
            Setup of the Lidar sensor.
    """

    def setup(self, path_to_conf_file):
        pylot.utils.set_tf_loglevel(logging.INFO)
        # Parse the flag file. Users can use the different flags defined
        # across the Pylot directory.
        flags.FLAGS([__file__, '--flagfile={}'.format(path_to_conf_file)])
        flags.FLAGS.simulator_port = CarlaDataProvider.get_carla_port()
        flags.FLAGS.simulator_host = CarlaDataProvider.get_carla_host()
        logger.debug(f"simulator_port: {flags.FLAGS.simulator_port}, simulator_host: {flags.FLAGS.simulator_host}")
        self.logger = erdos.utils.setup_logging('erdos_agent',
                                                FLAGS.log_file_name)
        self.csv_logger = erdos.utils.setup_csv_logging(
            'erdos_agent_csv', FLAGS.csv_log_file_name)
        enable_logging()

        # Town name is only used when the agent is directly receiving
        # traffic lights from the simulator.
        self._town_name = None
        # Stores a simulator handle to the ego vehicle. This handle is only
        # used when the agent is using a perfect localization or perception.
        self._ego_vehicle = None
        # Stores ego-vehicle's yaw from last game time. This is used in the
        # naive localization solution.
        self._last_yaw = 0
        # Stores the point cloud from the previous sensor reading.

        self._world = CarlaDataProvider.get_world()

        # config
        self._sent_open_drive = False
        self.initialized = False

        # Create the dataflow of AV components. Change the method
        # to add your operators.
        (self._camera_streams, self._pose_stream, self._route_stream,
         self._global_trajectory_stream, self._open_drive_stream,
         self._point_cloud_stream, self._imu_stream, self._gnss_stream,
         self._control_stream, self._control_display_stream,
         self._perfect_obstacles_stream, self._perfect_traffic_lights_stream,
         self._vehicle_id_stream, streams_to_send_top_on) = create_data_flow()
        # Execute the dataflow.
        # NOTE: graph_filename=f'pylot_{find_port}', start_port=find_port
        # TODO: may need port setting
        self._node_handle = erdos.run_async()
        # self._node_handle = erdos.run_async(start_port=CarlaDataProvider.get_agent_port())
        # logger.debug(f'Connect erdos from {CarlaDataProvider.get_agent_port()}')
        # Close the streams that are not used (i.e., send top watermark).
        for stream in streams_to_send_top_on:
            stream.send(erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))

    def set_global_plan(self):
        """
        Set the plan (route) for the agent
        global_plan_world_coord: route.append((wp_tuple[0].transform, wp_tuple[1]))
        """
        # get more waypoints in case of bugs
        ego_waypoint_route = self.route_config
        world = CarlaDataProvider.get_world()
        gps_route, route, wp_route = interpolate_trajectory(world, ego_waypoint_route)
        CarlaDataProvider.set_ego_vehicle_route(convert_transform_to_location(route))
        ds_ids = downsample_route(route, 50)
        self._global_route = wp_route
        self._global_plan_world_coord = [(route[x][0], route[x][1]) for x in ds_ids]
        self._global_plan = [gps_route[x] for x in ds_ids]
        self._plan_gps_HACK = gps_route  # FULL for waypoint planner TODO: this may has potential bugs in steer or something else
        self._plan_HACK = route

    def _init(self):
        # Attach recording sensors to the ego vehicle.
        self._world = CarlaDataProvider.get_world()
        self._open_drive_map = CarlaDataProvider.get_map().to_opendrive()
        self._ego_vehicle = CarlaDataProvider.get_ego()
        self._od_map = HDMap(carla.Map('map', self._open_drive_map), None)
        self._carla_map = self._world.get_map()
        self.bev_observer.attach_ego_vehicle(self._ego_vehicle)
        self.meta_observer.attach_ego_vehicle(self._ego_vehicle)
        logger.info("initialized")
        self.initialized = True

    def destroy(self):
        """Clean-up the agent. Invoked between different runs."""
        self.logger.info('ERDOSAgent destroy method invoked')
        # Stop the ERDOS node.
        self._node_handle.shutdown()
        # Reset the ERDOS dataflow graph.
        erdos.reset()
        time.sleep(1.0)
        if self.require_saver:
            self.data_saver.close()

    def send_vehicle_id_msg(self, vehicle_id_stream):
        # confirm the ego vehicle id, only send once at the start
        if not vehicle_id_stream.is_closed():
            vehicle_id_stream.send(erdos.Message(erdos.Timestamp(coordinates=[0]), self._ego_vehicle.id))
            vehicle_id_stream.send(erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))

    def compute_pose(self, speed_data, imu_data, gnss_data, timestamp):
        """Computes the pose of the ego-vehicle.

        This method implements a naive localization that transforms the
        noisy gnss readings into locations, and the noisy IMU readings into
        rotations.
        """
        forward_speed = speed_data['speed']
        latitude = gnss_data[0]
        longitude = gnss_data[1]
        altitude = gnss_data[2]
        location = pylot.utils.Location.from_gps(latitude, longitude, altitude)
        if np.isnan(imu_data[6]):
            yaw = self._last_yaw
        else:
            compass = np.degrees(imu_data[6])
            if compass < 270:
                yaw = compass - 90
            else:
                yaw = compass - 450
            self._last_yaw = yaw
        vehicle_transform = pylot.utils.Transform(
            location, pylot.utils.Rotation(yaw=yaw))
        velocity_vector = pylot.utils.Vector3D(forward_speed * np.cos(yaw),
                                               forward_speed * np.sin(yaw), 0)
        current_pose = pylot.utils.Pose(vehicle_transform, forward_speed,
                                        velocity_vector,
                                        timestamp.coordinates[0])
        return current_pose

    def send_imu_msg(self, imu_stream, imu_data, timestamp):
        """Sends the IMU data on the Pylot stream."""
        accelerometer = pylot.utils.Vector3D(imu_data[0], imu_data[1],
                                             imu_data[2])
        gyroscope = pylot.utils.Vector3D(imu_data[3], imu_data[4], imu_data[5])
        compass = imu_data[6]
        # Build a Pylot IMUMessage out of the challenge IMU sensor data.
        msg = IMUMessage(timestamp, None, accelerometer, gyroscope, compass)
        imu_stream.send(msg)
        imu_stream.send(erdos.WatermarkMessage(timestamp))

    def send_gnss_msg(self, gnss_stream, gnss_data, timestamp):
        """Sends the GNSS data on the Pylot stream."""
        latitude = gnss_data[0]
        longitude = gnss_data[1]
        altitude = gnss_data[2]
        location = pylot.utils.Location.from_gps(latitude, longitude, altitude)
        transform = pylot.utils.Transform(location, pylot.utils.Rotation())
        # Build a Pylot GNSSMessage out of the challenge GNSS data.
        msg = GNSSMessage(timestamp, transform, altitude, latitude, longitude)
        gnss_stream.send(msg)
        gnss_stream.send(erdos.WatermarkMessage(timestamp))

    def send_localization(self, timestamp, imu_data, gnss_data, speed_data):
        # The agent uses our localization. We need to send data on the
        # IMU and GNSS streams, and the initial position of the ego-vehicle
        # on the route stream.
        self.send_imu_msg(self._imu_stream, imu_data, timestamp)
        self.send_gnss_msg(self._gnss_stream, gnss_data, timestamp)
        # Naively compute the pose of the ego-vehicle, and send it on
        # the route stream. Pylot's localization operator will refine this
        # pose using the GNSS and IMU data.
        pose = self.compute_pose(speed_data, imu_data, gnss_data,
                                 timestamp)
        self._route_stream.send(erdos.Message(timestamp, pose))
        self._route_stream.send(erdos.WatermarkMessage(timestamp))

        # # In this configuration, the agent is not using a localization
        # # operator. It is driving using the noisy localization it receives
        # # from the leaderboard.
        # pose = self.compute_pose(speed_data, imu_data, gnss_data,
        #                          timestamp)
        # self._pose_stream.send(erdos.Message(timestamp, pose))
        # self._pose_stream.send(erdos.WatermarkMessage(timestamp))

    def send_perfect_detections(
            self,
            perfect_obstacles_stream,
            perfect_traffic_lights_stream,
            timestamp,
            tl_camera_location
    ):
        """Send perfect detections for agents and traffic lights.

        This method first connects to the simulator to extract all the
        agents and traffic light in a scenario. Next, it transforms them into
        the types Pylot expects, and sends them on the streams for perfect
        detections.

        Note: This is only used when executing using a perfect perception
        component.
        """
        if (not FLAGS.simulator_obstacle_detection) or (not FLAGS.simulator_traffic_light_detection):
            raise RuntimeError(
                f"Must set to use perfect perception {FLAGS.simulator_obstacle_detection}, {FLAGS.simulator_traffic_light_detection}")

        actor_list = self._world.get_actors()
        # vehicles, people, traffic_lights, speed_limits, traffic_stops
        # vehicles, people, traffic_lights, speed_limits, traffic_stops
        vehicles, people, traffic_lights, speed_limits, traffic_stops, _ = extract_data_in_pylot_format(actor_list)
        # print(info)
        # exit(-1)
        # if FLAGS.simulator_obstacle_detection:
        # send perfect obstacle detection
        perfect_obstacles_stream.send(
            ObstaclesMessage(timestamp, vehicles + people))
        perfect_obstacles_stream.send(erdos.WatermarkMessage(timestamp))

        # if FLAGS.simulator_traffic_light_detection:
        # send perfect traffic light detection
        vec_transform = pylot.utils.Transform.from_simulator_transform(
            self._ego_vehicle.get_transform())
        tl_camera_transform = pylot.utils.Transform(
            tl_camera_location, pylot.utils.Rotation())
        visible_tls = []
        if self._town_name is None:
            self._town_name = self._world.get_map().name
        for tl in traffic_lights:
            if tl.is_traffic_light_visible(
                    vec_transform * tl_camera_transform,
                    self._town_name,
                    distance_threshold=FLAGS.
                    static_obstacle_distance_threshold):
                if self._town_name not in ['Town01', 'Town02']:
                    delta_y = -5
                    if self._town_name == 'Town04':
                        delta_y = -2
                    # Move the traffic light location to the road.
                    tl.transform = tl.transform * pylot.utils.Transform(
                        pylot.utils.Location(delta_y, 0, 5),
                        pylot.utils.Rotation())
                visible_tls.append(tl)
        perfect_traffic_lights_stream.send(
            TrafficLightsMessage(timestamp, visible_tls))
        perfect_traffic_lights_stream.send(
            erdos.WatermarkMessage(timestamp))

    def send_global_trajectory_msg(self, global_trajectory_stream, timestamp):
        """Sends the route the agent must follow."""
        # Send once the global waypoints.
        if not global_trajectory_stream.is_closed():
            # Gets global waypoints from the agent.
            waypoints = deque([])
            road_options = deque([])
            for (transform, road_option) in self._global_plan_world_coord:
                waypoints.append(
                    pylot.utils.Transform.from_simulator_transform(transform))
                road_options.append(pylot.utils.RoadOption(road_option.value))
            waypoints = Waypoints(waypoints, road_options=road_options)
            global_trajectory_stream.send(
                WaypointsMessage(timestamp, waypoints))
            global_trajectory_stream.send(
                erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))

    def run_step(self, input_data, timestamp):
        # print(input_data.keys())

        start_time = time.time()
        game_time = int(timestamp * 1000)
        erdos_timestamp = erdos.Timestamp(coordinates=[game_time])

        if not self.initialized:
            self._init()
            # 1. send ego vehicle id
            self.send_vehicle_id_msg(self._vehicle_id_stream)

            # 2. send opendrive
            if not self._open_drive_stream.is_closed():
                # The data is only sent once because it does not change
                # throught the duration of a scenario.
                self._open_drive_stream.send(
                    erdos.Message(erdos_timestamp, input_data['opendrive'][1]['opendrive']))
                # Inform the operators that read the open drive stream that
                # they will not receive any other messages on this stream.
                self._open_drive_stream.send(
                    erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))

            # 3. send route here - only send once
            self.send_global_trajectory_msg(self._global_trajectory_stream, erdos_timestamp)

        # Parse the sensor data the agent receives from the leaderboard.
        speed_data = input_data['speed'][1]
        imu_data = input_data['imu'][1]
        gnss_data = input_data['gnss'][1]

        # 1. send pose
        self.send_localization(
            erdos_timestamp,
            imu_data,
            gnss_data,
            speed_data
        )

        # 2. send perfect detection
        self.send_perfect_detections(
            self._perfect_obstacles_stream,
            self._perfect_traffic_lights_stream,
            erdos_timestamp,
            CENTER_CAMERA_LOCATION
        )

        if not self._open_drive_stream.is_closed():
            # We do not have access to the open drive map. Send top watermark.
            self._open_drive_stream.send(
                erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))

        sensor_send_runtime = (time.time() - start_time) * 1000
        self.csv_logger.info('{},{},sensor_send_runtime,{:.4f}'.format(
            pylot.utils.time_epoch_ms(), game_time, sensor_send_runtime))

        # print('placeholder')

        # Return the control command received on the control stream.
        command = read_control_command(self._control_stream)
        # print(command)
        e2e_runtime = (time.time() - start_time) * 1000
        self.csv_logger.info('{},{},e2e_runtime,{:.4f}'.format(
            pylot.utils.time_epoch_ms(), game_time, e2e_runtime))

        # control
        self.tick_save(
            self.step,
            curr_control=command,
            speed=input_data['speed'][1]['speed'],  # speed = input_data['speed'][1]['speed']
            features=None,
            gps=gnss_data,
            imu=input_data['imu'][1],
            target_point=np.array([0.0, 0.0]),  # TODO: may need the message from the pylot
            view_img=cv2.cvtColor(input_data['recorder_rgb_front'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        )
        return command

    def sensors(self):
        roach_senors = [
            {
                'type': 'sensor.other.imu',
                'x': 0.0, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'sensor_tick': 0.05,
                'id': 'imu'
            },
            {
                'type': 'sensor.other.gnss',
                'x': 0.0, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'sensor_tick': 0.01,
                'id': 'gnss'
            },
            {
                'type': 'sensor.speedometer',
                'reading_frequency': 20,
                'id': 'speed'
            },
            {
                'type': 'sensor.opendrive_map',
                'reading_frequency': 20,
                'id': 'opendrive'
            }
        ]

        recorder_sensors = self.get_recorder_sensors()
        return roach_senors + recorder_sensors


def create_camera_setups():
    camera_setups = {}
    # Add a center front camera.
    transform = pylot.utils.Transform(CENTER_CAMERA_LOCATION,
                                      pylot.utils.Rotation())
    # only use traffic in this project
    # Add a camera with a narrow field of view. The traffic light
    # camera is added in the same position as the center camera.
    # We use this camera for traffic light detection.
    # TODO: Check if requires the actual sensor
    center_camera_setup = RGBCameraSetup('center_camera',
                                         FLAGS.camera_image_width,
                                         FLAGS.camera_image_height, transform,
                                         90)
    camera_setups['center_camera'] = center_camera_setup

    tl_camera_setup = RGBCameraSetup(TL_CAMERA_NAME,
                                     FLAGS.camera_image_width,
                                     FLAGS.camera_image_height, transform,
                                     45)
    camera_setups[TL_CAMERA_NAME] = tl_camera_setup
    return camera_setups


def create_data_flow():
    """Creates a dataflow graph of operators.

    This is the place to add other operators (e.g., other detectors,
    behavior planning).
    """
    streams_to_send_top_on = []

    camera_setups = create_camera_setups()
    camera_streams = {}
    for name in camera_setups:
        camera_streams[name] = erdos.IngestStream()

    # Creates a stream on which the agent sends the high-level route the
    # agent must follow in the challenge.
    global_trajectory_stream = erdos.IngestStream()  # high-level route!!!
    # Creates a stream on which the agent sends the open drive stream it
    # receives when it executes in the MAP track.
    open_drive_stream = erdos.IngestStream()
    # create for localization
    point_cloud_stream = erdos.IngestStream()  # not used
    imu_stream = erdos.IngestStream()
    gnss_stream = erdos.IngestStream()
    route_stream = erdos.IngestStream()  # used in localization!!!
    time_to_decision_loop_stream = erdos.LoopStream()

    # localization, TODO: add localization information in monitor
    pose_stream = pylot.operator_creator.add_localization(
        imu_stream, gnss_stream, route_stream)
    # pose_stream = erdos.IngestStream()

    # perception streams (perfect)
    obstacles_stream = erdos.IngestStream()
    # traffic light requires the TL_CAMERA
    traffic_lights_stream = erdos.IngestStream()
    camera_streams[TL_CAMERA_NAME] = erdos.IngestStream()
    streams_to_send_top_on.append(camera_streams[TL_CAMERA_NAME])

    vehicle_id_stream = erdos.IngestStream()
    # streams_to_send_top_on.append(vehicle_id_stream) # not sure

    # tracking (perfect)
    # Adds an operator for tracking detected agents. The operator uses the
    # frames from the center camera, and the bounding boxes found by the
    # obstacle detector operator.
    obstacles_tracking_stream = \
        pylot.operator_creator.add_perfect_tracking(
            vehicle_id_stream, obstacles_stream, pose_stream)

    # lane detection
    # The lanes stream is not used when running in the Map track.
    # We add the stream to the list of streams that are not used, and
    # must be manually "closed" (i.e., send a top watermark).
    # NOTE: the stream is manually closed!!! by sending a top watermark
    lanes_stream = erdos.IngestStream()
    streams_to_send_top_on.append(lanes_stream)

    # prediction
    prediction_stream, _, _ = pylot.component_creator.add_prediction(
        obstacles_tracking_stream,
        vehicle_id_stream,
        time_to_decision_loop_stream,
        camera_transform=None,  # should disable FLAGS.visualize_prediction
        release_sensor_stream=None,  # should disable FLAGS.visualize_prediction
        pose_stream=pose_stream)

    # TODO: original is following, check the third stream (time to decision loop)
    # prediction_stream, _, _ = pylot.component_creator.add_prediction(
    #     obstacles_tracking_stream,
    #     vehicle_id_stream,
    #     time_to_decision_loop_stream,
    #     pose_stream=pose_stream)

    # planning
    # Adds a planner to the agent. The planner receives the pose of
    # the ego-vehicle, detected traffic lights, predictions for other
    # agents, the route the agent must follow, and the open drive data if
    # the agent is executing in the Map track, or detected lanes if it is
    # executing in the Sensors-only track.
    # NOTE: one of lanes and open_drive must receive data
    waypoints_stream = pylot.component_creator.add_planning(
        None, pose_stream, prediction_stream, traffic_lights_stream,
        lanes_stream, open_drive_stream, global_trajectory_stream,
        time_to_decision_loop_stream)

    # display - disable
    control_display_stream = None

    # control
    # Adds a controller which tries to follow the waypoints computed
    # by the planner.
    control_stream = pylot.component_creator.add_control(
        pose_stream, waypoints_stream)
    # The controller returns a stream of commands (i.e., throttle, steer)
    # from which the agent can read the command it must return to the
    # challenge.
    extract_control_stream = erdos.ExtractStream(control_stream)

    # Operator that computes how much time each component gets to execute.
    # This is needed in Pylot, but can be ignored when running in challenge
    # mode.
    time_to_decision_stream = pylot.operator_creator.add_time_to_decision(
        pose_stream, obstacles_stream)
    time_to_decision_loop_stream.set(time_to_decision_stream)

    # others are set to None
    perfect_obstacles_stream = obstacles_stream
    perfect_traffic_lights_stream = traffic_lights_stream

    return (camera_streams, pose_stream, route_stream,
            global_trajectory_stream, open_drive_stream, point_cloud_stream,
            imu_stream, gnss_stream, extract_control_stream,
            control_display_stream, perfect_obstacles_stream,
            perfect_traffic_lights_stream, vehicle_id_stream,
            streams_to_send_top_on)


def add_perfect_obstacle_tracking(vehicle_id_stream, ground_obstacles_stream, pose_stream):
    assert (pose_stream is not None and ground_obstacles_stream is not None)
    obstacles_tracking_stream = pylot.operator_creator.add_perfect_tracking(
        vehicle_id_stream, ground_obstacles_stream, pose_stream)
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
    while True:
        # Read the control command from the control stream.
        control_msg = control_stream.read()
        if not isinstance(control_msg, erdos.WatermarkMessage):
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
