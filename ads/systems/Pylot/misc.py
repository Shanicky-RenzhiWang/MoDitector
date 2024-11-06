import socket

from pylot.utils import Location
from pylot.perception.detection.lane import Lane
from pylot.perception.detection.obstacle import Obstacle
from pylot.perception.detection.speed_limit_sign import SpeedLimitSign
from pylot.perception.detection.stop_sign import StopSign
from pylot.perception.detection.traffic_light import TrafficLight

def extract_data_in_pylot_format(actor_list):
    """Extracts actor information in pylot format from an actor list.

    Args:
        actor_list (carla.ActorList): An actor list object with all the
            simulation actors.

    Returns:
        A tuple that contains objects for all different types of actors.
    """
    # Note: the output will include the ego vehicle as well.
    vec_actors = actor_list.filter('*vehicle*')
    vehicles = [
        Obstacle.from_simulator_actor(vec_actor) for vec_actor in vec_actors
    ]

    person_actors = actor_list.filter('*walker.pedestrian*')
    people = [
        Obstacle.from_simulator_actor(ped_actor) for ped_actor in person_actors
    ]

    tl_actors = actor_list.filter('*traffic_light*')
    traffic_lights = [
        TrafficLight.from_simulator_actor(tl_actor) for tl_actor in tl_actors
    ]

    speed_limit_actors = actor_list.filter('*speed_limit*')
    speed_limits = [
        SpeedLimitSign.from_simulator_actor(ts_actor)
        for ts_actor in speed_limit_actors
    ]

    traffic_stop_actors = actor_list.filter('*stop*')
    traffic_stops = [
        StopSign.from_simulator_actor(ts_actor)
        for ts_actor in traffic_stop_actors
    ]

    traffic_cone_actors = actor_list.filter('*static.prop.trafficcone*')
    traffic_cones = [
        Obstacle.from_simulator_actor(tc_actor) for tc_actor in traffic_cone_actors
    ]

    return (vehicles, people, traffic_lights, speed_limits, traffic_stops, traffic_cones)


def lateral_shift(transform, shift):
    transform.rotation.yaw += 90
    shifted = transform.location + shift * transform.get_forward_vector()
    return shifted

def get_lane(
        carla_map,
        location,
        waypoint_precision: float = 0.05,
        lane_id: int = 0,
        left_road_width: float = 0.85,
        right_road_width: float = 0.85
):
    lane_waypoints = []
    # Consider waypoints in opposite direction of camera so we can get
    # lane data for adjacent lanes in opposing directions.
    previous_wp = [carla_map.get_waypoint(location)]

    count = 0
    while len(previous_wp) == 1:
        count += 1
        if count > 10 / 0.05:
            break
        lane_waypoints.append(previous_wp[0])
        previous_wp = previous_wp[0].previous(waypoint_precision)

    next_wp = [carla_map.get_waypoint(location)]

    count = 0
    while len(next_wp) == 1:
        count += 1
        if count > 10 / 0.05:
            break
        lane_waypoints.append(next_wp[0])
        next_wp = next_wp[0].next(waypoint_precision)

    # Get the left and right markings of the lane and send it as a message.
    left_markings_carla = [
        lateral_shift(w.transform, -w.lane_width * 0.5)
        for w in lane_waypoints
    ]
    right_markings_carla = [
        lateral_shift(w.transform, w.lane_width * 0.5)
        for w in lane_waypoints
    ]

    left_markings = [
        Location(l.x, l.y, l.z)
        for l in left_markings_carla
    ]

    right_markings = [
        Location(l.x, l.y, l.z)
        for l in right_markings_carla
    ]

    return Lane(lane_id, left_markings, right_markings, left_road_width, right_road_width)

def find_unoccupied_port(start_port=1024, end_port=65535):
    for port in range(start_port, end_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                # If the bind is successful, return this port
                return port
            except OSError:
                # This means the port is already in use, so continue checking the next one
                continue
    raise Exception("No unoccupied ports found in the specified range.")