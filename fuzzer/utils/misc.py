
def estimate_lane_length(carla_map, road_id, lane_id):

    initial_wp = carla_map.get_waypoint_xodr(road_id, lane_id, 0.0)
    lane_length = 0.0

    curr_wp = initial_wp
    next_wps = curr_wp.next(1.0)
    while len(next_wps) == 1:
        curr_wp = next_wps[0]
        curr_road_id = curr_wp.road_id
        curr_lane_id = curr_wp.lane_id
        if curr_lane_id != lane_id or curr_road_id != road_id:
            break
        lane_length += 1
        next_wps = curr_wp.next(1.0)

    curr_wp = initial_wp
    prev_wps = curr_wp.previous(1.0)
    while len(prev_wps) == 1:
        curr_wp = prev_wps[0]
        curr_road_id = curr_wp.road_id
        curr_lane_id = curr_wp.lane_id
        if curr_lane_id != lane_id or curr_road_id != road_id:
            break
        lane_length += 1
        prev_wps = curr_wp.previous(1.0)

    return lane_length

