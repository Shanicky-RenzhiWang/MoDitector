import copy
import carla
import random
import traceback
import numpy as np
import networkx as nx

from typing import List
from loguru import logger

from shapely.geometry import LineString, Point

from scenario.configuration import (ScenarioConfig,
                                    WaypointUnit,
                                    ObjectInfo,
                                    BasicAgent,
                                    VehicleSection,
                                    WalkerSection,
                                    StaticSection,
                                    WeatherSection)

from scenario.utils.misc import estimate_lane_length

MIN_WALKER_SPEED = 0.1
MAX_WALKER_SPEED = 3.0
MIN_VEHICLE_SPEED = 0.6
MAX_VEHICLE_SPEED = 16.0
WEATHER_RANGE={
    'cloudiness': (0,100),
    'precipitation': (0,100),
    'fog_density': (0,100),
    'fog_distance': (0,100),
    'cloudiness': (0,100),
    'precipitation_deposits': (0,100),
    'wind_intensity': (0,100),
    'sun_azimuth_angle': (0,360),
    'sun_altitude_angle': (-90,90),
    'wetness': (0,100),
    'fog_falloff': (0,100),
}

class ScenarioMutator:

    frame_rate = 20.0

    def __init__(
            self,
            client,
            npc_vehicle_num: int,
            npc_walker_num: int,
            npc_static_num: int,
            prob_mutation: float = 0.5,
            sampling_resolution: float = 2.0
    ):
        self._client = client
        self._town = ''
        self._npc_vehicle_num = npc_vehicle_num
        self._npc_walker_num = npc_walker_num
        self._npc_static_num = npc_static_num
        self._prob_mutation = prob_mutation
        self._sampling_resolution = sampling_resolution

    def _setup(self):
        try:
            self._world = self._client.load_world(self._town)
        except Exception as e:
            logger.error(traceback.print_exc())

        settings = self._world.get_settings()
        settings.fixed_delta_seconds = 1.0 / self.frame_rate
        settings.synchronous_mode = True
        self._world.apply_settings(settings)
        self._world.reset_all_traffic_lights()
        self._world.tick()
        self._map = self._world.get_map()

    def _cleanup(self):
        # Reset to asynchronous mode
        settings = self._world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self._world.apply_settings(settings)

    def _put_actor(self, agent):
        # agent_info = agent.object_info

        bp_filter = agent.object_info.model
        blueprint = random.choice(self._world.get_blueprint_library().filter(bp_filter))

        # Get the spawn point
        agent_initial_waypoint = agent.get_initial_waypoint()
        agent_initial_transform = carla.Transform(
            location=carla.Location(
                x=agent_initial_waypoint.x,
                y=agent_initial_waypoint.y,
                z=0.0
            ),
            rotation=carla.Rotation(
                pitch=agent_initial_waypoint.pitch,
                yaw=agent_initial_waypoint.yaw,
                roll=agent_initial_waypoint.roll
            )
        )

        _spawn_point = carla.Transform()
        _spawn_point.rotation = agent_initial_transform.rotation
        _spawn_point.location.x = agent_initial_transform.location.x
        _spawn_point.location.y = agent_initial_transform.location.y
        _spawn_point.location.z = agent_initial_transform.location.z + 1.321  # + 0.2

        actor = self._world.try_spawn_actor(blueprint, _spawn_point)
        self._world.tick()

        if actor is None:
            return None

        return actor

    def _location_check(self, new_agent, existing_agents, ignore_ids: List = []):

        def clean(actors):
            for _actor in actors:
                if _actor:
                    _actor.destroy()

        # logger.debug(ignore_ids)
        added_actors = list()
        for agent in existing_agents:
            # logger.debug(agent.id)
            if agent.id in ignore_ids:
                # logger.debug(f"find {agent.id}")
                continue
            carla_actor = self._put_actor(agent)
            if carla_actor is None:
                clean(added_actors)
                return False

            # logger.debug(f"put {agent.id}")
            carla_actor.set_simulate_physics(enabled=True)
            added_actors.append(carla_actor)

        # logger.debug(f"new_agent {new_agent.id}")
        carla_actor = self._put_actor(new_agent)
        # logger.debug(carla_actor)
        if carla_actor is None:
            clean(added_actors)
            return False
        added_actors.append(carla_actor)

        clean(added_actors)
        return True

    def generate_new_vehicle(self, npc_id, scenario: ScenarioConfig):

        lanes_vehicle = scenario.map_section.lanes_vehicle

        vehicle_start_lane = random.choice(lanes_vehicle)
        road_id = int(vehicle_start_lane.split('_')[0])
        lane_id = int(vehicle_start_lane.split('_')[1])
        vehicle_start_lane_length = estimate_lane_length(self._map, road_id, lane_id)
        vehicle_start_s = random.uniform(0.01, vehicle_start_lane_length - 0.05)
        vehicle_start_wp = self._map.get_waypoint_xodr(road_id, lane_id, vehicle_start_s)
        vehicle_start_wp = self._map.get_waypoint(vehicle_start_wp.transform.location)

        route = [vehicle_start_wp]
        last_wp = route[-1]
        for i in range(500):
            next_wps = last_wp.next(2.0)
            if len(next_wps) == 0:
                break
            last_wp = random.choice(next_wps)
            route.append(last_wp)

        agent_route = list()
        agent_speed = random.uniform(3.0, 11.0)
        for i, rt in enumerate(route):
            wp = rt  # [0]
            wp = self._map.get_waypoint(wp.transform.location)
            location = wp.transform.location
            rotation = wp.transform.rotation

            agent_route.append(
                WaypointUnit(
                    x=location.x,
                    y=location.y,
                    z=location.z,
                    pitch=rotation.pitch,
                    yaw=rotation.yaw,
                    roll=rotation.roll,
                    speed=agent_speed
                )
            )

        # find type
        blueprint = random.choice(self._world.get_blueprint_library().filter('vehicle.*'))
        vehicle_object_info = ObjectInfo(
            model=blueprint.id,
            rolename='npc_vehicle',
            color=None,
            category='vehicle'
        )

        agent = BasicAgent(
            id=npc_id,
            route=agent_route,
            object_info=vehicle_object_info,
            mutable=True,
            trigger_time=random.uniform(0.0, 5.0)
        )
        return agent

    def generate_new_walker(self, npc_id, scenario: ScenarioConfig):

        nodes_walker = scenario.map_section.nodes_walker
        edges_walker = scenario.map_section.edges_walker

        graph_walker = nx.Graph()
        graph_walker.add_nodes_from(nodes_walker)
        graph_walker.add_edges_from(edges_walker)

        walker_sample_nodes = random.sample(nodes_walker, 2)
        walker_start_node = walker_sample_nodes[0]
        walker_end_node = walker_sample_nodes[1]
        walker_trace_nodes = nx.shortest_path(graph_walker, source=walker_start_node, target=walker_end_node)

        route = []
        for road_lane_i in range(len(walker_trace_nodes)):
            road_lane = walker_trace_nodes[road_lane_i]

            road_id = int(road_lane.split('_')[0])
            lane_id = int(road_lane.split('_')[1])

            lane_length = estimate_lane_length(self._map, road_id, lane_id)

            if road_lane_i == 0 or road_lane_i == len(walker_trace_nodes) - 1:
                point_s = random.uniform(0.01, lane_length)
                point_wp = self._map.get_waypoint_xodr(road_id, lane_id, point_s)
                point_wp = self._map.get_waypoint(point_wp.transform.location, project_to_road=True,
                                                 lane_type=carla.LaneType.Sidewalk)
                route.append(point_wp)
            else:
                point_s = lane_length / 2.0
                point_wp = self._map.get_waypoint_xodr(road_id, lane_id, point_s)
                point_wp = self._map.get_waypoint(point_wp.transform.location, project_to_road=True,
                                                 lane_type=carla.LaneType.Sidewalk)
                route.append(point_wp)

        # interpolate
        interpolate_route = list()
        for i in range(len(route) - 1):
            point_start = Point([route[i].transform.location.x, route[i].transform.location.y])
            point_end = Point([route[i + 1].transform.location.x, route[i + 1].transform.location.y])
            line = LineString([point_start, point_end])
            points = [line.interpolate(distance) for distance in range(0, int(line.length) + 1, 2)]

            # Print or use the points
            for point in points:
                point_wp = self._map.get_waypoint(
                    carla.Location(
                        x=point.x,
                        y=point.y,
                        z=0.0
                    ),
                    project_to_road=True,
                    lane_type=carla.LaneType.Sidewalk
                )
                if len(interpolate_route) == 0:
                    interpolate_route.append(point_wp)
                else:
                    if interpolate_route[-1].transform.location.distance(point_wp.transform.location) >= 1.0:
                        interpolate_route.append(point_wp)

        agent_route = list()
        agent_speed = random.uniform(0.1, 2.0)
        for i, rt in enumerate(interpolate_route):
            wp = rt
            location = wp.transform.location
            rotation = wp.transform.rotation

            agent_route.append(
                WaypointUnit(
                    x=location.x,
                    y=location.y,
                    z=location.z,
                    pitch=rotation.pitch,
                    yaw=rotation.yaw,
                    roll=rotation.roll,
                    speed=agent_speed
                )
            )

        # find type
        blueprint = random.choice(self._world.get_blueprint_library().filter('walker.pedestrian.*'))
        vehicle_object_info = ObjectInfo(
            model=blueprint.id,
            rolename='npc_walker',
            color=None,
            category='walker'
        )

        agent = BasicAgent(
            id=npc_id,
            route=agent_route,
            object_info=vehicle_object_info,
            mutable=True,
            trigger_time=random.uniform(0.0, 5.0)
        )
        return agent

    def generate_new_static(self, npc_id, scenario: ScenarioConfig):
        lanes_static = scenario.map_section.lanes_static

        selected_lane = random.choice(lanes_static)
        road_id = int(selected_lane.split('_')[0])
        lane_id = int(selected_lane.split('_')[1])
        selected_lane_length = estimate_lane_length(self._map, road_id, lane_id)
        selected_s = random.uniform(0.01, selected_lane_length - 0.05)
        selected_wp = self._map.get_waypoint_xodr(road_id, lane_id, selected_s)
        selected_wp = self._map.get_waypoint(selected_wp.transform.location)

        location = selected_wp.transform.location
        rotation = selected_wp.transform.rotation
        agent_route = [
            WaypointUnit(
                x=location.x,
                y=location.y,
                z=location.z,
                pitch=rotation.pitch,
                yaw=rotation.yaw,
                roll=rotation.roll,
                speed=0.0
            )
        ]

        # static.prop.trafficcone01
        blueprint = random.choice(self._world.get_blueprint_library().filter('*vehicle*'))
        static_object_info = ObjectInfo(
            model=blueprint.id,
            rolename='npc_static',
            color=None,
            category='static'
        )

        agent = BasicAgent(
            id=npc_id,
            route=agent_route,
            object_info=static_object_info,
            mutable=True
        )
        return agent

    def generate_new_scenario(self, scenario: ScenarioConfig, scenario_id: str):

        self._town = scenario.map_section.town
        self._setup()

        mutated_scenario = copy.deepcopy(scenario)
        mutated_scenario.id = scenario_id

        map_section = mutated_scenario.map_section
        ego_section = mutated_scenario.ego_section
        ego_agent = ego_section.agents[0]
        ego_route = [mutated_scenario.ego_section.agents[0].route[0],
                     mutated_scenario.ego_section.agents[0].route[-1]]
        mutated_scenario.ego_section.agents[0].route = ego_route

        lanes_static = map_section.lanes_static
        lanes_vehicle = map_section.lanes_vehicle
        nodes_walker = map_section.nodes_walker

        static_agents = list()
        if len(lanes_static) > 0:
            count = 0
            added = 0
            while added < self._npc_static_num:
                count += 1
                if count > 100:
                    break
                new_agent = self.generate_new_static(f'static_{added + 1}', mutated_scenario)
                if new_agent is None:
                    continue
                check_pass = self._location_check(new_agent, [ego_agent] + static_agents)
                # logger.debug(check_pass)
                if check_pass:
                    static_agents.append(new_agent)
                    added += 1
        mutated_scenario.static_section = StaticSection(static_agents)

        vehicle_agents = list()
        if len(lanes_vehicle) > 0:
            count = 0
            added = 0
            while added < self._npc_vehicle_num:
                count += 1
                if count > 100:
                    break
                new_agent = self.generate_new_vehicle(f'vehicle_{added + 1}', mutated_scenario)
                if new_agent is None:
                    continue
                check_pass = self._location_check(new_agent, [ego_agent] + static_agents + vehicle_agents)
                # logger.debug(check_pass)
                if check_pass:
                    vehicle_agents.append(new_agent)
                    added += 1
        mutated_scenario.vehicle_section = VehicleSection(vehicle_agents)

        walker_agents = list()
        if len(nodes_walker) > 0:
            count = 0
            added = 0
            while added < self._npc_walker_num:
                count += 1
                if count > 100:
                    break
                new_agent = self.generate_new_walker(f'walker_{added + 1}', mutated_scenario)
                if new_agent is None:
                    continue
                check_pass = self._location_check(new_agent, [ego_agent] + static_agents + vehicle_agents + walker_agents)
                if check_pass:
                    # logger.debug(f'Add walker_{added + 1}')
                    walker_agents.append(new_agent)
                    added += 1
        mutated_scenario.walker_section = WalkerSection(walker_agents)

        mutated_scenario.weather_section = self._generate_new_weather()

        logger.info(
            f'Generate new scenarios: Vehicle: {len(mutated_scenario.vehicle_section.agents)} Walker: {len(mutated_scenario.walker_section.agents)} Static: {len(mutated_scenario.static_section.agents)}')

        self._cleanup()

        return mutated_scenario

    def modify_current_vehicle(self, agent, scenario: ScenarioConfig):
        # change speed and trigger time only
        mutated_agent = copy.deepcopy(agent)

        tigger_time = mutated_agent.trigger_time
        new_time = random.gauss(tigger_time, 1.0)
        new_time = float(np.clip(new_time, 0.0, 6.0))
        mutated_agent.trigger_time = new_time

        agent_route = mutated_agent.route
        route_length = len(agent_route)
        if route_length < 1:
            return None

        selected_indices_num = min(3, route_length)
        random_indices = random.sample(range(route_length), selected_indices_num)

        curr_index = 0
        for i in range(len(agent_route)):
            if curr_index < len(random_indices) and i >= random_indices[curr_index]:
                curr_speed = agent_route[i].speed
                new_speed = random.gauss(curr_speed, 4.0)
                new_speed = float(np.clip(new_speed, MIN_VEHICLE_SPEED, MAX_VEHICLE_SPEED))
                agent_route[i].speed = new_speed
                curr_index += 1
        mutated_agent.route = agent_route
        return mutated_agent

    def modify_current_walker(self, agent, scenario: ScenarioConfig):
        # change speed only
        mutated_agent = copy.deepcopy(agent)

        tigger_time = mutated_agent.trigger_time
        new_time = random.gauss(tigger_time, 1.0)
        new_time = float(np.clip(new_time, 0.0, 6.0))
        mutated_agent.trigger_time = new_time

        agent_route = mutated_agent.route
        route_length = len(agent_route)
        if route_length < 1:
            return None

        selected_indices_num = min(3, route_length)
        random_indices = random.sample(range(route_length), selected_indices_num)

        curr_index = 0
        for i in range(len(agent_route)):
            if curr_index < len(random_indices) and i >= random_indices[curr_index]:
                curr_speed = agent_route[i].speed
                new_speed = random.gauss(curr_speed, 1.0)
                new_speed = float(np.clip(new_speed, MIN_WALKER_SPEED, MAX_WALKER_SPEED))
                agent_route[i].speed = new_speed
                curr_index += 1
        mutated_agent.route = agent_route
        return mutated_agent

    def modify_current_static(self, agent, scenario: ScenarioConfig):
        mutated_agent = copy.deepcopy(agent)
        # change position
        if random.random() > 0.5:

            agent_initial_waypoint = agent.get_initial_waypoint()
            agent_initial_transform = carla.Transform(
                location=carla.Location(
                    x=agent_initial_waypoint.x,
                    y=agent_initial_waypoint.y,
                    z=0.0
                ),
                rotation=carla.Rotation(
                    pitch=agent_initial_waypoint.pitch,
                    yaw=agent_initial_waypoint.yaw,
                    roll=agent_initial_waypoint.roll
                )
            )

            agent_waypoint = self._map.get_waypoint(agent_initial_transform.location)

            road_id = agent_waypoint.road_id
            lane_id = agent_waypoint.lane_id
            s = agent_waypoint.s
            lane_length = estimate_lane_length(self._map, road_id, lane_id)

            new_s = random.gauss(s, 2.0)
            new_s = float(np.clip(new_s, 0.02, lane_length - 0.02))

            new_waypoint = self._map.get_waypoint_xodr(road_id, lane_id, new_s)
            new_waypoint = self._map.get_waypoint(new_waypoint.transform.location)

            location = new_waypoint.transform.location
            rotation = new_waypoint.transform.rotation
            agent_route = [
                WaypointUnit(
                    x=location.x,
                    y=location.y,
                    z=location.z,
                    pitch=rotation.pitch,
                    yaw=rotation.yaw,
                    roll=rotation.roll,
                    speed=0.0
                )
            ]
            mutated_agent.route = agent_route
        else:
            mutated_agent = self.generate_new_static(mutated_agent.id, scenario)
        return mutated_agent

    def mutate_current_scenario(self, scenario: ScenarioConfig, scenario_id: str,max_weather_mutate=0.1):
        self._town = scenario.map_section.town
        self._setup()

        mutated_scenario = copy.deepcopy(scenario)
        mutated_scenario.id = scenario_id

        ego_route = [mutated_scenario.ego_section.agents[0].route[0],
                     mutated_scenario.ego_section.agents[0].route[-1]]
        mutated_scenario.ego_section.agents[0].route = ego_route

        static_agents = mutated_scenario.static_section.agents
        vehicle_agents = mutated_scenario.vehicle_section.agents
        walker_agents = mutated_scenario.walker_section.agents

        all_agents = static_agents + vehicle_agents + walker_agents

        mutated_static = 0
        if len(static_agents) > 0:
            for i in range(len(static_agents)):
                if random.random() > self._prob_mutation:
                    continue
                selected_agent = static_agents[i]
                npc_id = selected_agent.id
                count = 0
                while count < 50:
                    count += 1
                    new_agent = self.modify_current_static(selected_agent, scenario)
                    if new_agent is None:
                        continue
                    check_pass = self._location_check(new_agent, all_agents, ignore_ids=[npc_id])
                    if check_pass:
                        static_agents[i] = new_agent
                        mutated_static += 1
                        break
            mutated_scenario.static_section = StaticSection(static_agents)

        mutated_vehicle = 0
        if len(vehicle_agents) > 0:
            for i in range(len(vehicle_agents)):
                if random.random() > self._prob_mutation:
                    continue
                selected_agent = vehicle_agents[i]
                new_agent = self.modify_current_vehicle(selected_agent, scenario)
                vehicle_agents[i] = new_agent
                mutated_vehicle += 1
                # npc_id = selected_agent.id
                # count = 0
                # while count < 50:
                #     count += 1
                #     new_agent = self.modify_current_vehicle(selected_agent, scenario)
                #     if new_agent is None:
                #         continue
                #     check_pass = self._location_check(new_agent, all_agents, ignore_ids=[npc_id])
                #     if check_pass:
                #         vehicle_agents[i] = new_agent
                #         mutated_vehicle += 1
                #         break
            mutated_scenario.vehicle_section = VehicleSection(vehicle_agents)

        mutated_walker = 0
        if len(walker_agents) > 0:
            for i in range(len(walker_agents)):
                if random.random() > 0.5:
                    continue
                selected_agent = walker_agents[i]
                new_agent = self.modify_current_walker(selected_agent, scenario)
                walker_agents[i] = new_agent
                mutated_walker += 1
                # npc_id = selected_agent.id
                # count = 0
                # while count < 50:
                #     count += 1
                #     new_agent = self.modify_current_walker(selected_agent, scenario)
                #     if new_agent is None:
                #         continue
                #     check_pass = self._location_check(new_agent, all_agents, ignore_ids=[npc_id])
                #     if check_pass:
                #         walker_agents[i] = new_agent
                #         mutated_walker += 1
                #         break
            mutated_scenario.walker_section = WalkerSection(walker_agents)

        logger.info(f'Mutate current scenarios: Vehicle: {mutated_vehicle} Walker: {mutated_walker} Static: {mutated_static}')
        mutated_scenario.weather_section = self._weather_mutation(mutated_scenario.weather_section, max_weather_mutate)

        self._cleanup()

        need_run = False
        if (mutated_static + mutated_walker + mutated_vehicle) > 0:
            need_run = True

        return need_run, mutated_scenario

    def crossover(self, scenario1: ScenarioConfig, scenario2: ScenarioConfig, pc):
        self._town = scenario1.map_section.town
        self._setup()

        scenario1_c = copy.deepcopy(scenario1)
        scenario2_c = copy.deepcopy(scenario2)

        # crossover
        scenario1_vehicles = scenario1_c.vehicle_section.agents
        scenario2_vehicles = scenario2_c.vehicle_section.agents

        for i in range(len(scenario1_vehicles)):
            if random.random() > pc:
                continue

            target_vehicle_index = random.choice(range(len(scenario2_vehicles)))
            target_vehicle = copy.deepcopy(scenario2_vehicles[target_vehicle_index])
            source_vehicle = copy.deepcopy(scenario1_vehicles[i])

            # verify the location
            all_agents = scenario1_c.vehicle_section.agents + scenario1_c.walker_section.agents + scenario1_c.static_section.agents
            check_pass = self._location_check(target_vehicle, all_agents, ignore_ids=[source_vehicle.id])
            if check_pass:
                scenario1_vehicles[i] = copy.deepcopy(target_vehicle)
                scenario1_vehicles[i].id = source_vehicle.id

            all_agents = scenario2_c.vehicle_section.agents + scenario2_c.walker_section.agents + scenario2_c.static_section.agents
            check_pass = self._location_check(source_vehicle, all_agents, ignore_ids=[target_vehicle.id])
            if check_pass:
                scenario2_vehicles[target_vehicle_index] = copy.deepcopy(source_vehicle)
                scenario2_vehicles[target_vehicle_index].id = target_vehicle.id

            scenario1_c.vehicle_section.agents = scenario1_vehicles
            scenario2_c.vehicle_section.agents = scenario2_vehicles

        # walkers
        scenario1_walkers = scenario1_c.walker_section.agents
        scenario2_walkers = scenario2_c.walker_section.agents

        for i in range(len(scenario1_walkers)):
            if random.random() > pc:
                continue

            target_walker_index = random.choice(range(len(scenario2_walkers)))
            target_walker = copy.deepcopy(scenario2_walkers[target_walker_index])
            source_walker = copy.deepcopy(scenario1_walkers[i])

            # verify the location
            all_agents = scenario1_c.vehicle_section.agents + scenario1_c.walker_section.agents + scenario1_c.static_section.agents
            check_pass = self._location_check(target_walker, all_agents, ignore_ids=[source_walker.id])
            if check_pass:
                scenario1_walkers[i] = copy.deepcopy(target_walker)
                scenario1_walkers[i].id = source_walker.id

            all_agents = scenario2_c.vehicle_section.agents + scenario2_c.walker_section.agents + scenario2_c.static_section.agents
            check_pass = self._location_check(source_walker, all_agents, ignore_ids=[target_walker.id])
            if check_pass:
                scenario2_walkers[target_walker_index] = copy.deepcopy(source_walker)
                scenario2_walkers[target_walker_index].id = target_walker.id

            scenario1_c.walker_section.agents = scenario1_walkers
            scenario2_c.walker_section.agents = scenario2_walkers

        self._cleanup()

        return scenario1_c, scenario2_c

    def _weather_mutation(self, curr_weather, max_perturb = 0.1):
        weather_param = {}
        for weather_option, value_range in WEATHER_RANGE.items():
            max_perturb_val = max_perturb * (value_range[1] - value_range[0])
            weather_param[weather_option] = 2*(random.random()-0.5) * max_perturb_val + getattr(curr_weather, weather_option)
            if weather_param[weather_option] > value_range[1]:
                weather_param[weather_option] -= value_range[1]
            if weather_param[weather_option] < value_range[0]:
                weather_param[weather_option] += value_range[0]
        new_weather_section = WeatherSection(**weather_param)
        return new_weather_section

    def _generate_new_weather(self):
        weather_param = {weather_option: 2*(random.random()-0.5)*(value_range[1]-value_range[0])+value_range[0] for weather_option, value_range in WEATHER_RANGE.items()}
        new_weather_section = WeatherSection(**weather_param)
        print(getattr(new_weather_section,'cloudiness'))
        return new_weather_section