import numpy as np
from typing import Tuple
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs import HighwayEnv, CircularLane
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle


class ExitEnv(HighwayEnv):
    """
    """
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "ExitObservation",
                "vehicles_count": 15,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "clip": False
            },
            "action": {
                "type": "DiscreteMetaAction",
                # "longitudinal": False,
                # "lateral": True,
            },
            "lanes_count": 3,
            "collision_reward": 0.1,
            "high_speed_reward": 0.1,
            "right_lane_reward": 0,
            "goal_reward": 1,
            "vehicles_count": 3,
            "vehicles_density": 1.5,
            "controlled_vehicles": 1,
            "duration": 200,  # [s],
            "simulation_frequency": 5,
            "scaling": 5
        })
        return config

    def _reset(self) -> None:
        # TODO: add lower speed also
        # high_speed
        vehicles_density = np.random.uniform(0.3, 0.4)
        avg_speed = 30.
        if np.random.random() < 0.5:  # low_speed
            vehicles_density = np.random.uniform(0.5, 0.7)
            avg_speed = 15.
        self.config.update({"vehicles_density": vehicles_density, })
        self.config.update({"avg_speed": avg_speed, })

        self._create_road()
        self._create_vehicles()

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, terminal, info = super().step(action)
        info.update({"is_success": self._is_success()})
        return obs, reward, terminal, info

    def _create_road(self, road_length=1600, exit_position=1400, exit_length=100) -> None:
        init_distance = 5
        net = RoadNetwork.straight_road_network(self.config["lanes_count"], start=0,
                                                length=init_distance, nodes_str=("-1", "0"))
        net = RoadNetwork.straight_road_network(self.config["lanes_count"], start=0,
                                                length=exit_position, nodes_str=("0", "1"))
        net = RoadNetwork.straight_road_network(self.config["lanes_count"] + 1, start=exit_position,
                                                length=exit_length, nodes_str=("1", "2"), net=net)
        net = RoadNetwork.straight_road_network(self.config["lanes_count"], start=exit_position+exit_length,
                                                length=road_length-exit_position-exit_length,
                                                nodes_str=("2", "3"), net=net)
        for _from in net.graph:
            for _to in net.graph[_from]:
                for _id in range(len(net.graph[_from][_to])):
                    net.get_lane((_from, _to, _id)).speed_limit = 26 - 3.4 * _id
        exit_position = np.array([exit_position + exit_length, self.config["lanes_count"] * CircularLane.DEFAULT_WIDTH])
        radius = 150
        exit_center = exit_position + np.array([0, radius])
        lane = CircularLane(center=exit_center,
                            radius=radius,
                            start_phase=3*np.pi/2,
                            end_phase=2*np.pi,
                            forbidden=True)
        net.add_lane("2", "exit", lane)

        self.road = Road(network=net,
                         np_random=self.np_random,
                         record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        self.controlled_vehicles = []
        lane_id_ego = np.random.randint(2)
        for _ in range(self.config["controlled_vehicles"]):
            vehicle = self.action_type.vehicle_class.create_random(self.road,
                                                                   speed=20,
                                                                   lane_from="-1",
                                                                   lane_to="0",
                                                                   lane_id=lane_id_ego,
                                                                   spacing=self.config["ego_spacing"])
            vehicle.SPEED_MIN = 18
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        for _ in range(self.config["vehicles_count"]):
            lanes = np.arange(self.config["lanes_count"])
            lane_id = self.road.np_random.choice(lanes, size=1,
                                                 p=lanes / lanes.sum()).astype(int)[0]
            lane = self.road.network.get_lane(("0", "1", lane_id))
            vehicle = vehicles_type.create_random(self.road,
                                                  lane_from="0",
                                                  lane_to="1",
                                                  lane_id=lane_id,
                                                  speed=lane.speed_limit,
                                                  spacing=1 / self.config["vehicles_density"],
                                                  ).plan_route_to("3")
            vehicle.enable_lane_change = False
            self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        lane_index = self.vehicle.target_lane_index if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        reward = self.config["collision_reward"] * self.vehicle.crashed \
                 + self.config["goal_reward"] * self._is_success() \
                 + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1) \
                 + self.config["right_lane_reward"] * lane_index[-1]

        reward = utils.lmap(reward,
                          [self.config["collision_reward"], self.config["goal_reward"]],
                          [0, 1])
        reward = np.clip(reward, 0, 1)
        return reward

    def _is_success(self):
        lane_index = self.vehicle.target_lane_index if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index
        goal_reached = lane_index == ("1", "2", self.config["lanes_count"]) or lane_index == ("2", "exit", 0)
        return goal_reached

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        print(self.vehicle.position[0])
        return self.vehicle.crashed or self.steps >= self.config["duration"] or self.vehicle.position[0] > 1450


# class DenseLidarExitEnv(DenseExitEnv):
#     @classmethod
#     def default_config(cls) -> dict:
#         return dict(super().default_config(),
#                     observation=dict(type="LidarObservation"))


register(
    id='exit-v0',
    entry_point='highway_env.envs:ExitEnv',
)
