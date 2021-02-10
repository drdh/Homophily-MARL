import numpy as np
import torch as th

from .agent import HarvestAgent  # HARVEST_VIEW_SIZE
from .constants import HARVEST_N10_MAP
from .map_env import MapEnv, ACTIONS

APPLE_RADIUS = 2

# Add custom actions to the agent
ACTIONS['FIRE'] = 5  # length of firing range

SPAWN_PROB = [0, 0.005, 0.02, 0.05] # default


class HarvestEnv(MapEnv):

    def __init__(self, ascii_map=HARVEST_N10_MAP, num_agents=1, render=False, seed=None, episode_limit=100,is_replay=False, view_size=7, map="default", extra_args=None):

        if map=="default10":
            ascii_map = HARVEST_N10_MAP
            self.SPAWN_PROB = [0, 0.05, 0.08, 0.1]

        print("map difficulty: {}".format(map))
        for row in ascii_map:
            print(row)

        self.view_size = view_size
        super().__init__(ascii_map, num_agents, render, episode_limit=episode_limit,is_replay=is_replay,env_name='harvest', extra_args=extra_args)

        self.apple_points = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == 'A':
                    self.apple_points.append([row, col])

        self.simplified_color_map = self.color_map.copy()
        for k in self.simplified_color_map.keys():
            if k == 'A':
                self.simplified_color_map[k] = [0, 255, 0]
            elif k == '@' or k.isdigit():
                self.simplified_color_map[k] = [0, 0, 255]
            else:
                self.simplified_color_map[k] = [0, 0, 0]

        self.color_map_obs = self.color_map
        if extra_args['obs_color'] == 'simplified':
            self.color_map_obs = self.simplified_color_map

    @property
    def action_space(self):
        agents = list(self.agents.values())
        return agents[0].action_space

    @property
    def observation_space(self):
        agents = list(self.agents.values())
        return agents[0].observation_space

    def setup_agents(self):
        map_with_agents = self.get_map_with_agents()

        for i in range(self.num_agents):
            agent_id = 'agent-' + str(i)
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation()
            grid = map_with_agents
            agent = HarvestAgent(agent_id, spawn_point, rotation, grid, self.view_size)
            # grid = util.return_view(map_with_agents, spawn_point,
            #                         HARVEST_VIEW_SIZE, HARVEST_VIEW_SIZE)
            # agent = HarvestAgent(agent_id, spawn_point, rotation, grid)
            self.agents[agent_id] = agent

    def custom_reset(self):
        """Initialize the walls and the apples"""
        for apple_point in self.apple_points:
            self.world_map[apple_point[0], apple_point[1]] = 'A'

    def custom_action(self, agent, action):
        agent.fire_beam('F')
        updates = self.update_map_fire(agent.get_pos().tolist(),
                                       agent.get_orientation(),
                                       ACTIONS['FIRE'], fire_char='F')
        return updates

    def custom_map_update(self):
        "See parent class"
        # spawn the apples
        new_apples = self.spawn_apples()
        self.update_map(new_apples)

    def spawn_apples(self):
        """Construct the apples spawned in this step.

        Returns
        -------
        new_apple_points: list of 2-d lists
            a list containing lists indicating the spawn positions of new apples
        """

        new_apple_points = []
        for i in range(len(self.apple_points)):
            row, col = self.apple_points[i]
            # apples can't spawn where agents are standing or where an apple already is
            # [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
            if [row, col] not in self.agent_pos and self.world_map[row, col] != 'A':
                num_apples = 0
                for j in range(-APPLE_RADIUS, APPLE_RADIUS + 1): #
                    for k in range(-APPLE_RADIUS, APPLE_RADIUS + 1):
                        if j**2 + k**2 <= APPLE_RADIUS:
                            x, y = self.apple_points[i]
                            if 0 <= x + j < self.world_map.shape[0] and \
                                    self.world_map.shape[1] > y + k >= 0:
                                symbol = self.world_map[x + j, y + k]
                                if symbol == 'A':
                                    num_apples += 1

                spawn_prob = self.SPAWN_PROB[min(num_apples, 3)] # [0, 0.005, 0.02, 0.05]
                rand_num = np.random.rand(1)[0]
                if rand_num < spawn_prob:
                    new_apple_points.append((row, col, 'A'))
        return new_apple_points

    def count_apples(self, window):
        # compute how many apples are in window
        unique, counts = np.unique(window, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        num_apples = counts_dict.get('A', 0)
        return num_apples
