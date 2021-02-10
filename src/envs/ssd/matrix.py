"""Base map class that defines the rendering process
"""

import random

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
import numpy as np
from ..multiagentenv import MultiAgentEnv
import os
import time
import math
from types import SimpleNamespace as SN


class MatrixEnv(MultiAgentEnv):

    def __init__(self, num_agents=1, render=False, seed=None, episode_limit=100, is_replay=False, map="default",
                 extra_args=None):
        self.extra_args = SN(**extra_args)
        self.num_agents = num_agents

        self.agents = {}

        self.n_actions = 2  # [C,D]
        self.n_agents = self.num_agents
        self.episode_limit = episode_limit
        self._episode_steps = 0

        self.G_b = 1  # contribution
        self.G_c = 0.4  # cost

        self.rewards = None
        self.actions = None
        self.is_replay = is_replay

        self.map_dims = (self.num_agents + 2, 1 + 2)  # include walls

    # ************************************** pymarl *********************************************************************

    def step(self, actions):  # action [0,1,1,0,1]
        """A single environment step. Returns reward, terminated, info."""
        if self.is_replay:
            pass

        actions = [int(a) for a in actions]
        actions = np.array(actions)
        n_C = np.sum(actions == 0)
        n_D = np.sum(actions == 1)
        self.actions = actions

        reward = np.zeros(self.num_agents, dtype=float)
        reward[(actions == 0)] = n_C * self.G_b / self.num_agents - self.G_c
        reward[(actions == 1)] = n_C * self.G_b / self.num_agents

        if self.rewards is None:
            self.rewards = reward
        else:
            self.rewards += reward

        self._episode_steps += 1
        if self._episode_steps >= self.episode_limit:
            terminated = True
        else:
            terminated = False

        info = {}
        if terminated:
            if self.is_replay:
                pass

            collective_return = 0.0
            equality_metric = 1.0
            if not self.rewards is None:
                collective_return = self.rewards.sum()
                if self.rewards.sum() != 0:
                    equality_metric = 1 - (np.abs(self.rewards.reshape(1, -1) - self.rewards.reshape(-1, 1)).sum()) / (
                            2 * len(self.rewards) * np.abs(self.rewards).sum())

            info = {
                "collective_return": collective_return,
                "equality_metric": equality_metric,
            }
        info["clean_num"] = (actions == 0).astype(float)
        info["apple_den"] = np.zeros(self.num_agents, dtype=float) + np.mean((actions == 0).astype(float))
        return reward, terminated, info

    def get_agent_pos(self):
        return np.zeros((self.num_agents, 2), dtype=float)

    def get_agent_orientation(self):
        return np.zeros((self.num_agents, 2), dtype=float)

    def get_obs(self):
        """Returns all agent observations in a list.
        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        agents_obs = [self.get_obs_agent(i) for i in range(self.num_agents)]
        return agents_obs

    def get_own_feature_size(self):
        return None

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id.
        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        rgb_arr = np.zeros((self.map_dims + (3,)))
        if self.actions is not None:
            rgb_arr[1:-1,1][self.actions == 0, 1] = 1
            rgb_arr[1:-1,1][self.actions == 1, 2] = 1
        return rgb_arr.transpose(2, 0, 1)

    def get_obs_size(self):
        return self.get_obs_agent(0).shape

    def get_state(self):
        """Returns the global state.
        NOTE: This functon should not be used during decentralised execution.
        """
        rgb_arr = np.zeros((self.map_dims + (3,)))
        if self.actions is not None:
            rgb_arr[1:-1,1][self.actions == 0, 1] = 1
            rgb_arr[1:-1,1][self.actions == 1, 2] = 1
        return rgb_arr.transpose(2, 0, 1)

    def get_state_size(self):
        """Returns the size of the global state."""
        return self.get_state().shape

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        avail_action = [self.get_avail_agent_actions(i) for i in range(self.num_agents)]
        return avail_action

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        return [1] * self.n_actions
        if agent_id in {0, 1}:
            return [1] * (self.n_actions - 1) + [0]
        else:
            return [1] * self.n_actions
        # TODOSSD: modify available actions

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.n_actions

    def reset(self):
        """Reset the environment. Required after each full episode.
        Returns initial observations and states.
        """
        self._episode_steps = 0
        self.rewards = None
        self.actions = None
        return self.get_obs(), self.get_state()

    def render(self):
        pass

    def close(self):
        return

    def seed(self):
        return None

    def save_replay(self):
        pass

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit,
                    "units_type_id": self.get_units_type_id(),
                    "own_feature_size": self.get_own_feature_size(),
                    "state_dims": (self.get_state().shape[1], self.get_state().shape[2]),
                    "obs_dims": (self.get_obs_agent(0).shape[1], self.get_obs_agent(0).shape[2])
                    }
        return env_info

    def get_stats(self):
        return {}
