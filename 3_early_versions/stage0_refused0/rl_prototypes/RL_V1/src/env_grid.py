import gym
from gym import spaces
import numpy as np
import pandas as pd

class GridEnv(gym.Env):
    def __init__(self, data_path):
        super(GridEnv, self).__init__()

        self.data = pd.read_csv(data_path)
        self.current_step = 0

        # ----- Define State Space -----
        self.state_dim = self.data.shape[1] - 1
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )

        # ----- Define Action Space -----
        # Example: dispatch scaling factor
        self.action_space = spaces.Box(
            low=np.array([0.0]),
            high=np.array([1.0]),
            dtype=np.float32
        )

    def reset(self):
        self.current_step = 0
        return self._get_state()

    def step(self, action):
        reward = self._calculate_reward(action)

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        next_state = self._get_state()
        return next_state, reward, done, {}

    def _get_state(self):
        row = self.data.iloc[self.current_step].values
        return row[:-1].astype(np.float32)

    def _calculate_reward(self, action):
        row = self.data.iloc[self.current_step]

        carbon_penalty = row["carbon_intensity"] * action[0]
        demand_gap_penalty = abs(row["supply_demand_gap"])

        reward = - (carbon_penalty + demand_gap_penalty)
        return reward