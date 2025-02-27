import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import odeint

from utils import generate_road
from plant_NonlinearQuarterCar import quarter_car_dynamics

class QuarterCarEnv(gym.Env):
    def __init__(self, t_step):
        super(QuarterCarEnv, self).__init__()
        self.t_step = t_step
        self.max_force = 1000
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.reset()

    def step(self, action):
        u = action[0]
        self.u_track[self.current_step] = u
        self.x_controlled = odeint(quarter_car_dynamics, self.x_controlled, [0, self.t_step], args=(self.z_r[self.current_step], self.max_force*u))[-1]
        self.x_track[self.current_step, :] = self.x_controlled
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= len(self.t_sim)
        reward = -(self.x_controlled[0]/0.15)**2 
        return self.x_controlled, reward, terminated, truncated, {}

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self.t_sim, self.z_r = generate_road(self.t_step)
        self.x_controlled = np.zeros(4)
        self.x_track = np.zeros((len(self.t_sim), len(self.x_controlled)))
        self.u_track = np.zeros(len(self.t_sim))
        return self.x_controlled, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    