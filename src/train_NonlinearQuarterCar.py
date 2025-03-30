import numpy as np
from env_NonlinearQuarterCar import QuarterCarEnv
import os
import time

from stable_baselines3 import PPO
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from utils import getLQR_K

def load_replay_buffer_w_lQR(env, model, t_step, num_itr=10000):
    model.replay_buffer.handle_timeout_termination = False
    obs = env.reset()[0]
    K = getLQR_K(t_step)
    experiences = []
    # Store experiences in an array
    for i in range(len(env.t_sim)):
        action = -np.dot(K, obs)/env.max_force
        next_obs, reward, terminated, truncated, infos = env.step(action)
        experiences.append((obs, action, reward, next_obs, truncated, infos))
        obs = next_obs
        if truncated or terminated:
            obs = env.reset()[0]
    # Iterate through the array and store in the replay buffer
    for i in range(num_itr):
        for exp in experiences:
            model.replay_buffer.add(obs=exp[0], action=exp[1], reward=exp[2], next_obs=exp[3], done=exp[4], infos=exp[5])
    return model

# Define simulation parameters
t_step = 0.01

# Create the environment
env = QuarterCarEnv(t_step)

sim_name = "TD3_QuarterCar_quadraticZs_LQRBuff"
models_dir = "models/" + sim_name + "_" + str(time.time())
logdir = "logs/" + sim_name + "_" + str(time.time())

if not os.path.exists(logdir):
    os.makedirs(logdir)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Create the PPO agent
# model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir, device='cuda')

# Create TD3 agent
model = TD3('MlpPolicy', env, verbose=1, tensorboard_log=logdir, device='cuda')

# Load replay buffer with LQR
model = load_replay_buffer_w_lQR(env, model, t_step)

print("Replay buffer loaded")

# Train the agent
TIMESTEPS = 100000
for i in range(1, 5000):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=sim_name)
    model.save(f"{models_dir}/{TIMESTEPS*i}")