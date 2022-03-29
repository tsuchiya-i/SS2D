import gym
import numpy as np

import ss2d

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

env = gym.make('ss2d-v0')

# The noise objects for TD3
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

policy_kwargs = dict(net_arch=[256, 128, 64])
model = TD3("MlpPolicy", env, tensorboard_log="./logs", policy_kwargs=policy_kwargs, action_noise=action_noise, verbose=1)
model.learn(total_timesteps=100000, callback=env.render, log_interval=1)
model.save("td3_pendulum")

print("done")
