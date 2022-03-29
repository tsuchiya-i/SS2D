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

env = model.get_env()

del model # remove to demonstrate saving and loading

model = TD3.load("td3_pendulum")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
