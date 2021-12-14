#coding:utf-8

import sys
import os
import gym
import ss2d
import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Input, Concatenate

check_weight_path = "./trained_weight/ddpg_weights_actor.h5f.index"
weight_path = "./trained_weight/ddpg_weights_actor.h5f"

args = sys.argv
if len(args)>1:
    args_str = args[1]
    if args_str[0] != "." and args_str[0] != "/":
        args_str = "./"+args_str
    if args_str[-1] != "/":
        args_str = args_str + "/"
    check_weight_path = args_str + "ddpg_actor.h5f.index"
    weight_path = args_str + "ddpg_actor.h5f"

# Get the environment and extract the number of actions.
env = gym.make('ss2d-v0')
env.reset()
np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(256))
actor.add(Activation('relu'))
actor.add(Dense(128))
actor.add(Activation('relu'))
actor.add(Dense(64))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('tanh'))
print(actor.summary())

if os.path.exists(check_weight_path):
    actor.load_weights(weight_path)
    print("find weights-file")
else:
    print("not found weights-file")

observation = env.reset()
while True:
    env.render()
    action = actor.predict(observation.reshape(1,1,len(observation)))
    observation, reward, done,  GoalOrNot = env.step(action.reshape(2))
    if done or env.step_count>150:
        env.reset()
