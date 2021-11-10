#coding:utf-8
import numpy as np
import gym
import ss2d
from time import time

env = gym.make('ss2d-v0')
observation = env.reset()

#input,output property
print("----------------------------")
print("observation_space: ",env.observation_space)
print("action_space: ",env.action_space)
print("action_space.high: ", env.action_space.high)
print("action_space.low: ", env.action_space.low)
print("----------------------------")

for i in range(10000): #10000step
    stime = time()
    env.render()
    #print(time()-stime)
    action = env.action_space.sample() #random_action
    action = [0.1,0.0] #straight
    observation, reward, done,  _ = env.step(action)
    if done:
        env.reset()

env.close()
