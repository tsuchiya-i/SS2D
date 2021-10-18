#coding:utf-8

import gym
import numpy as np
import time
import ss2d

#np.set_printoptions(suppress=True)

env = gym.make('ss2d-v0')
observation = env.reset()


#入力・出力関連
print("----------------------------")
print(env.observation_space)
print("----------------------------")
print(env.action_space)
print("----------------------------")
print(env.action_space.high)
print("----------------------------")
print(env.action_space.low)
print("----------------------------")

action = [0.0,0.0]

for i in range(10000):
    env.render()
    action = env.action_space.sample() #random_action
    #action = [0.8,0.0] #直進
    observation, reward, done,  _ = env.step(action)

    #print(observation)

    if done:
        env.reset() #衝突ORゴールで環境リセット
        #break
