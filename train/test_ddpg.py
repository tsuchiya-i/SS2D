#coding:utf-8

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import gym
import gym_pathplan
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
import time
import random

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
import matplotlib.pyplot as plt
import tensorflow as tf

import rl
print(rl)

import keyboard

input_mode = 1#0:人認識なし 1:人認識あり

ENV_NAME = 'Simple-v0'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
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

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(256)(x)
x = Activation('relu')(x)
x = Dense(128)(x)
x = Activation('relu')(x)
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())


# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)

agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=10, nb_steps_warmup_actor=10,
                  random_process=random_process, gamma=.99, target_model_update=1e-3)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

try:
    agent.load_weights('./now_train_weight/human_simple1/ddpg_{}_weights.h5f'.format(ENV_NAME))
    #agent.load_weights('./now_train_weight/simple7/ddpg_{}_weights.h5f'.format(ENV_NAME))
    print("find weights-file")
except:
    print("not found weights-file")


def dwa(observe_data):
    div = 9
    a = 40#30 angle 
    b = 1#1 obstacle_cost
    min_c = 10000
    min_i = 4
    ow = -0.6
    od = np.array([observe_data[5], observe_data[6], observe_data[7],observe_data[8], observe_data[0], observe_data[1], observe_data[2], observe_data[3], observe_data[4]])

    #平滑化
    ood = od
    od[0] = (ood[0]+ood[1]*0.5)/1.5
    for i in range(1,8):
        od[i] = (ood[i-1]+ood[i]+ood[i+1])/3
    od[8] = (ood[8]+ood[7]*0.5)/1.5

    for i in range(div):
        w = ow - ow/4*i
        dist_c = 0
        for l in range(9):
            if l==i:
                dist_c -= od[l]*3
            elif l==i+1 or l==i-1:
                dist_c -= od[l]*2
            elif l==i+2 or l==i-2:
                dist_c -= od[l]*1
            else:
                dist_c -= od[l]*0.5

        if min_c > a*abs(observe_data[19]+ow/div*i) + b*dist_c:
            min_c = a*abs(observe_data[19]+ow/div*i) + b*dist_c
            min_n = i
    #print(max(od))
    #print(min_c)
    #print(observe_data[10])
    vel = max(od)/10-0.1
    if min(od[3:6])<0.5:
        vel = 0.0
    predict_action = np.array([vel, ow-ow/4*min_n])
    return predict_action




# Finally, evaluate our algorithm for 5 episodes.
t_rwd = 0

success_rate = 0
ep_count = 0
step_count = 0

fig, ax = plt.subplots(1, 1)
ax.set_ylim(math.radians(-50), math.radians(50))
x = []
y = []
observation = env.reset()

v1 = [0.2]*30
v2 = [0.2]*30
v_count = 0
ex_count = 0
safe_count = 0

finish_count = [0]*4

#for i in range(100):
while True:
    env.render()
    #action = env.action_space.sample()
    action = agent.forward(observation)
    #if keyboard.read_key() == "d":
    #    action = dwa(observation)
    #action = [1.0,-0.0]
    v_count = v_count%30
    
    if (sum(v1)/len(v1)) < 0.001 and (sum(v2)/len(v2)) < 0.001 and ex_count == 0 and safe_count > 60:# and min(observation[9:18])==10:
        ex_count += 1
        omega = min(0.2,observation[-1])
        omega = max(-0.2,observation[-1])
        omega *= 0#random.randint(0,1)
        if True:#omega == 0:
            ex_action = dwa(observation)
        else:
            ex_action = [0, omega]
    if ex_count > 0:
        action = ex_action
        #print("\rEX",end="")
        ex_count += 1
        if ex_count == 30:
            safe_count = 0
            ex_count = 0
    else:
        #print("\rON",end="")
        safe_count += 1
    v1[v_count] = action[0]
    v2[v_count] = action[1]
    #print(observation[-1])
    v_count += 1

    observation, reward, done,  GoalOrNot = env.step(action)
    t_rwd += reward
    #print(observation[-2])
    #print(reward)
    step_count += 1
    x.append(step_count)
    y.append(action[1])
    if(len(x)>50):
        x.pop(0)
        y.pop(0)
    plt.xlim(x[0],x[0]+50)
    line, = ax.plot(x, y, color='blue')
    #plt.pause(0.01)
    # グラフをクリア
    line.remove()

    if done or (step_count>10000):
        x.clear()
        y.clear()
        #print("finish : "+str(env.world_time))
        if observation[-2] <= 0.9:
            success_rate += 1
        ep_count += 1
        if 10 > reward >-10:
            finish_judge = "Time Over"
            finish_count[0] += 1
        elif reward == -25:
            finish_judge = "Static obstacle collision"
            finish_count[1] += 1
        elif reward == 25:
            finish_judge = "Goal"
            finish_count[2] += 1
        else:
            finish_judge = "Dynamic obstacle collision"
            finish_count[3] += 1

        print("{}回目:".format(ep_count)+finish_judge)
        print("{}%".format(success_rate/ep_count *100))
        env.reset()
        step_count = 0
        if ep_count == 27:
            finish_count = np.array(finish_count)
            finish_count = finish_count/sum(finish_count)*100
            print("Time, Static, Goal, Dynamic")
            print(finish_count)
            break

