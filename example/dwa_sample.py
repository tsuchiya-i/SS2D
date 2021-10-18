#coding:utf-8

import gym
import numpy as np
import time
import ss2d

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

np.set_printoptions(suppress=True)
env = gym.make('ss2d-v0')
observation = env.reset()
start=env.world_time
print("start : "+str(start))
t_rwd = 0
action = [0.0,0.0]

for i in range(10000):
    stime = time.time()
    if i%1==0:
        env.render()
    #action = env.action_space.sample()
    observation, reward, done,  _ = env.step(action)
    #action  = dwa(observation)
    action = [0.0,0.0]
    ctime = time.time()-stime
    #print("check_time="+str(ctime))

    #print("=========================")
    t_rwd += reward
    #print(observation)
    #print(now_state)
    #print(observation)
    #print(world_time)
    #print(observation['lidar'])
    ltime = time.time()-stime
    #print("loop_time="+str(ltime))
    #print("=========================")
    #time.sleep(0.1)

    if done:
        print("finish : "+str(env.world_time))
        env.reset()
        #break
