#coding:utf-8
from pynput import keyboard
import gym
import numpy as np
import ss2d

#################################
### keyboard controll program ###
###    cursor movement keys   ###
#################################


env = gym.make('ss2d-v0')
env.reset()

def on_press(key):
    if keyboard.Key.left == key:
        gym_run(0,0.3)
    if keyboard.Key.right == key:
        gym_run(0,-0.3)
    if keyboard.Key.down == key:
        gym_run(-0.2,0)
    if keyboard.Key.up == key:
        gym_run(0.5,0)

def on_release(key):
    if key == keyboard.Key.esc:
        return False

def gym_run(x,z):
    global env
    env.render()
    action = [x,z]
    observation, reward, done,  _ = env.step(action)
   
    if done:
        env.reset() 

with keyboard.Listener(on_press=on_press,on_release=on_release) as listener:
    listener.join()
