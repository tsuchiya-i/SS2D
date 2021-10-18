import numpy as np
import gym

import gym_pathplan

from keras.callbacks import TensorBoard
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
import matplotlib.pyplot as plt

import tensorflow as tf
from playsound import playsound

def main():
    ENV_NAME = 'Simple-v0'
    
    # Get the environment and extract the number of actions.
    env = gym.make(ENV_NAME)
    env.reset()
    np.random.seed(123)
    env.seed(123)
    assert len(env.action_space.shape) == 1
    nb_actions = env.action_space.shape[0]
    
    # Next, we build a very simple model.
    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    #actor.add(Dense(512))
    #actor.add(Activation('relu'))
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
    #x = Dense(512)(x)
    #x = Activation('relu')(x)
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
                      memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                      random_process=random_process, gamma=.99, target_model_update=1e-3)
    agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])
    
    #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
    tensorboard_callback = TensorBoard(log_dir="logs", histogram_freq=1)

    try:
        agent.load_weights('./now_train_weight/ddpg_{}_weights.h5f'.format(ENV_NAME))
        print("find weights-file")
    except:
        print("not found weights-file")
    
    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    
    nb_steps_ = 50000000
    nb_max_episode_steps_ = 1500
    plt_num = int(nb_steps_/nb_max_episode_steps_)
    
    train_history = agent.fit(env, nb_steps=nb_steps_, visualize=True, verbose=1, nb_max_episode_steps=nb_max_episode_steps_)
    
    # After training is done, we save the final weights.
    #agent.save_weights('./now_train_weight/ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
    agent.save_weights('./now_train_weight/ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
    
    train_rewards = train_history.history['episode_reward']
    episode_step = train_history.history['nb_episode_steps']
    total_step = train_history.history['nb_steps']
    rwd_plt = np.array(train_rewards)
    episode_step_plt = np.array(episode_step)
    total_step_plt = np.array(total_step)
    csvnp = np.stack([rwd_plt, episode_step_plt, total_step_plt], 1)
    try:
        load = np.loadtxt('./now_train_weight/ddpg_Simple-v0.csv')
        print("find csvfile")
        csvnp = np.concatenate([load, csvnp])
    except:
        print("not found csvfile")
    
    np.savetxt('./now_train_weight/ddpg_Simple-v0.csv',csvnp)
    
    #playsound("robot_s.mp3")

    # reward gragh show
    #plt.plot(total_step_plt, rwd_plt, marker=",", color = "blue", linestyle = "-")
    #plt.plot(history.history['nb_episode_steps'], label='nb_episode_steps')
    plt.plot(train_rewards, label='episode_reward')
    plt.legend()

    plt.show()
    
    # Finally, evaluate our algorithm for 5 episodes.
    agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=1500)

main()
