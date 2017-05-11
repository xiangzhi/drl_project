#!/usr/bin/env python

import gym
import time

from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input, Lambda,
                          Permute)
import keras.layers
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.initializers import RandomUniform, VarianceScaling

from deep.ddpg import DDPGAgent
from deep.preprocessors import HistoryPreprocessor, PendulumPreprocessor, PreprocessorSequence, KerasPreprocessor
from deep.policy import LinearDecayGreedyEpsilonPolicy, UniformRandomPolicy
#from deeprl_hw2.action_replay_memory import ActionReplayMemory
from deep.action_replay_memory import ActionReplayMemory
from deep.objectives import huber_loss
from deep.utils import memory_burn_in
from deep.noise_generator import OU_Generator

import numpy as np
import json
import sys

import tensorflow as tf
from keras import backend as K
from keras.layers.normalization import BatchNormalization



def create_actor_model(hist_window,state_size,action_dim,model_name):
    pendulum_input = Input(shape=(state_size,hist_window), name='pendulum_input')
    merged_layer = Flatten()(pendulum_input)
    merged_layer = Dense(16, activation='relu')(merged_layer)
    merged_layer = Dense(16, activation='relu')(merged_layer)
    merged_layer = Dense(16, activation='relu')(merged_layer)
    output_layer = Dense(action_dim, activation='tanh')(merged_layer)
    scaled_output_layer = Lambda(lambda x:x*2)(output_layer)

    return Model(inputs=pendulum_input, outputs=scaled_output_layer, name=model_name)


def create_critic_model(hist_window,state_size,action_dim,model_name):

    #the current pendulum state
    pendulum_input = Input(shape=(state_size,hist_window), name='pendulum_input')
    merged_layer = Flatten()(pendulum_input)
    merged_layer = Dense(32, activation='relu')(merged_layer)

    #merge in the action input after the first hidden layer
    action_input = Input(shape=(action_dim,),name='action_input')
    action_layer = action_input
    merged_layer = keras.layers.concatenate([merged_layer, action_layer])
    merged_layer = Dense(32, activation='relu')(merged_layer)
    merged_layer = Dense(32, activation='relu')(merged_layer)
    output_layer = Dense(action_dim, activation='linear')(merged_layer)

    return Model(inputs=[pendulum_input,action_input], outputs=output_layer, name=model_name)

def main():


    #let the user decide on the type environment to work in
    #the default settings
    #env_name = "Pendulum-v0"
    #env_name = "MountainCarContinuous-v0"
    env_name = "LunarLanderContinuous-v2"
    run_name = '{}-{}'.format(env_name,0)

    if(len(sys.argv) > 1):
        env_name = sys.argv[1]
    if(len(sys.argv) > 2):
        run_name = sys.argv[2]

    #create the environment
    env = gym.make(env_name)

    #initialize tensorflow
    sess = tf.Session()
    K.set_session(sess)

    #get the state_size
    state_size = env.observation_space.shape[0]
    #set the batch size
    batch_size = 64
    #get the size of the action
    action_dim = env.action_space.shape[0]

    #burn in memory
    memory_size = int(1e6)
    memory_burn_in_num = int(memory_size*0.05)# 5% of the whole memory

    #the history
    history_size = 1

    #the learning rate
    actor_learning_rate = 0.01
    critic_learning_rate = 0.001
    
    #set the preprocessors
    history_prep = HistoryPreprocessor(history_size)
    pendulum_prep  = PendulumPreprocessor()
    keras_prep = KerasPreprocessor(['pendulum_input'])
    preprocessors = PreprocessorSequence([history_prep,pendulum_prep,keras_prep]) #from left to right

    #create noise generator
    noise_generator = OU_Generator(np.zeros(action_dim))

    #generate the two models
    actor_model = create_actor_model(history_size,state_size,action_dim,'actor_model_{}'.format(run_name,))
    actor_model.summary()
    critic_model = create_critic_model(history_size,state_size,action_dim,'critic_model_{}'.format(run_name,))
    critic_model.summary()

    #set the optimizer and loss function for the critic
    optimizer = Adam(lr=critic_learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    loss_func = huber_loss
   
    #create memory and start the burn in
    memory = ActionReplayMemory(memory_size,4)
    memory_burn_in(env,memory,preprocessors,memory_burn_in_num)

    #initialize and set the noise and compile
    agent = DDPGAgent(sess,actor_model, critic_model, preprocessors, memory, 0.99, batch_size,run_name)
    agent.set_noise_generator(noise_generator)
    agent.compile(optimizer, loss_func, actor_learning_rate,1,action_dim)

    #start the fitting process
    agent.fit(env,1000000,100000)

if __name__ == '__main__':
    main()
