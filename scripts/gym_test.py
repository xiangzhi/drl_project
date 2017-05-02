#!/usr/bin/env python

import gym
import time

from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input, Lambda,
                          Permute)
import keras.layers
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.initializers import RandomUniform 

from deep.ddpg import DDPGAgent
from deep.preprocessors import HistoryPreprocessor, PendulumPreprocessor, PreprocessorSequence
from deep.policy import LinearDecayGreedyEpsilonPolicy, UniformRandomPolicy
#from deeprl_hw2.action_replay_memory import ActionReplayMemory
from deep.action_replay_memory import ActionReplayMemory
from deep.objectives import huber_loss
from deep.utils import memory_burn_in

import numpy as np
import json
import sys

import tensorflow as tf
from keras import backend as K
from keras.layers.normalization import BatchNormalization


def create_actor_model(hist_window,state_size,action_dim,model_name):

    #the current pendulum state
    pendulum_input = Input(shape=(state_size,hist_window), name='actor_pendulum_input')
    merged_layer = Flatten()(pendulum_input)
    #merged_layer = BatchNormalization(axis=1)(merged_layer)
    #bunch of dense layers
    merged_layer = Dense(400)(merged_layer)
    #merged_layer = BatchNormalization(axis=1)(merged_layer)
    merged_layer = Activation('relu')(merged_layer)


    merged_layer = Dense(300)(merged_layer)
    #merged_layer = BatchNormalization(axis=1)(merged_layer)
    merged_layer = Activation('relu')(merged_layer)

    #output layer
    uniform_initializer = RandomUniform(minval=-3e-3, maxval=3e-3)
    output_layer = Dense(action_dim, activation='tanh', kernel_initializer=uniform_initializer)(merged_layer)


    scaled_out_put_layer = Lambda(lambda x:x*2)(output_layer)
   
    model = Model(inputs=pendulum_input, outputs=scaled_out_put_layer, name=model_name)

    return model

def create_critic_model(hist_window,state_size,action_dim,model_name):

   #the current pendulum state
    pendulum_input = Input(shape=(state_size,hist_window), name='critic_pendulum_input')
    merged_layer = Flatten()(pendulum_input)
    #merged_layer = BatchNormalization()(merged_layer)
    #bunch of dense layers
    merged_layer = Dense(400)(merged_layer)
    #merged_layer = BatchNormalization()(merged_layer)
    merged_layer = Activation('relu')(merged_layer)


    #merged_layer = Dense(300,activation='relu')(merged_layer)
    #merge with the action
    #the actual output inputs
    action_input = Input(shape=(action_dim,),name='action_input')
    #action_layer = BatchNormalization()(action_input)

    #action_layer = Dense(300,activation='relu')(action_input)
    #action_layer = Flatten()(action_input)
    #merge all layers
    #merged_layer = keras.layers.concatenate([merged_layer, action_layer])
    merged_layer = keras.layers.concatenate([merged_layer, action_input])

    merged_layer = Dense(600)(merged_layer)
    #merged_layer = BatchNormalization()(merged_layer)
    merged_layer = Activation('relu')(merged_layer)

    #output layer
    uniform_initializer = RandomUniform(minval=-3e-3, maxval=3e-3)
    output_layer = Dense(action_dim, activation='linear', kernel_initializer=uniform_initializer)(merged_layer)

    
    model = Model(inputs=[pendulum_input,action_input], outputs=output_layer, name=model_name)

    return model


def main():

    #env = gym.make("Enduro-v0")
    #env = gym.make("SpaceInvaders-v0")
    #env = gym.make("Breakout-v0")

    env_name = "Pendulum-v0"
    #env_name = "MountainCarContinuous-v0"

    env = gym.make(env_name)
    run_name = '{}-{}'.format(env_name,1)

    #initialize tensorflow
    sess = tf.Session()
    K.set_session(sess)
    K.set_learning_phase(1)

    state_size = env.observation_space.shape[0]
    batch_size = 64
    action_dim = env.action_space.shape[0]
    memory_size = 20000
    memory_burn_in_num = 1000
    history_size = 1

    actor_learning_rate = 0.01
    critic_learning_rate = 0.001
    
    history_prep = HistoryPreprocessor(history_size)
    pendulum_prep  = PendulumPreprocessor()
    #baxter_prep = BaxterPreprocessor(input_shape)
    #numpy_prep = NumpyPreprocessor()
    preprocessors = PreprocessorSequence([history_prep,pendulum_prep]) #from left to right

    actor_model = create_actor_model(history_size,state_size,action_dim,'actor_model_{}'.format(run_name,))
    critic_model = create_critic_model(history_size,state_size,action_dim,'critic_model_{}'.format(run_name,))

    # linear_model = create_model(history_size, input_shape, action_dim, model_name)
    # linear_model.summary()
    optimizer = Adam(lr=critic_learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #loss_func = huber_loss
    loss_func = 'mse'
    #memory = ActionReplayMemory(1000000,4)
    memory = ActionReplayMemory(memory_size,4)
    memory_burn_in(env,memory,preprocessors,memory_burn_in_num)


    agent = DDPGAgent(sess,actor_model, critic_model, preprocessors, memory, 0.99, batch_size,run_name)
    agent.compile(optimizer, loss_func, actor_learning_rate)



    # #print(reward_arr)
    # #print(curr_state_arr)
    # agent = DoubleQNAgent(linear_model, preprocessors, memory, policy,0.99, target_update_freq,None,train_freq,batch_size)
    # agent.compile(optimizer, loss_func)
    # agent.save_models()
    agent.fit(env,1000000,100000)
    #reward_arr, length_arr = agent.evaluate_detailed(env,100,render=True, verbose=True)

    #agent.evaluate(env, 5)

if __name__ == '__main__':
    main()
