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


def create_actor_model(hist_window,state_size,action_dim,scale,model_name):

    #the current pendulum state
    pendulum_input = Input(shape=(state_size,hist_window), name='pendulum_input')
    merged_layer = Flatten()(pendulum_input)
    #merged_layer = BatchNormalization()(merged_layer)
    #bunch of dense layers
    fan_limit = 1./np.sqrt(state_size * hist_window)
    #merged_layer = Dense(400, kernel_initializer=RandomUniform(minval=-fan_limit, maxval=fan_limit))(merged_layer)
    merged_layer = Dense(400)(merged_layer)
    merged_layer = Activation('relu')(merged_layer)
    #merged_layer = BatchNormalization()(merged_layer)

    fan_limit = 1./np.sqrt(400)
    #merged_layer = Dense(300, kernel_initializer=RandomUniform(minval=-fan_limit, maxval=fan_limit))(merged_layer)
    merged_layer = Dense(300)(merged_layer)
    merged_layer = Activation('relu')(merged_layer)
    #merged_layer = BatchNormalization()(merged_layer)

    #output layer
    uniform_initializer = RandomUniform(minval=-3e-3, maxval=3e-3)
    output_layer = Dense(action_dim, activation='tanh', kernel_initializer=uniform_initializer)(merged_layer)


    scaled_out_put_layer = Lambda(lambda x:x*1)(output_layer)
    model = Model(inputs=pendulum_input, outputs=scaled_out_put_layer, name=model_name)

    return model

def create_critic_model(hist_window,state_size,action_dim,model_name):

   #the current pendulum state
    pendulum_input = Input(shape=(state_size,hist_window), name='pendulum_input')
    merged_layer = Flatten()(pendulum_input)
    #merged_layer = BatchNormalization()(merged_layer)
    #bunch of dense layers
    fan_limit = 1./np.sqrt(state_size * hist_window)
    #merged_layer = Dense(400, kernel_initializer=RandomUniform(minval=-fan_limit, maxval=fan_limit))(merged_layer)
    merged_layer = Dense(400)(merged_layer)
    merged_layer = Activation('relu')(merged_layer)
    #merged_layer = BatchNormalization()(merged_layer)


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
    fan_limit = 1./np.sqrt(400+action_dim)
    #merged_layer = Dense(600, kernel_initializer=RandomUniform(minval=-fan_limit, maxval=fan_limit))(merged_layer)
    merged_layer = Dense(600)(merged_layer)
    merged_layer = Activation('relu')(merged_layer)
    #merged_layer = BatchNormalization()(merged_layer)

    #output layer
    uniform_initializer = RandomUniform(minval=-3e-3, maxval=3e-3)
    output_layer = Dense(action_dim, activation='linear', kernel_initializer=uniform_initializer)(merged_layer)

    
    model = Model(inputs=[pendulum_input,action_input], outputs=output_layer, name=model_name)

    return model


def main():


    #env = gym.make("Enduro-v0")
    #env = gym.make("SpaceInvaders-v0")
    #env = gym.make("Breakout-v0")

    #env_name = "Pendulum-v0"
    #env_name = "MountainCarContinuous-v0"
    env_name = "LunarLanderContinuous-v2"

    env = gym.make(env_name)
    run_name = '{}-{}'.format(env_name,2)

    #initialize tensorflow
    sess = tf.Session()
    K.set_session(sess)
    K.set_learning_phase(1)

    state_size = env.observation_space.shape[0]
    batch_size = 64
    #batch_size = 32
    action_dim = env.action_space.shape[0]
    #memory_size = 20000
    memory_size = int(1e6)
    memory_burn_in_num = int(memory_size*0.05)# 5% of the whole memory
    #memory_burn_in_num = 5000
    #memory_burn_in_num = 0
    history_size = 1
    #scale = np.min(env.action_space.high)
    scale = np.min(env.action_space.high)
    actor_learning_rate = 0.01
    critic_learning_rate = 0.001
    
    history_prep = HistoryPreprocessor(history_size)
    pendulum_prep  = PendulumPreprocessor()
    keras_prep = KerasPreprocessor(['pendulum_input'])
    #baxter_prep = BaxterPreprocessor(input_shape)
    #numpy_prep = NumpyPreprocessor()
    preprocessors = PreprocessorSequence([history_prep,pendulum_prep,keras_prep]) #from left to right

    #create noise generator
    #noise_generator = OU_Generator(np.array([0,0]))
    noise_generator = OU_Generator(np.zeros(action_dim))


    actor_model = create_actor_model(history_size,state_size,action_dim,scale,'actor_model_{}'.format(run_name,))
    critic_model = create_critic_model(history_size,state_size,action_dim,'critic_model_{}'.format(run_name,))

    # linear_model = create_model(history_size, input_shape, action_dim, model_name)
    # linear_model.summary()
    #optimizer = Adam(lr=critic_learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-2)
    optimizer = Adam(lr=critic_learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    loss_func = huber_loss
    #loss_func = 'mse'
    #memory = ActionReplayMemory(1000000,4)
    memory = ActionReplayMemory(memory_size,4)
    memory_burn_in(env,memory,preprocessors,memory_burn_in_num)


    agent = DDPGAgent(sess,actor_model, critic_model, preprocessors, memory, 0.99, batch_size,run_name)
    agent.set_noise_generator(noise_generator)
    agent.compile(optimizer, loss_func, actor_learning_rate,1,action_dim)



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
