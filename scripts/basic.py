#!/usr/bin/env python

import gym
import time

from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute)
import keras.layers
from keras.models import Model, Sequential
from keras.optimizers import Adam

from deep.ddpg import DDPGAgent
from deep.preprocessors import PreprocessorSequence, HistoryPreprocessor, BaxterPreprocessor, NumpyPreprocessor
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

from gym.envs.registration import registry, register, make, spec
register(
    id='BaxterEnv-v0',
    entry_point='baxter_env:BaxterEnv',
    max_episode_steps=50,
)



def create_actor_model(hist_window,img_input_shape,action_dim,model_name):

    #main_input = Input(shape=(hist_window, img_input_shape[0], img_input_shape[1]), name='image_input')
    main_input = Input(shape=(img_input_shape[0], img_input_shape[1],img_input_shape[2]), name='image_input')
    layers = Convolution2D(filters=32, kernel_size=8, strides=4, activation='relu')(main_input)
    img_out_layer = Convolution2D(filters=32, kernel_size=8, strides=4, activation='relu')(layers)
    img_out_layer = Flatten()(img_out_layer)
    # layers = Convolution2D(filters=64, kernel_size=4, strides=2, activation='relu')(layers)
    # layers = Convolution2D(filters=64, kernel_size=3, strides=1, activation='relu')(layers)
    #image layers

    #action inputs
    joint_input = Input(shape=(action_dim,hist_window),name='joint_input')
    joint_layer = Dense(64,activation='relu')(joint_input)
    joint_layer = Flatten()(joint_layer)
    #merge both layers
    merged_layer = keras.layers.concatenate([joint_layer, img_out_layer])
    merged_layer = Dense(256,activation='relu')(merged_layer)

    #output layer
    output_layer = Dense(action_dim,activation='tanh')(merged_layer)

    model = Model(inputs=[main_input,joint_input], outputs=output_layer, name=model_name)

    return model

def create_critic_model(hist_window,img_input_shape,action_dim,model_name):

    main_input = Input(shape=(img_input_shape[0], img_input_shape[1],img_input_shape[2]), name='image_input')
    layers = Convolution2D(filters=32, kernel_size=8, strides=4, activation='relu')(main_input)
    img_out_layer = Convolution2D(filters=32, kernel_size=4, strides=2, activation='relu')(layers)
    # layers = Convolution2D(filters=64, kernel_size=3, strides=1, activation='relu')(layers)
    #image layers
    img_out_layer = Flatten()(img_out_layer)

    #action inputs
    joint_input = Input(shape=(action_dim,hist_window),name='joint_input')
    joint_layer = Dense(64,activation='relu')(joint_input)
    joint_layer = Flatten()(joint_layer)

    #the actual output inputs
    action_input = Input(shape=(action_dim,),name='action_input')
    #action_layer = Flatten()(action_input)

    #merge all three layers
    merged_layer = keras.layers.concatenate([joint_layer, img_out_layer, action_input])
    merged_layer = Dense(256,activation='relu')(merged_layer)

    #output layer
    output_layer = Dense(action_dim,activation='tanh')(merged_layer)

    model = Model(inputs=[main_input,joint_input,action_input], outputs=output_layer, name=model_name)

    return model


def main():

    #env = gym.make("Enduro-v0")
    #env = gym.make("SpaceInvaders-v0")
    #env = gym.make("Breakout-v0")

    env = gym.make("BaxterEnv-v0")
    run_name = 'basic_1'

    #initialize tensorflow
    sess = tf.Session()
    K.set_session(sess)


    input_shape = (80,80,3)
    batch_size = 32
    action_dim = 7
    memory_size = 100000
    memory_burn_in_num = 5000
    start_epsilon = 1
    end_epsilon = 0.01
    decay_steps = 1000000
    target_update_freq = 10000
    train_freq = 4 #How often you train the network
    history_size = 4
    
    history_prep = HistoryPreprocessor(history_size)
    baxter_prep = BaxterPreprocessor(input_shape)
    numpy_prep = NumpyPreprocessor()
    preprocessors = PreprocessorSequence([baxter_prep, history_prep,numpy_prep]) #from left to right

    policy = LinearDecayGreedyEpsilonPolicy(start_epsilon, end_epsilon,decay_steps)


    actor_model = create_actor_model(history_size,input_shape,action_dim,'actor_model_{}'.format(run_name,))
    critic_model = create_critic_model(history_size,input_shape,action_dim,'critic_model_{}'.format(run_name,))

    # linear_model = create_model(history_size, input_shape, action_dim, model_name)
    # linear_model.summary()
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    loss_func = huber_loss

    #memory = ActionReplayMemory(1000000,4)
    memory = ActionReplayMemory(memory_size,4)
    memory_burn_in(env,memory,preprocessors,memory_burn_in_num)


    agent = DDPGAgent(sess,actor_model, critic_model, preprocessors, memory, policy, 0.99, target_update_freq, batch_size,run_name)
    agent.compile(optimizer, loss_func)



    # #print(reward_arr)
    # #print(curr_state_arr)
    # agent = DoubleQNAgent(linear_model, preprocessors, memory, policy,0.99, target_update_freq,None,train_freq,batch_size)
    # agent.compile(optimizer, loss_func)
    # agent.save_models()
    agent.fit(env,1000000,100000)
    #agent.evaluate(env, 5)

if __name__ == '__main__':
    main()
