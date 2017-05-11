#!/usr/bin/env python

import gym
import time

from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input, Lambda,
                          Permute)
import keras.layers
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.initializers import VarianceScaling, RandomUniform

from deep.ddpg import DDPGAgent
from deep.preprocessors import PreprocessorSequence, HistoryPreprocessor, BaxterPreprocessor, NumpyPreprocessor, KerasPreprocessor
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

import bax_env_register



def create_actor_model(hist_window,img_input_shape,number_joint,action_dim,model_name):

    #image_inputs
    main_input = Input(shape=(img_input_shape[0], img_input_shape[1],img_input_shape[2]), name='image_input')
    layers = Convolution2D(filters=32, kernel_size=8, strides=4, activation='relu')(main_input)
    img_out_layer = Convolution2D(filters=32, kernel_size=4, strides=2, activation='relu')(layers)
    img_out_layer = Convolution2D(filters=32, kernel_size=3, strides=1, activation='relu')(layers)
    img_out_layer = Flatten()(img_out_layer)

    #action inputs
    joint_input = Input(shape=(number_joint,hist_window),name='joint_input')
    joint_layer = Dense(32,activation='relu')(joint_input)
    joint_layer = Flatten()(joint_layer)
    #merge both layers
    merged_layer = keras.layers.concatenate([joint_layer, img_out_layer])
    merged_layer = Dense(64,activation='relu')(merged_layer)
    merged_layer = Dense(64,activation='relu')(merged_layer)
    merged_layer = Dense(64,activation='relu')(merged_layer)
    merged_layer = Dense(64,activation='relu')(merged_layer)
    #output layer
    output_layer = Dense(action_dim, activation='tanh')(merged_layer)

    #QUICK hack to deal with the problem of velocity kept getting clipped
    scaled_out_put_layer = Lambda(lambda x:x*0.5)(output_layer)
    model = Model(inputs=[main_input,joint_input], outputs=scaled_out_put_layer, name=model_name)

    return model

def create_critic_model(hist_window,img_input_shape,number_joint,action_dim,model_name):

    #image inputs
    main_input = Input(shape=(img_input_shape[0], img_input_shape[1],img_input_shape[2]), name='image_input')
    layers = Convolution2D(filters=32, kernel_size=8, strides=4, activation='relu')(main_input)
    img_out_layer = Convolution2D(filters=32, kernel_size=4, strides=2, activation='relu')(layers)
    img_out_layer = Convolution2D(filters=32, kernel_size=3, strides=1, activation='relu')(layers)
    img_out_layer = Flatten()(img_out_layer)

    #action inputs
    joint_input = Input(shape=(number_joint, hist_window),name='joint_input')
    joint_layer = Dense(32,activation='relu')(joint_input)
    joint_layer = Flatten()(joint_layer)

    #merge and add first hidden layer
    merged_layer = keras.layers.concatenate([joint_layer, img_out_layer])
    merged_layer = Dense(128,activation='relu')(merged_layer)

    #the actual output inputs
    action_input = Input(shape=(action_dim,),name='action_input')
    #action_layer = Flatten()(action_input)

    #merge all three layers
    merged_layer = keras.layers.concatenate([merged_layer, action_input])
    merged_layer = Dense(128,activation='relu')(merged_layer)
    merged_layer = Dense(128,activation='relu')(merged_layer)
    merged_layer = Dense(128,activation='relu')(merged_layer)
    #output layer
    output_layer = Dense(action_dim, activation='linear')(merged_layer)

    model = Model(inputs=[main_input,joint_input,action_input], outputs=output_layer, name=model_name)

    return model


def main():

    #env = gym.make("Enduro-v0")
    #env = gym.make("SpaceInvaders-v0")
    #env = gym.make("Breakout-v0")

    env = gym.make("BaxterEnv-v0")
    run_name = 'basic_2' if (len(sys.argv) < 2) else sys.argv[1]

    #initialize tensorflow
    sess = tf.Session()
    K.set_session(sess)


    input_shape = (80,80,3)
    batch_size = 32
    action_dim = env.action_space.shape[0]
    joint_state_num = env.observation_space.shape[0]
    memory_size = 1000000
    memory_burn_in_num = int(memory_size*0.05)# 5% of the whole memory
    #memory_burn_in_num = 100# 5% of the whole memory
    history_size = 4
    
    actor_learning_rate = 0.01
    critic_learning_rate = 0.001

    history_prep = HistoryPreprocessor(history_size)
    baxter_prep = BaxterPreprocessor(input_shape)
    numpy_prep = NumpyPreprocessor()
    keras_prep = KerasPreprocessor(["joint_input","image_input"])
    preprocessors = PreprocessorSequence([baxter_prep, history_prep,numpy_prep,keras_prep]) #from left to right

    actor_model = create_actor_model(history_size,input_shape,joint_state_num, action_dim,'actor_model_{}'.format(run_name,))
    critic_model = create_critic_model(history_size,input_shape,number_joint=joint_state_num,
        action_dim=action_dim,
        model_name='critic_model_{}'.format(run_name,))

    #create optimizer for critic
    optimizer = Adam(lr=critic_learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    loss_func = huber_loss

    noise_generator = OU_Generator(np.zeros(action_dim),theta=0.1, sigma=0.1)

    #memory = ActionReplayMemory(1000000,4)
    memory = ActionReplayMemory(memory_size,4)
    memory_burn_in(env,memory,preprocessors,memory_burn_in_num)


    agent = DDPGAgent(sess,actor_model, critic_model, preprocessors, memory, 0.99, batch_size,run_name)
    agent.compile(optimizer, loss_func, actor_learning_rate,2,action_dim)
    agent.set_noise_generator(noise_generator)

    agent.fit(env,1000000,100000)
    #agent.evaluate(env, 5)

if __name__ == '__main__':
    main()
