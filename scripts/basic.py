
import gym
import time
from dqn.dqn import DQNAgent

from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute)
from keras.models import Model, Sequential
from keras.optimizers import Adam

from deeprl_hw2.preprocessors import PreprocessorSequence, HistoryPreprocessor, BaxterPreprocessor, NumpyPreprocessor
from deeprl_hw2.policy import LinearDecayGreedyEpsilonPolicy, UniformRandomPolicy
#from deeprl_hw2.action_replay_memory import ActionReplayMemory
from deeprl_hw2.action_replay_memory import ActionReplayMemory
from deeprl_hw2.objectives import huber_loss
from deeprl_hw2.utils import memory_burn_in

import numpy as np
import json
import sys

from gym.envs.registration import registry, register, make, spec
register(
    id='BaxterEnv-v0',
    entry_point='baxter_env:BaxterEnv',
    max_episode_steps=10,
)



def create_model(window, input_shape, action_dim,
                 model_name='q_network'):

    model = Sequential(name=model_name)
    model.add(Flatten(input_shape=(input_shape[0],input_shape[1],input_shape[2],window)))
    model.add(Dense(units=action_dim, activation='tanh'))
    return model

def main():

    #env = gym.make("Enduro-v0")
    #env = gym.make("SpaceInvaders-v0")
    #env = gym.make("Breakout-v0")

    model_name = "basic"
    
    env = gym.make("BaxterEnv-v0")

    input_shape = (80,80,3)
    batch_size = 32
    action_dim = 7
    memory_size = 100000
    memory_burn_in_num = 500
    start_epsilon = 1
    end_epsilon = 0.01
    decay_steps = 1000000
    target_update_freq = 10000
    train_freq = 4 #How often you train the network
    history_size = 4
    
    history_prep = HistoryPreprocessor(history_size)
    baxter_prep = BaxterPreprocessor(input_shape)
    numpy_prep = NumpyPreprocessor()
    preprocessors = PreprocessorSequence([baxter_prep, history_prep, numpy_prep]) #from left to right

    policy = LinearDecayGreedyEpsilonPolicy(start_epsilon, end_epsilon,decay_steps)

    linear_model = create_model(history_size, input_shape, action_dim, model_name)
    linear_model.summary()
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    loss_func = huber_loss
    #linear_model.compile(optimizer, loss_func)

    #memory = ActionReplayMemory(1000000,4)
    memory = ActionReplayMemory(memory_size,4)
    memory_burn_in(env,memory,preprocessors,memory_burn_in_num)

    #print(reward_arr)
    #print(curr_state_arr)
    agent = DoubleQNAgent(linear_model, preprocessors, memory, policy,0.99, target_update_freq,None,train_freq,batch_size)
    agent.compile(optimizer, loss_func)
    agent.save_models()
    agent.fit(env,1000000,100000)
    # #agent.evaluate(env, 5)

if __name__ == '__main__':
    main()
