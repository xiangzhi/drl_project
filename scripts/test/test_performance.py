
import unittest
import time
import gym
from deeprl_hw2.dqn import DQNAgent

from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute)
from keras.models import Model, Sequential
from keras.optimizers import Adam

from deeprl_hw2.preprocessors import PreprocessorSequence, HistoryPreprocessor, AtariPreprocessor, NumpyPreprocessor
from deeprl_hw2.policy import LinearDecayGreedyEpsilonPolicy, UniformRandomPolicy,SamePolicy
from deeprl_hw2.action_replay_memory import ActionReplayMemory
from deeprl_hw2.objectives import huber_loss
from deeprl_hw2.utils import memory_burn_in

import numpy as np
import json
import sys


from deeprl_hw2.preprocessors import PreprocessorSequence, HistoryPreprocessor, AtariPreprocessor, NumpyPreprocessor


class PerformanceTestMethods(unittest.TestCase):

    def testPerformance(self):
        """
        Test to make sure each model(DQN, DDQN, DoubleQN) could be created and compiled
        """

        #create a model of the world  
        env = gym.make("SpaceInvaders-v0")
        env.frameskip = 1
        #create a fake keras model
        input_shape = (84,84)
        window = 4
        num_actions = env.action_space.n
        model = Sequential(name="test_model")
        model.add(Convolution2D(filters=16, kernel_size=8, strides=4, activation='relu', input_shape=(input_shape[0],input_shape[1],window)))
        model.add(Convolution2D(filters=32, kernel_size=4, strides=2, activation='relu'))
        model.add(Convolution2D(filters=64, kernel_size=3, strides=1, activation='relu'))
        model.add(Flatten())
        model.add(Dense(units=512, activation='relu'))
        model.add(Dense(units=num_actions, activation='linear'))
        #create loss function & optimizer
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        loss_func = huber_loss
        #preprocessors
        history_prep = HistoryPreprocessor(4)
        atari_prep = AtariPreprocessor(input_shape,0,999)
        numpy_prep = NumpyPreprocessor()
        preprocessors = PreprocessorSequence([atari_prep, history_prep, numpy_prep]) #from left to right
        memory = ActionReplayMemory(100000,4)
        #policy = LinearDecayGreedyEpsilonPolicy(1, 0.1,100000)
        policy = SamePolicy(1)

        #agent = DQNAgent(model, preprocessors, memory, policy,0.99, target_update_freq,None,train_freq,batch_size)
        dqn_agent = DQNAgent(model, preprocessors, memory, policy, 0.99, 10000, None, 4, 32)
        dqn_agent.compile(optimizer, loss_func)
        total_time = 0
        times = 50
        for i in range(0,times):
            start_time = time.time()
            dqn_agent.evaluate_detailed(env,1)
            total_time += (time.time() - start_time)
            sys.stdout.write('\r{}'.format(i))
            sys.stdout.flush()
        print("average evaluation time:{} total time:{}".format(total_time/times, total_time))
        #dqn_agent.fit(env,1,1)


if __name__ == '__main__':
    unittest.main()