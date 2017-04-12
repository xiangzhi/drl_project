import gym
import time
from deeprl_hw2.dqn import DQNAgent

from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute)
from keras.models import model_from_json
from keras.optimizers import Adam
from keras.engine.topology import Layer as BLayer

from deeprl_hw2.preprocessors import PreprocessorSequence, HistoryPreprocessor, AtariPreprocessor, NumpyPreprocessor
from deeprl_hw2.policy import GreedyEpsilonPolicy, UniformRandomPolicy
from deeprl_hw2.action_replay_memory import ActionReplayMemory

import sys
import json
from scipy import misc
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K


class Eq9(BLayer):
    def __init__(self, num_actions,**kwargs):
        super(Eq9, self).__init__(**kwargs)
        self._num_actions = num_actions

    def build(self, input_shape):
        super(Eq9, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        value_network, advantage_network = tf.split(x,[1,self._num_actions],axis=1)
        mean_advantage = K.mean(advantage_network)
        advantage_diff = tf.subtract(advantage_network,mean_advantage)
        result = tf.add(value_network, advantage_diff)
        return result

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self._num_actions)

    def get_config(self):
        config = {
            "num_actions":self._num_actions
        }
        base_config = super(Eq9, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def main():
    if(len(sys.argv) != 6):
        print("usage:{} <env> <model_json> <weights> <render> <random>".format(sys.argv[0]))
        return sys.exit()
    env = gym.make(sys.argv[1])
    env.frameskip = 1
    with open(sys.argv[2]) as json_file:
        model = model_from_json(json.load(json_file),{"Eq9":Eq9})
    model.load_weights(sys.argv[3])
    epsilon = 0.01
    input_shape = (84,84)
    history_size = 4
    eval_size = 100
    render = (sys.argv[4] == "y")

    history_prep = HistoryPreprocessor(history_size)
    atari_prep = AtariPreprocessor(input_shape,0,999)
    numpy_prep = NumpyPreprocessor()
    preprocessors = PreprocessorSequence([atari_prep, history_prep, numpy_prep]) #from left to right

    if(sys.argv[5] == "y"):
        print("using random policy")
        policy = UniformRandomPolicy(env.action_space.n)
    else:
        print("using greedy policy")
        policy = GreedyEpsilonPolicy(epsilon)

    agent = DQNAgent(model, preprocessors, None, policy, 0.99, None,None,None,None)
    agent.add_keras_custom_layers({"Eq9":Eq9})
    reward_arr, length_arr = agent.evaluate_detailed(env,eval_size,render=render, verbose=True)
    print("\rPlayed {} games, reward:M={}, SD={} length:M={}, SD={}".format(eval_size, np.mean(reward_arr),np.std(reward_arr),np.mean(length_arr), np.std(reward_arr)))
    print("max:{} min:{}".format(np.max(reward_arr), np.min(reward_arr)))

    plt.hist(reward_arr)
    plt.show()

    #check for preprocessors 
    # state = env.reset()

    # for i in range(0,5):
    #     process_state = preprocessors.process_state_for_network(state)
    #     state,reward,is_teminal,debug = env.step(0)
    # print(process_state.shape)
    # misc.imshow(process_state[:,:,0])
    # misc.imshow(process_state[:,:,1])
    # misc.imshow(process_state[:,:,2])

if __name__ == '__main__':
    main()