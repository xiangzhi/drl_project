#!/usr/bin/env python

import gym
import time

from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute)
from keras.models import model_from_json
from keras.optimizers import Adam
from keras.engine.topology import Layer as BLayer

from deep.preprocessors import PreprocessorSequence, HistoryPreprocessor, BaxterPreprocessor, NumpyPreprocessor
from deep.ddpg import DDPGAgent


import sys
import json
from scipy import misc
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K

from gym.envs.registration import registry, register, make, spec
register(
    id='BaxterEnv-v0',
    entry_point='baxter_env:BaxterEnv',
    max_episode_steps=50,
)

def main():

    #initialize tensorflow
    sess = tf.Session()
    K.set_session(sess)


    if(len(sys.argv) != 3):
        print("usage:{} <model_json> <weights> ".format(sys.argv[0]))
        return sys.exit()
    env = gym.make("BaxterEnv-v0")
    with open(sys.argv[1]) as json_file:
        model = model_from_json(json.load(json_file))

    model.load_weights(sys.argv[2])
    print("weight loaded")
    epsilon = 0.01
    input_shape = (80,80)
    history_size = 4
    eval_size = 100

    history_prep = HistoryPreprocessor(history_size)
    baxter_prep = BaxterPreprocessor(input_shape)
    numpy_prep = NumpyPreprocessor()
    preprocessors = PreprocessorSequence([baxter_prep, history_prep,numpy_prep]) #from left to right


    agent = DDPGAgent(sess,model, model, preprocessors, None, None, None, None, None,"eval_run")
    reward_arr, length_arr = agent.evaluate_detailed(env,eval_size,render=False, verbose=True)
    print("\Ran {} Episodes, reward:M={}, SD={} length:M={}, SD={}".format(eval_size, np.mean(reward_arr),np.std(reward_arr),np.mean(length_arr), np.std(reward_arr)))
    print("max:{} min:{}".format(np.max(reward_arr), np.min(reward_arr)))

    plt.hist(reward_arr)
    plt.show()

if __name__ == '__main__':
    main()