#!/usr/bin/env python

import gym
import time

from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute)
from keras.models import model_from_json
from keras.optimizers import Adam
from keras.engine.topology import Layer as BLayer

from deep.preprocessors import PreprocessorSequence, HistoryPreprocessor, PendulumPreprocessor, KerasPreprocessor, BaxterPreprocessor, NumpyPreprocessor
from deep.ddpg import DDPGAgent


import sys
import json
from scipy import misc
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K

import bax_env_register

def main():

    #make sure it's the right format
    if(len(sys.argv) != 2):
        print("usage:{} <environment>".format(sys.argv[0]))
        return sys.exit()

    env_name = sys.argv[1]

    env = gym.make(env_name)
    #print(env)
    while True:
        state = env.reset()
        is_terminal = False
        while not is_terminal:
            action = env.action_space.sample()
            next_state, reward, is_terminal, debug_info = env.step(action)    
            env.render()

if __name__ == '__main__':
    main()