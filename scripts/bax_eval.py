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
from deep.utils import random_eval


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
    if(len(sys.argv) != 5):
        print("usage:{} <environment> <run_name> <weight_num> <render>".format(sys.argv[0]))
        return sys.exit()

    env_name = sys.argv[1]
    actor_model_name = "actor_model_{}.model".format(sys.argv[2])
    critic_model_name = "critic_model_{}.model".format(sys.argv[2])
    actor_weight_name = "actor_model_{}-{}.weights".format(sys.argv[2], sys.argv[3])
    critic_weight_name = "critic_model_{}-{}.weights".format(sys.argv[2], sys.argv[3])

    env = gym.make(env_name)
    #print(env)

    #initialize tensorflow
    sess = tf.Session()
    K.set_session(sess)
    #K.set_learning_phase(0)
    init = tf.global_variables_initializer()
    sess.run(init)

    render_flag = True if sys.argv[4] == 't' else False

    #Load the model and weights
    with open(actor_model_name) as json_file:
        actor_model = model_from_json(json.load(json_file))
    with open(critic_model_name) as json_file:
        critic_model = model_from_json(json.load(json_file))
    actor_model.load_weights(actor_weight_name)
    critic_model.load_weights(critic_weight_name)

    # actor_model.compile('adam','mse')
    # critic_model.compile('adam','mse')

    input_shape = (80,80)
    history_size = 1
    eval_size = 100
    joint_only = True

    history_prep = HistoryPreprocessor(history_size)
    baxter_prep = BaxterPreprocessor(input_shape)
    numpy_prep = NumpyPreprocessor()
    if(not joint_only):
        keras_prep = KerasPreprocessor(["joint_input","image_input"])
        preprocessors = PreprocessorSequence([baxter_prep, history_prep,numpy_prep,keras_prep]) #from left to right
    else:
        pendulum_prep  = PendulumPreprocessor()
        keras_prep = KerasPreprocessor(['joint_input'])
        preprocessors = PreprocessorSequence([history_prep,pendulum_prep,keras_prep]) #from left to right

    agent = DDPGAgent(sess,actor_model, critic_model, preprocessors, None, None, None,"eval_run")
    #reward_arr, length_arr = agent.evaluate_detailed(env,eval_size,render=render_flag, verbose=True)
    reward_arr, length_arr = random_eval(env,eval_size,render=render_flag, verbose=True)
    print("\nRan {} Episodes, reward:M={}, SD={} length:M={}, SD={}".format(eval_size, np.mean(reward_arr),np.std(reward_arr),np.mean(length_arr), np.std(reward_arr)))
    print("max:{} min:{}".format(np.max(reward_arr), np.min(reward_arr)))

    plt.hist(reward_arr)
    plt.show()

if __name__ == '__main__':
    main()