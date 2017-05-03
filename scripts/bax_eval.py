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

from gym.envs.registration import registry, register, make, spec

register(
    id='BaxterEnv-v0',
    entry_point='baxter_env:BaxterEnv',
    max_episode_steps=50,
)

def main():

    #make sure it's the right format
    if(len(sys.argv) != 4):
        print("usage:{} <environment> <run_name> <weight_num>".format(sys.argv[0]))
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


    #Load the model and weights
    with open(actor_model_name) as json_file:
        actor_model = model_from_json(json.load(json_file))
    with open(critic_model_name) as json_file:
        critic_model = model_from_json(json.load(json_file))
    actor_model.load_weights(actor_weight_name)
    critic_model.load_weights(critic_weight_name)

    # actor_model.compile('adam','mse')
    # critic_model.compile('adam','mse')

    print(actor_weight_name)

    input_shape = (80,80)
    batch_size = 32
    action_dim = 7
    history_size = 4
    eval_size = 100

    history_prep = HistoryPreprocessor(history_size)
    baxter_prep = BaxterPreprocessor(input_shape)
    numpy_prep = NumpyPreprocessor()
    keras_prep = KerasPreprocessor(["joint_input","image_input"])
    preprocessors = PreprocessorSequence([baxter_prep, history_prep,numpy_prep,keras_prep]) #from left to right

    agent = DDPGAgent(sess,actor_model, critic_model, preprocessors, None, None, None,"eval_run")
    reward_arr, length_arr = agent.evaluate_detailed(env,eval_size,render=False, verbose=True)
    print("\nRan {} Episodes, reward:M={}, SD={} length:M={}, SD={}".format(eval_size, np.mean(reward_arr),np.std(reward_arr),np.mean(length_arr), np.std(reward_arr)))
    print("max:{} min:{}".format(np.max(reward_arr), np.min(reward_arr)))

    plt.hist(reward_arr)
    plt.show()

if __name__ == '__main__':
    main()