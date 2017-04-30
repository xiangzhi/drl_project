import gym
import time

from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute)
from keras.models import model_from_json
from keras.optimizers import Adam
from keras.engine.topology import Layer as BLayer

from deep.preprocessors import PreprocessorSequence, HistoryPreprocessor, PendulumPreprocessor
from deep.ddpg import DDPGAgent


import sys
import json
from scipy import misc
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K

def main():
    # if(len(sys.argv) != 3):
    #     print("usage:{} <model_json> <weights> ".format(sys.argv[0]))
    #     return sys.exit()
    env = gym.make("Pendulum-v0")
    #print(env)
    with open(sys.argv[1]) as json_file:
        actor_model = model_from_json(json.load(json_file))

    with open(sys.argv[2]) as json_file:
        print(sys.argv[2])
        critic_model = model_from_json(json.load(json_file))

    actor_model.load_weights(sys.argv[3])
    critic_model.load_weights(sys.argv[4])

    # actor_model.compile()
    # critic_model.compile()

    epsilon = 0.01
    input_shape = (80,80)
    history_size = 1
    eval_size = 100

    history_prep = HistoryPreprocessor(history_size)
    pendulum_prep  = PendulumPreprocessor()
    #baxter_prep = BaxterPreprocessor(input_shape)
    #numpy_prep = NumpyPreprocessor()
    preprocessors = PreprocessorSequence([history_prep,pendulum_prep]) #from left to right

    #initialize tensorflow
    sess = tf.Session()
    K.set_session(sess)
    init = tf.global_variables_initializer()
    sess.run(init)

    agent = DDPGAgent(sess,actor_model, critic_model, preprocessors, None, None, None, None, None,"eval_run")
    reward_arr, length_arr = agent.evaluate_detailed(env,eval_size,render=True, verbose=True)
    print("\Ran {} Episodes, reward:M={}, SD={} length:M={}, SD={}".format(eval_size, np.mean(reward_arr),np.std(reward_arr),np.mean(length_arr), np.std(reward_arr)))
    print("max:{} min:{}".format(np.max(reward_arr), np.min(reward_arr)))

    plt.hist(reward_arr)
    plt.show()

if __name__ == '__main__':
    main()