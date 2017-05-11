import gym
import time

from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute)
from keras.models import model_from_json
from keras.optimizers import Adam
from keras.engine.topology import Layer as BLayer

from deep.preprocessors import PreprocessorSequence, HistoryPreprocessor, PendulumPreprocessor, KerasPreprocessor
from deep.ddpg import DDPGAgent


import sys
import json
from scipy import misc
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K

def random_eval(env, num_episodes, render=False, verbose=False, max_episode_length= None):
    reward_arr = np.zeros((num_episodes))
    length_arr = np.zeros((num_episodes))
    frame_num = 0


    #number of episodes
    for episode_num in range(0, num_episodes):

      curr_episode_reward = 0
      curr_episode_step = 0
      #get the initial state
      curr_state = env.reset()
      if(render):
        env.render()

      frame_num = 0
      curr_reward = 0
      curr_action = 0
      is_terminal = False

      while(max_episode_length == None or curr_episode_step <= max_episode_length):
       
        if(render):
          env.render()
        curr_episode_step += 1
        #progress and step through for a fix number of steps according the skip frame number
        next_state, reward, is_terminal, info = env.step(env.action_space.sample())

        if(is_terminal):
          break
        curr_state = next_state
        curr_episode_reward += reward


      #print("Episode {} ended with length:{} and reward:{}".format(episode_num, curr_episode_step, curr_episode_reward))
      reward_arr[episode_num] = curr_episode_reward
      length_arr[episode_num] = curr_episode_step

      if(verbose):
        sys.stdout.write("\revaluating game: {}/{} length:{} and reward:{}".format(episode_num+1, num_episodes, curr_episode_step, curr_episode_reward))
        sys.stdout.flush()
    return reward_arr, length_arr

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

    render_flag = True if sys.argv[4] == 't' else False

    epsilon = 0.01
    input_shape = (80,80)
    history_size = 1
    eval_size = 100

    history_prep = HistoryPreprocessor(history_size)
    pendulum_prep  = PendulumPreprocessor()
    keras_prep = KerasPreprocessor(['pendulum_input'])
    #baxter_prep = BaxterPreprocessor(input_shape)
    #numpy_prep = NumpyPreprocessor()
    preprocessors = PreprocessorSequence([history_prep,pendulum_prep, keras_prep]) #from left to right

    agent = DDPGAgent(sess,actor_model, critic_model, preprocessors, None, None, None,"eval_run")
    reward_arr, length_arr = agent.evaluate_detailed(env,eval_size,render=render_flag, verbose=True)
    #reward_arr, length_arr = random_eval(env,eval_size,render=render_flag, verbose=True)
    print("\nRan {} Episodes, reward:M={}, SD={} length:M={}, SD={}".format(eval_size, np.mean(reward_arr),np.std(reward_arr),np.mean(length_arr), np.std(reward_arr)))
    print("max:{} min:{}".format(np.max(reward_arr), np.min(reward_arr)))

    plt.hist(reward_arr)
    plt.show()

if __name__ == '__main__':
    main()