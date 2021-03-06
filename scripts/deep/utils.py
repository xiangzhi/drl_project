"""Common functions you may find useful in your implementation."""

import semver
import tensorflow as tf
from keras.models import Model, model_from_json
import sys
import numpy as np

def clone_keras_model(target, custom_objects=None):  
    """
    return a keras model with the same setup
    """
    new_model = model_from_json(target.to_json(),custom_objects)
    new_model.set_weights(target.get_weights())
    return new_model

def get_uninitialized_variables(variables=None):
    """Return a list of uninitialized tf variables.

    Parameters
    ----------
    variables: tf.Variable, list(tf.Variable), optional
      Filter variable list to only those that are uninitialized. If no
      variables are specified the list of all variables in the graph
      will be used.

    Returns
    -------
    list(tf.Variable)
      List of uninitialized tf variables.
    """
    sess = tf.get_default_session()
    if variables is None:
        variables = tf.global_variables()
    else:
        variables = list(variables)

    if len(variables) == 0:
        return []

    if semver.match(tf.__version__, '<1.0.0'):
        init_flag = sess.run(
            tf.pack([tf.is_variable_initialized(v) for v in variables]))
    else:
        init_flag = sess.run(
            tf.stack([tf.is_variable_initialized(v) for v in variables]))
    return [v for v, f in zip(variables, init_flag) if not f]


def get_soft_target_model_updates(target, source, tau):
    """Return list of target model update ops.

    These are soft target updates. Meaning that the target values are
    slowly adjusted, rather than directly copied over from the source
    model.

    The update is of the form:

    $W' \gets (1- \tau) W' + \tau W$ where $W'$ is the target weight
    and $W$ is the source weight.

    Parameters
    ----------
    target: keras.models.Model
      The target model. Should have same architecture as source model.
    source: keras.models.Model
      The source model. Should have same architecture as target model.
    tau: float
      The weight of the source weights to the target weights used
      during update.

    Returns
    -------
    list(tf.Tensor)
      List of tensor update ops.
    """
    target_weights = target.get_weights()
    tau_values = np.ones(np.shape(target_weights)) * tau
    new_weights = (1 - tau_values) * target.get_weights() + tau_values * source.get_weights()
    target.set_weights(new_weights)
    return target


def get_hard_target_model_updates(target, source):
    """Return list of target model update ops.

    These are hard target updates. The source weights are copied
    directly to the target network.

    Parameters
    ----------
    target: keras.models.Model
      The target model. Should have same architecture as source model.
    source: keras.models.Model
      The source model. Should have same architecture as target model.

    Returns
    -------
    list(tf.Tensor)
      List of tensor update ops.
    """
    target.set_weights(source.get_weights())

    return target


def memory_burn_in(env, memory, preprocessors, burn_in_size):
    """
    Do the memory burn in
    """

    curr_state = env.reset()
    processed_curr_state = preprocessors.process_state_for_memory(curr_state)
    is_terminate = False
    print("start memory burn in")
    index = 0
    while(len(memory) < burn_in_size):
        #select action
        action = env.action_space.sample()
        next_state, reward, is_terminal, debug_info = env.step(action)
        processed_next_state = preprocessors.process_state_for_memory(next_state)
        #append the current state to memory
        memory.insert(processed_curr_state,processed_next_state,action, preprocessors.process_reward(reward), is_terminal)
        if(is_terminal):
            curr_state = env.reset()
            processed_next_state = preprocessors.process_state_for_memory(curr_state)
            
        #move to next state
        processed_curr_state = processed_next_state
        sys.stdout.write("\rburning in: {}/{}".format(len(memory), burn_in_size))
        sys.stdout.flush()

    print("\nfinished memory burn in")

def rgb2gray(rgb):
    #http://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

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