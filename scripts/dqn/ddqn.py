import time
import numpy as np
import copy
import json
import sys

from deeprl_hw2.utils import get_hard_target_model_updates, clone_keras_model
from deeprl_hw2.policy import GreedyEpsilonPolicy
from deeprl_hw2.dqn import DQNAgent

"""Main DQN agent."""

class DDQNAgent(DQNAgent):
    """
    Implements DDQN agent
    """
    def __init__(self,
                 q_network,
                 preprocessor,
                 memory,
                 policy,
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 train_freq,
                 batch_size):
        super().__init__(q_network,
                 preprocessor,
                 memory,
                 policy,
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 train_freq,
                 batch_size)

    def update_policy(self, total_step_num):
        """Update your policy.

        Behavior may differ based on what stage of training your
        in. If you're in training mode then you should check if you
        should update your network parameters based on the current
        step and the value you set for train_freq.

        Inside, you'll want to sample a minibatch, calculate the
        target values, update your network, and then update your
        target values.

        You might want to return the loss and other metrics as an
        output. They can help you monitor how training is going.
        """

        #sample lists of current state, future states and reward, action from the replay memory
        curr_state_arr, next_state_arr, reward_arr, action_arr, terminal_arr  = self._replay_memory.sample(self._batch_size)
        #process all the inputs to make sure they are in the right format
        curr_state_arr = self._preprocessors.process_batch(curr_state_arr)
        next_state_arr = self._preprocessors.process_batch(next_state_arr)
        #create empty array to store q_values
        target_q_values = np.zeros((np.size(reward_arr,0),self._action_size))
        #get the current q_values using the calculate q function
        target_q_values = self.calc_q_values(curr_state_arr)
        #update each target

        for i, q_values in enumerate(target_q_values):
          target_q_values[i,action_arr[i]] = reward_arr[i]
          if(terminal_arr[i] == 0):
          
            action = np.argmax(self._network.predict(np.array([next_state_arr[i]]), batch_size=1))
            max_q_value = self._target_network.predict(np.array([next_state_arr[i]]), batch_size=1)
            target_q_values[i,action_arr[i]] = target_q_values[i,action_arr[i]] + self._gamma * max_q_value[0,action]

        #train the network
        training_loss = self._network.train_on_batch(curr_state_arr, target_q_values)

        #check whether we want to update the target network
        if(total_step_num % self._target_update_freq == 0):
          self._target_network = get_hard_target_model_updates(self._target_network, self._network)

        return training_loss