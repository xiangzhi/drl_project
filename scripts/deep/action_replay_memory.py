
from deep.core import ReplayMemory, Sample
import numpy as np
import copy
import sys

class ActionSample(Sample):
    """
    Implementation of the Sample Class 
    """
    def __init__(self, state, action, reward,is_terminal, next_state):
        """
        Initialization, we assumed the state is already preprocessed
        """
        self._state = copy.deepcopy(state)
        self._action = copy.deepcopy(action)
        self._reward = copy.deepcopy(reward)
        self._next_state = copy.deepcopy(next_state)
        self._is_terminal = copy.deepcopy(is_terminal)

    def is_terminal(self):
        return self._is_terminal

class ActionReplayMemory(ReplayMemory):

    def __init__(self, max_size, seed_val):
        """
        Set up the replay memory
        """
        self._max_size = max_size
        np.random.seed(seed_val) #seed the randomness
        self.clear()

    def __len__(self):
        return self._filled_size

    def size(self):
        return len(self)

    def _increment_index(self):
        self._filled_size = np.max([self._filled_size, (self._index+1)]) #make sure we know how many of them are there
        self._index = (self._index + 1)% self._max_size #increment the index

    def insert(self, cur_state, next_state, action, reward, is_terminal):
        new_sample = ActionSample(cur_state,action,reward,is_terminal,next_state)
        self._memory[self._index] = new_sample
        self._increment_index()

    #     self.append(cur_state,action,reward)
    #     if(is_terminal):
    #         self.end_episode(next_state,is_terminal)
    def append(self, state, action, reward):
        pass

    # def append(self, state, action, reward):

    #     if(self._state_size == None):
    #         self._state_size = np.shape(state)
    #     new_sample = ActionSample(state, action, reward, False)
    #     self._memory[self._index] = new_sample
    #     self._increment_index()

    # def end_episode(self, final_state,is_terminal):

    #     new_sample = ActionSample(final_state,None,None,True)
    #     self._memory[self._index] = new_sample
    #     self._increment_index()
    #     #Not sure how this helps


    def sample(self, batch_size, indexes=None):
        if(indexes == None):
            if(self._filled_size != 0):
                indexes = (np.random.randint(0,self._filled_size, size=batch_size)).tolist()
            else:
                indexes = []
        #the return sample will be in the form of [state, nxt_state, reward, action, is_terminal]
        curr_state_list = []
        next_state_list = []
        action_list = []
        terminal_list = []
        reward_list = []
        for i in indexes:
            #get sample
            sample = self._memory[i]
            #append to list
            curr_state_list.append(sample._state)
            next_state_list.append(sample._next_state)
            action_list.append(sample._action)
            reward_list.append(sample._reward)
            terminal_list.append(sample._is_terminal)

            # safety = 0
            # #check if terminal, if we got a terminal sample, we roll back to the previous frame
            # while(sample.is_terminal()):
            #     i = (i - 1)%self._filled_size
            #     sample = self._memory[i]
            #     safety += 1
            #     if(safety == self._filled_size):
            #         #this mean all of the states are terminal states, which is super weird.
            #         #we probably did something wrong here, or it's a one time thing
            #         raise("ERROR:all memory is terminal states")

            # #we might want to check that it's in a deprived state, where it doesn't have a next state
            # next_sample = self._memory[(i+1)%self._filled_size]
            # if(next_sample != None):
            #     curr_state_list.append(sample._state)
            #     next_state_list.append(next_sample._state)
            #     action_list.append(sample._action)
            #     reward_list.append(sample._reward)
            #     terminal_list.append(int(next_sample.is_terminal()))
            # else:
            #     curr_state_list.append(sample._state)
            #     next_state_list.append(sample._state)
            #     action_list.append(sample._action)
            #     reward_list.append(sample._reward)
            #     terminal_list.append(int(True))

        return curr_state_list, next_state_list, action_list, reward_list, terminal_list

    def clear(self):

        self._filled_size = 0
        self._index = 0 
        self._state_size = None
        self._memory = [None] * self._max_size

