
from deep.core import ReplayMemory, Sample
import numpy as np
import sys

class ActionSample(Sample):
    """
    Implementation of the Sample Class 
    """
    def __init__(self, state, action, reward,is_terminal):
        """
        Initialization, we assumed the state is already preprocessed
        """
        self._state = state
        self._action = action
        self._reward = reward
        self._is_terminal = is_terminal

    def is_terminal(self):
        return self._is_terminal

class ActionReplayMemory(ReplayMemory):

    def __init__(self, max_size, window_length):
        """
        Set up the replay memory
        """
        self._max_size = max_size
        self._window_length = window_length
        self.clear()

    def __len__(self):
        return self._filled_size

    def size(self):
        return sys.getsizeof(self._memory)

    def _increment_index(self):
        self._filled_size = np.max([self._filled_size, (self._index+1)]) #make sure we know how many of them are there
        self._index = (self._index + 1)% self._max_size #increment the index

    def insert(self, cur_state, next_state, action, reward, is_terminal):
        self.append(cur_state,action,reward)
        if(is_terminal):
            self.end_episode(next_state,is_terminal)


    def append(self, state, action, reward):

        if(self._state_size == None):
            self._state_size = np.shape(state)
        new_sample = ActionSample(state, action, reward, False)
        self._memory[self._index] = new_sample
        self._increment_index()

    def end_episode(self, final_state,is_terminal):

        new_sample = ActionSample(final_state,None,None,True)
        self._memory[self._index] = new_sample
        self._increment_index()
        #Not sure how this helps


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
            safety = 0
            #check if terminal, if we got a terminal sample, we roll back to the previous frame
            while(sample.is_terminal()):
                i = (i - 1)%self._filled_size
                sample = self._memory[i]
                safety += 1
                if(safety == self._filled_size):
                    #this mean all of the states are terminal states, which is super weird.
                    #we probably did something wrong here, or it's a one time thing
                    raise("ERROR:all memory is terminal states")

            #we might want to check that it's in a deprived state, where it doesn't have a next state
            next_sample = self._memory[(i+1)%self._filled_size]
            if(next_sample != None):
                curr_state_list.append(sample._state)
                next_state_list.append(next_sample._state)
                action_list.append(sample._action)
                reward_list.append(sample._reward)
                terminal_list.append(int(next_sample.is_terminal()))
            else:
                curr_state_list.append(sample._state)
                next_state_list.append(sample._state)
                action_list.append(sample._action)
                reward_list.append(sample._reward)
                terminal_list.append(int(True))

        #print(curr_state_list)
        return curr_state_list, next_state_list, np.array(reward_list), np.array(action_list), np.array(terminal_list)

    def clear(self):

        self._filled_size = 0
        self._index = 0 
        self._state_size = None
        self._memory = [None] * self._max_size

