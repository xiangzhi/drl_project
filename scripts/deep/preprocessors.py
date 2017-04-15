"""Suggested Preprocessors."""

import numpy as np

import utils
from .core import Preprocessor
from scipy import misc

import collections

class HistoryPreprocessor(Preprocessor):
    """Keeps the last k states.

    Useful for domains where you need velocities, but the state
    contains only positions.

    When the environment starts, this will just fill the initial
    sequence values with zeros k times.

    Parameters
    ----------
    history_length: int
      Number of previous states to prepend to state being processed.

    """

    def __init__(self, history_length=1):
        self._history_length = history_length
        self._name = "history_preprocessor"
        self.reset()

    def process_state_for_network(self, state):
        """You only want history when you're deciding the current action to take."""

        return self._hidden_processor(state)

    def _hidden_processor(self, state):
        #push it onto the list
        self._state_buffer.appendleft(state)
        #just return as a list
        return list(self._state_buffer)

    def process_state_for_memory(self,state):
        return self._hidden_processor(state)

    def process_batch(self, samples):
        return samples

    def reset(self):
        """Reset the history sequence.

        Useful when you start a new episode.
        """
        self._state_buffer = collections.deque([0]* self._history_length, maxlen=self._history_length)

    def process_reward(self, reward):
        return reward

    def get_config(self):
        return {'history_length': self._history_length}

    def clone(self):
        """
        Return an copied clean instance of itself
        """
        return HistoryPreprocessor(self._history_length)


class BaxterPreprocessor(Preprocessor):
    """
    downsample images
    """
    def __init__(self, new_size):
        self._new_size = new_size

    def process_state_for_memory(self, state):
        """
        The incoming state will be a tuple
        """
        joint_angles = state[0]
        image = state[1]
        #convert the image and downsample
        image = self._image_processing(image, np.uint8)
        #Scale, convert to greyscale and store as uint8.
        return (joint_angles, image)


    def process_state_for_network(self, state):
        """Scale, convert to greyscale and store as float32.

        Basically same as process state for memory, but this time
        outputs float32 images.
        """
        joint_angles = state[0]
        image = state[1]
        #convert the image and downsample
        image = self._image_processing(image, np.float32)
        #Scale, convert to greyscale and store as uint8.
        return (joint_angles, image)

    def _image_processing(self, state, dtype):
        """
        Scale and convert to gray scale plus change to dtype
        """
        #convert from rgb to greyscale
        #image_np = utils.rgb2gray(state)
        #resize and return
        return (misc.imresize(state,self._new_size)).astype(dtype)
        

    def process_batch(self, samples):
        """The batches from replay memory will be uint8, convert to float32.

        Same as process_state_for_network but works on a batch of
        samples from the replay memory. Meaning you need to convert
        both state and next state values.
        """
        new_list = []
        for sample in samples:
            new_list.append((sample[0], np.float32(sample[1])))
        return new_list

    def process_reward(self, reward):
        """
        We know the max score given the reward function is like 80 * 80 = 6400. We scale the reward to between [-1 and 1]
        """
        if(reward > 0):
            reward = reward/6400.0
            reward = 1 if(reward > 1) else reward
        return reward

    def clone(self):
        """
        Return a copied instance of itself
        """
        return BaxterPreprocessor(self._new_size)


class NumpyPreprocessor(Preprocessor):
    """
    Converts the state into a numpy object
    """
    def __init__(self):
        self._name = "numpy_preprocessor"

    def process_state_for_network(self, state):
        return self.converter(state, np.float32)

    def process_state_for_memory(self, state):
        return self.converter(state, np.uint8)


    def converter(self, state, dtype):

        
        size = len(state)
        image = np.zeros((np.size(state[0][1],0), np.size(state[0][1],1),np.size(state[0][1],2)), dtype=dtype)
        image[:,:,:] = state[0][1] #we only store the most recent image
        joint_arr = np.zeros((np.size(state[0][0]),size))
        for i in range(0,size):
            if(state[i] == 0):
                continue;
            i_state = state[i]
            #image_arr[:,:,:,i] = i_state[1]
            joint_arr[:,i] = i_state[0]  
        return (joint_arr, image)

    def clone(self):
        return NumpyPreprocessor()

class PreprocessorSequence(Preprocessor):
    """You may find it useful to stack multiple prepcrocesosrs (such as the History and the AtariPreprocessor).

    You can easily do this by just having a class that calls each preprocessor in succession.

    For example, if you call the process_state_for_network and you
    have a sequence of AtariPreprocessor followed by
    HistoryPreprocessor. This this class could implement a
    process_state_for_network that does something like the following:

    state = atari.process_state_for_network(state)
    return history.process_state_for_network(state)

    preprocessors()

    """
    def __init__(self, preprocessors):
        self._preprocessors = preprocessors

    def get_preprocessors_name_list(self):
        name_list = []
        for preprocessor in self._preprocessors:
            name_list.append(preprocessor._name)
        return name_list

    def process_state_for_network(self, state):
        """
        preprocess a state 
        """
        for preprocessor in self._preprocessors:
            state = preprocessor.process_state_for_network(state)
        
        return state

    def process_state_for_memory(self, state):
        """
        preprocess a state 
        """
        for preprocessor in self._preprocessors:
            state = preprocessor.process_state_for_memory(state)
        
        return state

    def process_batch(self, samples):
        """
        preprocess a state 
        """
        for preprocessor in self._preprocessors:
            samples = preprocessor.process_batch(samples)
        
        return samples        

    def process_reward(self, reward):
        for preprocessor in self._preprocessors:
            reward = preprocessor.process_reward(reward)
        return reward     

    def clone(self):

        list_preprocessors = []
        for preprocessor in self._preprocessors:
            list_preprocessors.append(preprocessor.clone())

        return PreprocessorSequence(list_preprocessors)
        
    def reset(self): 
        """
        Reset
        """
        for preprocessor in self._preprocessors:
            preprocessor.reset()
