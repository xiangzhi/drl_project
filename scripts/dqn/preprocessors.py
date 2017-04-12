"""Suggested Preprocessors."""

import numpy as np

from deeprl_hw2 import utils
from deeprl_hw2.core import Preprocessor
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
        """Scale, convert to greyscale and store as uint8.

        We don't want to save floating point numbers in the replay
        memory. We get the same resolution as uint8, but use a quarter
        to an eigth of the bytes (depending on float32 or float64)

        We recommend using the Python Image Library (PIL) to do the
        image conversions.
        """

        return self._internal_process(state, np.uint8)


    def process_state_for_network(self, state):
        """Scale, convert to greyscale and store as float32.

        Basically same as process state for memory, but this time
        outputs float32 images.
        """
        
        return self._internal_process(state, np.float32)


    def _internal_process(self, state, dtype):
        """
        Scale and convert to gray scale plus change to dtype
        """
        #convert from rgb to greyscale
        #image_np = utils.rgb2gray(state)
        #resize and return
        return (misc.imresize(image_np,self._new_size)).astype(dtype)
        

    def process_batch(self, samples):
        """The batches from replay memory will be uint8, convert to float32.

        Same as process_state_for_network but works on a batch of
        samples from the replay memory. Meaning you need to convert
        both state and next state values.
        """
        return np.float32(samples)

    def process_reward(self, reward):
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
        new_arr = np.zeros((np.size(state[0],0), np.size(state[0],1),size), dtype=dtype)
        for i in range(0,size):
            new_arr[:,:,i] = state[i]  
        return new_arr

    def clone(self):
        return NumpyPreprocessor()


class AtariPreprocessor(Preprocessor):
    """Converts images to greyscale and downscales.

    Based on the preprocessing step described in:

    @article{mnih15_human_level_contr_throug_deep_reinf_learn,
    author =	 {Volodymyr Mnih and Koray Kavukcuoglu and David
                  Silver and Andrei A. Rusu and Joel Veness and Marc
                  G. Bellemare and Alex Graves and Martin Riedmiller
                  and Andreas K. Fidjeland and Georg Ostrovski and
                  Stig Petersen and Charles Beattie and Amir Sadik and
                  Ioannis Antonoglou and Helen King and Dharshan
                  Kumaran and Daan Wierstra and Shane Legg and Demis
                  Hassabis},
    title =	 {Human-Level Control Through Deep Reinforcement
                  Learning},
    journal =	 {Nature},
    volume =	 518,
    number =	 7540,
    pages =	 {529-533},
    year =	 2015,
    doi =        {10.1038/nature14236},
    url =	 {http://dx.doi.org/10.1038/nature14236},
    }

    You may also want to max over frames to remove flickering. Some
    games require this (based on animations and the limited sprite
    drawing capabilities of the original Atari).

    Parameters
    ----------
    new_size: 2 element tuple
      The size that each image in the state should be scaled to. e.g
      (84, 84) will make each image in the output have shape (84, 84).
    """

    def __init__(self, new_size, min_reward, max_reward):
        self._new_size = new_size
        self._min_reward = min_reward
        self._max_reward = max_reward
        self._name = "atari_preprocessor"

    def process_state_for_memory(self, state):
        """Scale, convert to greyscale and store as uint8.

        We don't want to save floating point numbers in the replay
        memory. We get the same resolution as uint8, but use a quarter
        to an eigth of the bytes (depending on float32 or float64)

        We recommend using the Python Image Library (PIL) to do the
        image conversions.
        """

        return self._internal_process(state, np.uint8)


    def process_state_for_network(self, state):
        """Scale, convert to greyscale and store as float32.

        Basically same as process state for memory, but this time
        outputs float32 images.
        """
        
        return self._internal_process(state, np.float32)


    def _internal_process(self, state, dtype):
        """
        Scale and convert to gray scale plus change to dtype
        """
        #convert from rgb to greyscale
        image_np = utils.rgb2gray(state)
        #resize and return
        return (misc.imresize(image_np,self._new_size)).astype(dtype)
        

    def process_batch(self, samples):
        """The batches from replay memory will be uint8, convert to float32.

        Same as process_state_for_network but works on a batch of
        samples from the replay memory. Meaning you need to convert
        both state and next state values.
        """
        return np.float32(samples)

    def process_reward(self, reward):
        """Clip reward between -1 and 1."""

        #normalize reward to range >= 0
        #norm_reward = reward - self._min_reward
        #return np.max([-1,np.min([1,1 - ((self._max_reward - float(reward))/self._max_reward)*2])])
        return 1 if (reward > 0) else 0

    def clone(self):
        """
        Return a copied instance of itself
        """
        return AtariPreprocessor(self._new_size, self._min_reward, self._max_reward)


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
