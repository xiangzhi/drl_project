import numpy as np

class NoiseGenerator(object):
    def generate(self, state, time_step):
        raise NotImplementedError('This method should be overridden')


class OU_Generator(NoiseGenerator):

    def __init__(self, means, theta=0.15, sigma=0.2):

        self._means = means
        self._thetas = np.ones(means.shape) * theta
        self._sigma = np.ones(means.shape) * sigma


    def generate(self, action, step):

        wiener = np.random.randn(np.size(action))
        noise = self._thetas * (self._means - action) + self._sigma * wiener

        return noise
        """
        means = np.zeros(np.shape(action)) #all means are zero, so no movement around
        thetas = np.ones(np.shape(action)) * 0.15
        sigma = np.ones(np.shape(action)) * 0.2

        #calculate the weiner number
        wiener = np.random.randn(np.size(action)) #we can do this because Wiener process has a gaussian increment

        Noise = thetas * (means - action) + sigma * wiener

        #return action + np.array([(1. / (1. + time_step))])

        return action + Noise 
        """