import unittest

#move backwards to the root
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deep.noise_generator import OU_Generator
import numpy as np
import matplotlib.pyplot as plt


class NoiseGeneratorTestMethods(unittest.TestCase):

    def test_OU_noise(self):
        """
        A basic 
        """
        gen = OU_Generator(np.zeros(3))
        arr = np.zeros((100,3))
        for i in range(0,100):
            noise = (gen.generate(np.zeros(3),i))
            arr[i] = noise



if __name__ == '__main__':
    #test_memory_size()
    unittest.main()
