
import unittest
import time
from keras.models import Model, Sequential
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute)

from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.ddqn import DDQNAgent
from deeprl_hw2.doubleqn import DoubleQNAgent
from deeprl_hw2.objectives import huber_loss
from keras.optimizers import Adam

from deeprl_hw2.preprocessors import PreprocessorSequence


class AgentTestMethods(unittest.TestCase):

    def test_model_creation(self):
        """
        Test to make sure each model(DQN, DDQN, DoubleQN) could be created and compiled
        """
        #create a fake keras model
        model = Sequential(name="test_model")
        model.add(Flatten(input_shape=(3,3,4)))
        model.add(Dense(units=2, activation='linear'))
        #create loss function & optimizer
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        loss_func = huber_loss
        #preprocessors
        prep = PreprocessorSequence([])

        #agent = DQNAgent(model, preprocessors, memory, policy,0.99, target_update_freq,None,train_freq,batch_size)
        dqn_agent = DQNAgent(model, prep, None, None, 0.99, None, None, None, None)
        dqn_agent.compile(optimizer, loss_func)

        ddqn_agent = DDQNAgent(model, prep, None, None, 0.99, None, None, None, None)
        dqn_agent.compile(optimizer, loss_func)

        doubleqn_agent = DoubleQNAgent(model, prep, None, None, 0.99, None, None, None, None)
        doubleqn_agent.compile(optimizer, loss_func)


if __name__ == '__main__':
    unittest.main()