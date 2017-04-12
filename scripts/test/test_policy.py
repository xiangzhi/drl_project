import unittest
from deeprl_hw2.policy import *
import numpy as np

class PolicyTestMethods(unittest.TestCase):

	def test_linear_decay(self):

		delay_period = 100
		initial_epsilon = 1
		final_epsilon = 0.01

		end_decay = 100
		
		decay_policy = LinearDecayGreedyEpsilonPolicy(initial_epsilon,final_epsilon,delay_period)
		self.assertEqual(decay_policy._epsilon,initial_epsilon,"Initial Epsilon Incorrect")
		for i in range(0, delay_period+1):
			decay_policy.select_action(np.array([0,1]))
		self.assertEqual(decay_policy._epsilon,final_epsilon,"Final Epsilon Incorrect")
		for i in range(0, delay_period+1):
			decay_policy.select_action(np.array([0,1]))
		self.assertEqual(decay_policy._epsilon,final_epsilon,"Epsilon change after reach final")

	def test_epsilon_greedy(self):

		q_values = np.array([0,0,-1,5,0,0])
		bucket_arr = np.zeros(np.shape(q_values))
		policy = GreedyEpsilonPolicy(0.05)
		run_time = 15000

		for i in range(0,run_time):
			action = policy.select_action(q_values)
			bucket_arr[action] += 1
		self.assertTrue(bucket_arr[3] > (bucket_arr[2]*100))
		self.assertEqual(np.argmax(bucket_arr),3)

	def test_uniform_policy(self):

		num_action = 10
		bucket_arr = np.zeros(num_action)
		run_time = 500000
		expected = run_time/num_action

		policy = UniformRandomPolicy(num_action)
		for i in range(0,run_time):
			action =  policy.select_action()
			bucket_arr[action] += 1

		diff_from_expected = bucket_arr - expected
		self.assertTrue(np.all(np.less(np.abs(diff_from_expected)/expected,np.ones(num_action)*0.01)),"random policy deviates 1% from the expected")

if __name__ == '__main__':
	unittest.main()