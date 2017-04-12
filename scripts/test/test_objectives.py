
from deeprl_hw2.objectives import huber_loss
import numpy as np
import tensorflow as tf
import unittest


class ObjectiveFuncTest(unittest.TestCase):

	def test_huber_loss(self):
		with tf.Session() as sess:
			
			#test linear components	
			y_true = tf.constant([2.0,4.0,6.0])
			y_predict = tf.constant([1.0,1.0,2.0])
			expected_val = tf.constant([0.5,2.5,3.5])
			result = huber_loss(y_true, y_predict,1.)
			opr = tf.equal(result,expected_val)
			self.assertTrue(np.all(opr.eval()))

			expected_val = tf.constant([0.46875,1.96875,2.71875])
			result = huber_loss(y_true, y_predict,0.75)
			opr = tf.equal(result,expected_val)
			self.assertTrue(np.all(opr.eval()))

			#test the small cases
			y_true = tf.constant([0.4, 0.5, 0.6, 0.4,0.1])
			y_predict = tf.constant([0.1,0.5,0.5,0.2,0.4])
			result = (huber_loss(y_true, y_predict,1.))
			expected_val = tf.constant([0.045,0,0.005,0.02,0.045])
			self.assertTrue(np.all(np.isclose(result.eval(),expected_val.eval())))

			y_true = tf.constant([0.4, 0.5, 0.6, 0.4,0.1])
			y_predict = tf.constant([0.1,0.5,0.5,0.2,0.4])
			result = (huber_loss(y_true, y_predict,0.45))
			expected_val = tf.constant([0.045,0,0.005,0.02,0.045])
			self.assertTrue(np.all(np.isclose(result.eval(),expected_val.eval())))


if __name__ == '__main__':
	unittest.main()