"""Loss functions."""

import tensorflow as tf
import semver
import numpy as np

def huber_loss(y_true, y_pred, max_grad=1.):
    """Calculate the huber loss.

    See https://en.wikipedia.org/wiki/Huber_loss

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The huber loss.
    """
    # sess = tf.Session()
    # with sess.as_default():
    #   y_true = y_true.eval()
    #   y_pred = y_pred.eval()

    # print(y_pred)
    # print(y_pred.shape)

    # val = (y_true - y_pred)
    # abs_val = np.abs(val)

    # val[abs_val <= max_grad] = 0.5 * (val[abs_val <= max_grad] * val[abs_val <= max_grad])
    # val[abs_val > max_grad] = (max_grad * val[abs_val > max_grad]) - 0.5 * max_grad * max_grad

    # return tf.convert_to_tensor(val.eval())

    abs_diff = tf.abs(tf.subtract(y_true,y_pred))
    work_size = tf.shape(abs_diff)
    max_grad_tensor = tf.fill(work_size,max_grad)

  
    half_tensor = tf.fill(work_size,0.5)
    squared_tensor = tf.multiply(abs_diff,abs_diff)

    op1_tensor = tf.multiply(squared_tensor,half_tensor)
    op2_tensor = tf.subtract(tf.multiply(max_grad_tensor,abs_diff),tf.multiply(tf.multiply(max_grad_tensor,max_grad_tensor),half_tensor))

    result = tf.where(tf.less_equal(abs_diff,max_grad_tensor),op1_tensor,op2_tensor)
    return result


def mean_huber_loss(y_true, y_pred, max_grad=1.):
    """Return mean huber loss.

    Same as huber_loss, but takes the mean over all values in the
    output tensor.

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The mean huber loss.
    """
    pass
