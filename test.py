import numpy as np
import tensorflow as tf


def binary_crossentropy(y_true, y_pred):
    return -tf.reduce_sum(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred)) / y_true.shape[0]


if __name__ == '__main__':
    a = np.asarray([1, 0, 1]).astype("float")
    b = np.asarray([.1, .2, .3]).astype("float")
    print(tf.losses.binary_crossentropy(a, b))
    print(binary_crossentropy(a, b))
