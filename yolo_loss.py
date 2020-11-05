import tensorflow as tf
import numpy as np


def yolo_loss(y_true, y_pred):
    lambda_coord = 5.
    # return tf.losses.binary_crossentropy(y_true, y_pred)
    # return tf.losses.mean_squared_error(y_true[:, :, :, 0:2], y_pred[:, :, :, 0:2]) + \
    #     tf.losses.mean_squared_logarithmic_error(y_true[:, :, :, 2:4], y_pred[:, :, :, 2:4]) + \
    #     tf.losses.mean_squared_error(y_true[:, :, :, 4:], y_pred[:, :, :, 4:])

    return tf.losses.binary_crossentropy(y_true[:, :, :, 0:2], y_pred[:, :, :, 0:2]) + \
        tf.losses.binary_crossentropy(tf.sqrt(y_true[:, :, :, 2:4]), tf.sqrt(y_pred[:, :, :, 2:4])) + \
        tf.losses.binary_crossentropy(y_true[:, :, :, 4:], y_pred[:, :, :, 4:])
    # return lambda_coord * tf.reduce_sum(tf.square(y_true[:, :, :, 0:2] - y_pred[:, :, :, 0:2])) + \
    #     lambda_coord * tf.reduce_sum(tf.square(tf.sqrt(y_true[:, :, :, 2:4]) - tf.sqrt(y_pred[:, :, :, 2:4]))) + \
    #     tf.reduce_sum(tf.square(y_true[:, :, :, 4:] - y_pred[:, :, :, 4:]))



