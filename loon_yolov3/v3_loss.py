import tensorflow as tf

from losses import sum_squared_error
from loon_yolov3.common import grid_width_ratio, grid_height_ratio, anchor_width, anchor_height


def yolov3_loss(y_true, y_pred):
    lambda_coord, lambda_noobj = 5., .5
    confidence_channel = y_true[:, :, :, 4]

    xy_loss = lambda_coord * tf.reduce_sum(sum_squared_error(y_true[:, :, :, :2], tf.sigmoid(y_pred[:, :, :, :2]), axis=-1) * confidence_channel)

    w_loss = lambda_coord * tf.reduce_sum(sum_squared_error(tf.expand_dims(y_true[:, :, :, 2] ** .5, axis=-1), tf.expand_dims(anchor_width * tf.exp(y_pred[:, :, :, 2]) ** .5, axis=-1), axis=-1) * confidence_channel)
    h_loss = lambda_coord * tf.reduce_sum(sum_squared_error(tf.expand_dims(y_true[:, :, :, 3] ** .5, axis=-1), tf.expand_dims(anchor_height * tf.exp(y_pred[:, :, :, 3]) ** .5, axis=-1), axis=-1) * confidence_channel)
    # iou = iou(y_true, y_pred)
    iou = 1.
    confidence_loss = tf.reduce_sum(
        tf.square(y_true[:, :, :, 4] - iou * y_pred[:, :, :, 4]) * tf.where(
            tf.cast(confidence_channel, dtype=tf.bool),
            tf.ones(shape=tf.shape(input=confidence_channel)),
            tf.ones(shape=tf.shape(input=confidence_channel)) * lambda_noobj))
    class_loss = tf.reduce_sum(sum_squared_error(y_true[:, :, :, 5:], y_pred[:, :, :, 5:], axis=-1) * confidence_channel)

    return xy_loss + w_loss + h_loss + confidence_loss + class_loss


def iou(y_true, y_pred) -> tf.Tensor:
    a_x1 = y_true[:, :, :, 0] - y_true[:, :, :, 2] * .5
    a_y1 = y_true[:, :, :, 1] - y_true[:, :, :, 3] * .5
    a_x2 = y_true[:, :, :, 0] + y_true[:, :, :, 2] * .5
    a_y2 = y_true[:, :, :, 1] + y_true[:, :, :, 3] * .5

    b_x1 = y_pred[:, :, :, 0] - y_pred[:, :, :, 2] * .5
    b_y1 = y_pred[:, :, :, 1] - y_pred[:, :, :, 3] * .5
    b_x2 = y_pred[:, :, :, 0] + y_pred[:, :, :, 2] * .5
    b_y2 = y_pred[:, :, :, 1] + y_pred[:, :, :, 3] * .5

    intersection = (tf.minimum(a_x2, b_x2) - tf.maximum(a_x1, b_x1)) * (tf.minimum(a_y2, b_y2) - tf.maximum(a_y1, b_y1))
    union = y_true[:, :, :, 2] * y_true[:, :, :, 3] + y_pred[:, :, :, 2] * y_pred[:, :, :, 3] - intersection

    return intersection / union


import numpy as np

if __name__ == '__main__':
    arr = np.arange(5) + np.zeros(shape=(3, 5))
    print(arr)
    print(tf.transpose(tf.range(3, dtype=tf.float32) + tf.zeros(shape=(5, 3))))
