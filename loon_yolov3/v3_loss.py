import tensorflow as tf

from losses import sum_squared_error
from loon_yolov3.common import anchor_width, anchor_height


def yolov3_loss(y_true, y_pred):
    lambda_coord, lambda_noobj = 5., .5
    confidence_channel = y_true[:, :, :, 4]

    xy_loss = lambda_coord * tf.reduce_sum(sum_squared_error(y_true[:, :, :, :2], tf.sigmoid(y_pred[:, :, :, :2]), axis=-1) * confidence_channel)
    w_loss = lambda_coord * tf.reduce_sum(sum_squared_error(y_true[:, :, :, 2] ** .5, (anchor_width * tf.exp(y_pred[:, :, :, 2]) ** .5), axis=-1) * confidence_channel)
    h_loss = lambda_coord * tf.reduce_sum(sum_squared_error(y_true[:, :, :, 3] ** .5, (anchor_height * tf.exp(y_pred[:, :, :, 3]) ** .5), axis=-1) * confidence_channel)
    confidence_loss = tf.reduce_sum(
        tf.square(y_true[:, :, :, 4] - iou(y_true, y_pred) * y_pred[:, :, :, 4]) * tf.where(
            tf.cast(confidence_channel, dtype=tf.bool),
            tf.ones(shape=tf.shape(input=confidence_channel)),
            tf.ones(shape=tf.shape(input=confidence_channel)) * lambda_noobj))
    class_loss = tf.reduce_sum(sum_squared_error(y_true[:, :, :, 5:], y_pred[:, :, :, 5:], axis=-1) * confidence_channel)

    return xy_loss + w_loss + h_loss + confidence_loss + class_loss


def iou(y_true, y_pred) -> tf.Tensor:
    pass
