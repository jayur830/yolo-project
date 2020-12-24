import tensorflow as tf


def sum_squared_error(y_true, y_pred, axis=None):
    return tf.reduce_sum(tf.square(y_true - y_pred), axis=axis)


def yolo_loss(y_true, y_pred):
    lambda_coord, lambda_noobj = 5., .5
    p_channel = y_true[:, :, :, 4]

    xy_loss = lambda_coord * tf.reduce_sum(sum_squared_error(y_true[:, :, :, :2], y_pred[:, :, :, :2], axis=-1) * p_channel)
    wh_loss = lambda_coord * tf.reduce_sum(sum_squared_error(y_true[:, :, :, 2:4] ** .5, y_pred[:, :, :, 2:4] ** .5, axis=-1) * p_channel)
    conf_loss = tf.reduce_sum(
        tf.square(y_true[0, :, :, 4] - y_pred[0, :, :, 4]) * tf.where(
            tf.cast(p_channel, dtype=tf.bool),
            tf.ones(shape=p_channel.shape[1:]),
            tf.ones(shape=p_channel.shape[1:]) * lambda_noobj))
    class_loss = 0.
    if y_true.shape[-1] > 5:
        class_loss = tf.reduce_sum(sum_squared_error(y_true[:, :, :, 5:], y_pred[:, :, :, 5:], axis=-1) * p_channel)

    return xy_loss + wh_loss + conf_loss + class_loss


def mean_absolute_log_error(y_true, y_pred, axis=None):
    return tf.reduce_mean(-tf.math.log(1 + 1e-7 - tf.abs(y_true - y_pred)), axis=axis)


def mean_squared_log_error(y_true, y_pred, axis=None):
    return tf.reduce_mean(-tf.math.log(1 + 1e-7 - tf.square(y_true - y_pred)), axis=axis)


def yolo_mean_absolute_log_error(y_true, y_pred):
    lambda_coord, lambda_noobj = 5., .5
    p_channel = y_true[:, :, :, 4]

    xy_loss = lambda_coord * tf.reduce_mean(mean_absolute_log_error(y_true[:, :, :, :2], y_pred[:, :, :, :2], axis=-1) * p_channel)
    wh_loss = lambda_coord * tf.reduce_mean(mean_absolute_log_error(y_true[:, :, :, 2:4] ** .5, y_pred[:, :, :, 2:4] ** .5, axis=-1) * p_channel)
    conf_loss = tf.reduce_mean(
        -tf.math.log(1 + 1e-7 - tf.abs(y_true[0, :, :, 4] - y_pred[0, :, :, 4])) * tf.where(
            tf.cast(p_channel, dtype=tf.bool),
            tf.ones(shape=p_channel.shape[1:]),
            tf.ones(shape=p_channel.shape[1:]) * lambda_noobj))
    class_loss = 0.
    if y_true.shape[-1] > 5:
        class_loss = tf.reduce_mean(mean_absolute_log_error(y_true[:, :, :, 5:], y_pred[:, :, :, 5:], axis=-1) * p_channel)

    return xy_loss + wh_loss + conf_loss + class_loss


def yolo_mean_squared_log_error(y_true, y_pred):
    lambda_coord, lambda_noobj = 5., .5
    p_channel = y_true[:, :, :, 4]

    xy_loss = lambda_coord * tf.reduce_mean(mean_squared_log_error(y_true[:, :, :, :2], y_pred[:, :, :, :2], axis=-1) * p_channel)
    wh_loss = lambda_coord * tf.reduce_mean(mean_squared_log_error(y_true[:, :, :, 2:4] ** .5, y_pred[:, :, :, 2:4] ** .5, axis=-1) * p_channel)
    conf_loss = tf.reduce_mean(
        -tf.math.log(1 + 1e-7 - tf.square(y_true[0, :, :, 4] - y_pred[0, :, :, 4])) * tf.where(
            tf.cast(p_channel, dtype=tf.bool),
            tf.ones(shape=p_channel.shape[1:]),
            tf.ones(shape=p_channel.shape[1:]) * lambda_noobj))
    class_loss = 0.
    if y_true.shape[-1] > 5:
        class_loss = tf.reduce_mean(mean_squared_log_error(y_true[:, :, :, 5:], y_pred[:, :, :, 5:], axis=-1) * p_channel)

    return xy_loss + wh_loss + conf_loss + class_loss
