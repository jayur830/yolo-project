import tensorflow as tf


def sum_squared_error(y_true, y_pred, axis=None):
    return tf.reduce_sum(tf.square(y_true - y_pred), axis=axis)


def yolo_loss(y_true, y_pred):
    lambda_coord, lambda_noobj = 5., .5
    p_channel = y_true[:, :, :, 4]

    xy_loss = lambda_coord * tf.reduce_sum(sum_squared_error(y_true[:, :, :, :2], y_pred[:, :, :, :2], axis=-1) * p_channel)
    # wh_loss = lambda_coord * tf.reduce_sum(sum_squared_error(y_true[:, :, :, 2:4] ** .5, y_pred[:, :, :, 2:4] ** .5, axis=-1) * p_channel)
    wh_loss = lambda_coord * tf.reduce_sum(tf.reduce_sum(4. * (tf.abs(y_true[:, :, :, 2:4] - y_pred[:, :, :, 2:4]) - .5) ** 3. + .5, axis=-1) * p_channel)
    conf_loss = tf.reduce_sum(
        tf.square(y_true[:, :, :, 4] - y_pred[:, :, :, 4]) * tf.where(
            tf.cast(p_channel, dtype=tf.bool),
            tf.ones(shape=p_channel.shape[1:]),
            tf.ones(shape=p_channel.shape[1:]) * lambda_noobj))
    # conf_loss = tf.reduce_sum(tf.square(y_true[0, :, :, 4] - y_pred[0, :, :, 4]) * p_channel)
    class_loss = 0.
    if y_true.shape[-1] > 5:
        class_loss = tf.reduce_sum(sum_squared_error(y_true[:, :, :, 5:], y_pred[:, :, :, 5:], axis=-1) * p_channel)

    return xy_loss + wh_loss + conf_loss + class_loss


class YoloLoss(tf.keras.losses.Loss):
    def __init__(self, coord=5.0):
        self.coord = coord
        super(YoloLoss, self).__init__()

    def call(self, y_true, y_pred):
        from tensorflow.python.framework.ops import convert_to_tensor_v2
        y_pred = convert_to_tensor_v2(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        p_loss = tf.reduce_sum(tf.square(y_true[:, :, :, 4] - y_pred[:, :, :, 4]))
        x_loss = tf.reduce_sum(tf.square(y_true[:, :, :, 0] - y_pred[:, :, :, 0]) * y_true[:, :, :, 4]) * self.coord
        y_loss = tf.reduce_sum(tf.square(y_true[:, :, :, 1] - y_pred[:, :, :, 1]) * y_true[:, :, :, 4]) * self.coord
        w_loss = tf.reduce_sum(tf.square(tf.sqrt(y_true[:, :, :, 2]) - tf.sqrt(y_pred[:, :, :, 2])) * y_true[:, :, :, 4]) * self.coord
        h_loss = tf.reduce_sum(tf.square(tf.sqrt(y_true[:, :, :, 3]) - tf.sqrt(y_pred[:, :, :, 3])) * y_true[:, :, :, 4]) * self.coord
        class_loss = tf.reduce_sum(tf.reduce_sum(tf.math.square(y_true[:, :, :, 5:] - y_pred[:, :, :, 5:]), axis=-1) * y_true[:, :, :, 4])
        return p_loss + x_loss + y_loss + w_loss + h_loss + class_loss


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


def grid_mean_squared_error(y_true, y_pred):
    p_channel = tf.expand_dims(y_true[0, :, :, 4], axis=-1)
    return tf.reduce_mean(tf.square(y_true - y_pred) * tf.repeat(input=p_channel, repeats=y_true.shape[-1], axis=-1))


def jotganzi_loss(y_true, y_pred):
    alpha = .5

    p_loss = tf.reduce_sum(tf.square(y_true[:, :, :, 4] - y_pred[:, :, :, 4]))
    max_p_loss = y_true.shape[1] * y_true.shape[2]
    obj_loss = tf.reduce_sum(tf.square(y_true - y_pred), axis=-1) * y_true[:, :, :, 4]
    obj_loss = tf.reduce_sum(obj_loss)
    max_obj_loss = tf.reduce_sum(y_true[:, :, :, 4]) * y_true.shape[-1] + 1e-7
    loss = (p_loss / max_p_loss) * alpha + (obj_loss / max_obj_loss) * (1. - alpha)
    # return -tf.math.log(1.0 + 1e-7 - loss)
    return p_loss + obj_loss
