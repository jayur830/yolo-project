import tensorflow as tf


def sum_squared_error(y_true, y_pred, axis=None):
    return tf.reduce_sum(tf.square(y_true - y_pred), axis=axis)


def yolo_loss(y_true, y_pred):
    from tensorflow.python.framework.ops import convert_to_tensor_v2
    y_pred = convert_to_tensor_v2(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    lambda_coord, lambda_noobj = 5., .5
    p_channel = y_true[:, :, :, 4]

    xy_loss = lambda_coord * tf.reduce_sum(sum_squared_error(y_true[:, :, :, :2], y_pred[:, :, :, :2], axis=-1) * p_channel)
    # xy_loss = 0.
    wh_loss = lambda_coord * tf.reduce_sum(sum_squared_error(y_true[:, :, :, 2:4], y_pred[:, :, :, 2:4], axis=-1) * p_channel)
    # wh_loss = 0.
    # confidence_loss = tf.reduce_sum(
    #     tf.square(y_true[:, :, :, 4] - y_pred[:, :, :, 4]) * tf.where(
    #         tf.cast(p_channel, dtype=tf.bool),
    #         tf.ones(shape=tf.shape(input=p_channel)),
    #         tf.ones(shape=tf.shape(input=p_channel)) * lambda_noobj))
    confidence_loss = 0.
    # class_loss = tf.reduce_sum(sum_squared_error(y_true[:, :, :, 5:], y_pred[:, :, :, 5:], axis=-1) * p_channel)
    class_loss = 0.
    return xy_loss + wh_loss + confidence_loss + class_loss


class YoloLoss(tf.keras.losses.Loss):
    def __init__(self, coord=5.0):
        self.coord = coord
        super(YoloLoss, self).__init__()

    def call(self, y_true, y_pred):
        from tensorflow.python.framework.ops import convert_to_tensor_v2
        y_pred = convert_to_tensor_v2(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        x, y, w, h, p = 0, 1, 2, 3, 4

        p_true = y_true[:, :, :, p]
        p_loss = tf.reduce_sum(tf.square(y_true[:, :, :, p] - y_pred[:, :, :, p]))
        # box_true = 4.0 * (y_true[:, :, :, 1:5] - 0.5) ** 3.0 + 0.5
        # box_pred = 4.0 * (y_pred[:, :, :, 1:5] - 0.5) ** 3.0 + 0.5
        box_true = tf.sqrt(y_true[:, :, :, x:h + 1])
        box_pred = tf.sqrt(y_pred[:, :, :, x:h + 1])
        box_loss = tf.reduce_sum(tf.reduce_sum(tf.square(box_true - box_pred), axis=-1) * p_true)
        class_loss = tf.reduce_sum(tf.reduce_sum(tf.math.square(y_true[:, :, :, 5:] - y_pred[:, :, :, 5:]), axis=-1) * p_true)
        return p_loss + (box_loss * self.coord) + class_loss


def mean_absolute_log_error(y_true, y_pred, axis=None):
    epsilon = 1e-7
    return tf.reduce_mean(-tf.math.log(1 + epsilon - tf.abs(y_true - y_pred)), axis=axis)


class MeanAbsoluteLogError(tf.keras.losses.Loss):
    def __init__(self, axis=None):
        super(MeanAbsoluteLogError, self).__init__()
        self.__axis = axis

    def call(self, y_true, y_pred):
        return mean_absolute_log_error(y_true, y_pred, self.__axis)


def mean_squared_log_error(y_true, y_pred, axis=None):
    return tf.reduce_mean(-tf.math.log(1 + 1e-7 - tf.square(y_true - y_pred)), axis=axis)


class MeanSquaredLogError(tf.keras.losses.Loss):
    def __init__(self, axis=None):
        super(MeanSquaredLogError, self).__init__()
        self.__axis = axis

    def call(self, y_true, y_pred):
        return mean_squared_log_error(y_true, y_pred, self.__axis)


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
