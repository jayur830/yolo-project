import tensorflow as tf


def sum_squared_error(y_true, y_pred, axis=None):
    return tf.reduce_sum(tf.square(y_true - y_pred), axis=axis)


def yolo_loss_iou(y_true, y_pred):
    anchor_width, anchor_height = 5, 4

    confidence_channel = y_true[:, :, :, 4]

    y_true_shape = tf.cast(tf.shape(y_true), dtype="float32")
    num_samples, grid_width, grid_height = y_true_shape[0], y_true_shape[2], y_true_shape[1]
    anchor_width, anchor_height = anchor_width / grid_width, anchor_height / grid_height

    x_range, y_range = \
        tf.range(grid_width, dtype="float32"), \
        tf.transpose(tf.reshape(tf.range(grid_height, dtype="float32"), shape=(1, grid_height, 1)))

    c_x = confidence_channel * (x_range + tf.zeros(shape=(num_samples, grid_height, grid_width)))
    c_y = confidence_channel * (y_range + tf.zeros(shape=(num_samples, grid_height, grid_width)))

    y_true_x = y_true[:, :, :, 0] + c_x
    y_true_y = y_true[:, :, :, 1] + c_y
    y_true_w = y_true[:, :, :, 2] * grid_width
    y_true_h = y_true[:, :, :, 3] * grid_height

    y_true_x1 = y_true_x - y_true_w * .5
    y_true_y1 = y_true_y - y_true_h * .5
    y_true_x2 = y_true_x + y_true_w * .5
    y_true_y2 = y_true_y + y_true_h * .5

    y_pred_x = y_pred[:, :, :, 0] + c_x
    y_pred_y = y_pred[:, :, :, 1] + c_y
    y_pred_w = anchor_width * y_pred[:, :, :, 2] * grid_width
    y_pred_h = anchor_height * y_pred[:, :, :, 3] * grid_height

    y_pred_x1 = y_pred_x - y_pred_w * .5
    y_pred_y1 = y_pred_y - y_pred_h * .5
    y_pred_x2 = y_pred_x + y_pred_w * .5
    y_pred_y2 = y_pred_y + y_pred_h * .5

    min_x2 = tf.minimum(y_true_x2, y_pred_x2)
    max_x1 = tf.maximum(y_true_x1, y_pred_x1)
    min_y2 = tf.minimum(y_true_y2, y_pred_y2)
    max_y1 = tf.maximum(y_true_y1, y_pred_y1)

    intersection_width = tf.maximum(min_x2 - max_x1, 0)
    intersection_height = tf.maximum(min_y2 - max_y1, 0)
    intersection = intersection_width * intersection_height

    y_true_area = y_true_w * y_true_h
    y_pred_area = y_pred_w * y_pred_h

    return intersection / (y_true_area + y_pred_area - intersection)


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
    confidence_loss = tf.reduce_sum(
        tf.square(y_true[:, :, :, 4] - y_pred[:, :, :, 4]) * tf.where(
            tf.cast(p_channel, dtype=tf.bool),
            tf.ones(shape=tf.shape(input=p_channel)),
            tf.ones(shape=tf.shape(input=p_channel)) * lambda_noobj))
    # confidence_loss = 0.
    class_loss = tf.reduce_sum(sum_squared_error(y_true[:, :, :, 5:], y_pred[:, :, :, 5:], axis=-1) * p_channel)
    # class_loss = 0.
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
