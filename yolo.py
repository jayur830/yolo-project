import tensorflow as tf

from losses import sum_squared_error


class YOLO:
    def __init__(self):
        self.__model = self.build()

    def compile(self, optimizer: tf.optimizers.Optimizer):
        self.__model.compile(
            optimizer=optimizer,
            loss=YOLO.loss)

    def fit(self, x, y, batch_size, epochs, callbacks):
        return self.__model.fit(
            x=x,
            y=y,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks)

    @staticmethod
    def loss(y_true, y_pred):
        from tensorflow.python.framework.ops import convert_to_tensor_v2
        y_pred = convert_to_tensor_v2(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        lambda_coord, lambda_noobj = 5., .5
        p_channel = y_true[:, :, :, 4]

        xy_loss = lambda_coord * tf.reduce_sum(sum_squared_error(y_true[:, :, :, :2], y_pred[:, :, :, :2], axis=-1) * p_channel)
        wh_loss = lambda_coord * tf.reduce_sum(sum_squared_error(y_true[:, :, :, 2:4] ** .5, y_pred[:, :, :, 2:4] ** .5, axis=-1) * p_channel)
        conf_loss = tf.reduce_sum(
            tf.square(y_true[:, :, :, 4] - y_pred[:, :, :, 4]) * tf.where(
                tf.cast(p_channel, dtype=tf.bool),
                tf.ones(shape=tf.shape(input=p_channel)),
                tf.ones(shape=tf.shape(input=p_channel)) * lambda_noobj))
        class_loss = tf.reduce_sum(sum_squared_error(y_true[:, :, :, 5:], y_pred[:, :, :, 5:], axis=-1) * p_channel)

        return xy_loss + wh_loss + conf_loss + class_loss

    def build(self) -> tf.keras.models.Model:
        pass
