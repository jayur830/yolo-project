import tensorflow as tf


class YoloOutput(tf.keras.layers.Layer):
    def __init__(self, grid_width, grid_height, p_w, p_h, **kwargs):
        super(YoloOutput, self).__init__(False, **kwargs)
        self.__grid_width = grid_width
        self.__grid_height = grid_height
        self.__p_w = p_w
        self.__p_h = p_h

    def build(self, input_shape):
        super(YoloOutput, self).build(input_shape)
        self.built = True

    @tf.function
    def call(self, inputs, **kwargs):
        t_x, t_y, t_w, t_h = inputs[:, :, :, 0], inputs[:, :, :, 1], inputs[:, :, :, 2], inputs[:, :, :, 3]
        b_x = tf.expand_dims(tf.sigmoid(t_x) + self.__grid_width, axis=-1)
        b_y = tf.expand_dims(tf.sigmoid(t_y) + self.__grid_height, axis=-1)
        b_w = tf.expand_dims(self.__p_w * tf.exp(t_w), axis=-1)
        b_h = tf.expand_dims(self.__p_h * tf.exp(t_h), axis=-1)
        return tf.concat([b_x, b_y, b_w, b_h, inputs[:, :, :, 4:]], axis=-1, name="yolo_output")
