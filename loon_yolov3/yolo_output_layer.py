import tensorflow as tf


class YoloOutputLayer(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        bbox, others = inputs[:, :, :, :4], inputs[:, :, :, 4:]
        return tf.concat([bbox, tf.sigmoid(others)], axis=-1)