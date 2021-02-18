import tensorflow as tf


class LayerWrapper:
    @staticmethod
    def bn_relu(inputs, momentum=.9):
        x = tf.keras.layers.BatchNormalization(momentum=momentum)(inputs)
        return tf.keras.layers.ReLU()(x)

    @staticmethod
    def bn_lrelu(inputs, momentum=.9, alpha=.1):
        x = tf.keras.layers.BatchNormalization(momentum=momentum)(inputs)
        return tf.keras.layers.LeakyReLU(alpha=alpha)(x)

    @staticmethod
    def bn_prelu(inputs, momentum=.9, kernel_initializer="he_normal"):
        x = tf.keras.layers.BatchNormalization(momentum=momentum)(inputs)
        return tf.keras.layers.PReLU(alpha_initializer=kernel_initializer)(x)
