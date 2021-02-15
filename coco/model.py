import tensorflow as tf

class YOLO(tf.keras.models.Model):
    def __init__(self, kernel_initializer="he_normal", input_shape=None):
        super(YOLO, self).__init__()
        self.__kernel_initializer = kernel_initializer

        model = tf.keras.layers.Input(shape=input_shape)


    def call(self, inputs, training=None, mask=None):

    def __bn_relu(self, inputs, bn_momentum=.9):
        x = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(inputs)
        x = tf.keras.layers.ReLU()(x)
        return x

    def __bn_leaky_relu(self, inputs, bn_momentum=.9, lrelu_alpha=.1):
        x = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(inputs)
        x = tf.keras.layers.LeakyReLU(alpha=lrelu_alpha)(x)
        return x
