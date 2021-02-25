import tensorflow as tf

from losses import yolo_loss
from wildlife.common import target_width, target_height


def model(
        num_classes: int,
        kernel_initializer="he_normal",
        bn_momentum=.9,
        lrelu_alpha=.1,
        learning_rate=1e-2):
    # (416, 416, 3)
    input_layer = tf.keras.layers.Input(shape=(target_height, target_width, 3))
    # (416, 416, 3) -> (208, 208, 8)
    x = tf.keras.layers.SeparableConv2D(
        filters=8,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer=kernel_initializer,
        use_bias=False)(input_layer)
    x = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(x)
    x = tf.keras.layers.LeakyReLU(alpha=lrelu_alpha)(x)
    # (208, 208, 8) -> (104, 104, 16)
    x = tf.keras.layers.SeparableConv2D(
        filters=16,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer=kernel_initializer,
        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(x)
    x = tf.keras.layers.LeakyReLU(alpha=lrelu_alpha)(x)
    # (104, 104, 16) -> (52, 52, 32)
    x = tf.keras.layers.SeparableConv2D(
        filters=32,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer=kernel_initializer,
        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(x)
    x = tf.keras.layers.LeakyReLU(alpha=lrelu_alpha)(x)
    # (52, 52, 32) -> (26, 26, 64)
    x = tf.keras.layers.SeparableConv2D(
        filters=64,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer=kernel_initializer,
        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(x)
    x = tf.keras.layers.LeakyReLU(alpha=lrelu_alpha)(x)
    # (26, 26, 64) -> (13, 13, 128)
    x = tf.keras.layers.SeparableConv2D(
        filters=128,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer=kernel_initializer,
        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(x)
    x = tf.keras.layers.LeakyReLU(alpha=lrelu_alpha)(x)
    # (13, 13, 128) -> (13, 13, 5 + num_classes)
    x = x = tf.keras.layers.Conv2D(
        filters=5 + num_classes,
        kernel_size=1,
        kernel_initializer=kernel_initializer)(x)
    x = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(x)

    model = tf.keras.models.Model(input_layer, x)
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        loss=yolo_loss,
        metrics=[tf.metrics.Recall()])
    model.summary()

    return model
