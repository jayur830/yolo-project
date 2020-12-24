import tensorflow as tf

from utils import yolo_loss
from loon.vars import target_width, target_height


def yolo_model(kernel_initializer: str = "he_normal"):
    input_layer = tf.keras.layers.Input(shape=(target_height, target_width, 3))

    # (128, 512, 3) -> (128, 512, 8)
    model = tf.keras.layers.Conv2D(
        filters=8,
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer,
        use_bias=False)(input_layer)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.LeakyReLU(alpha=1e-2)(model)
    model = tf.keras.layers.DepthwiseConv2D(
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer,
        use_bias=False)(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.LeakyReLU(alpha=1e-2)(model)
    # (128, 512, 8) -> (64, 256, 8)
    model = tf.keras.layers.MaxPool2D()(model)
    # (64, 256, 8) -> (64, 256, 16)
    model = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer,
        use_bias=False)(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.LeakyReLU(alpha=1e-2)(model)
    model = tf.keras.layers.DepthwiseConv2D(
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer,
        use_bias=False)(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.LeakyReLU(alpha=1e-2)(model)
    # (64, 256, 16) -> (32, 128, 16)
    model = tf.keras.layers.MaxPool2D()(model)
    # (32, 128, 16) -> (32, 128, 32)
    model = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer,
        use_bias=False)(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.LeakyReLU(alpha=1e-2)(model)
    model = tf.keras.layers.DepthwiseConv2D(
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer,
        use_bias=False)(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.LeakyReLU(alpha=1e-2)(model)
    # (32, 128, 32) -> (16, 64, 32)
    model = tf.keras.layers.MaxPool2D()(model)
    # (16, 64, 32) -> (16, 64, 9)
    model = tf.keras.layers.Conv2D(
        filters=9,
        kernel_size=1,
        kernel_initializer=kernel_initializer)(model)
    model = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(model)

    model = tf.keras.models.Model(input_layer, model)
    model.summary()
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=.01),
        loss=yolo_loss,
        # loss=mean_absolute_log_error,
        # loss=sum_squared_error,
        # loss=tf.keras.losses.mean_squared_error,
        # loss=tf.keras.losses.mean_squared_logarithmic_error,
        # loss=tf.keras.losses.mean_absolute_error,
        # loss=tf.keras.losses.mean_absolute_percentage_error,
        # loss=tf.keras.losses.binary_crossentropy,
        metrics=["acc"])

    return model
