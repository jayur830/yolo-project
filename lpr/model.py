import tensorflow as tf

from losses import yolo_loss, yolo_mean_absolute_log_error, yolo_mean_squared_log_error
from lpr.vars import target_width, target_height


def yolo_model(kernel_initializer: str = "he_normal"):
    input_layer = tf.keras.layers.Input(shape=(target_height, target_width, 3))

    # (368, 640, 3) -> (368, 640, 8)
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
    # (368, 640, 8) -> (184, 320, 8)
    model = tf.keras.layers.MaxPool2D()(model)
    # (184, 320, 8) -> (184, 320, 16)
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
    # (184, 320, 16) -> (92, 160, 16)
    model = tf.keras.layers.MaxPool2D()(model)
    # (92, 160, 16) -> (92, 160, 32)
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
    # (92, 160, 32) -> (46, 80, 32)
    model = tf.keras.layers.MaxPool2D()(model)
    # (46, 80, 32) -> (46, 80, 64)
    model = tf.keras.layers.Conv2D(
        filters=64,
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
    # (46, 80, 64) -> (23, 40, 64)
    model = tf.keras.layers.MaxPool2D()(model)
    # (23, 40, 64) -> (23, 40, 5)
    model = tf.keras.layers.Conv2D(
        filters=5,
        kernel_size=1,
        kernel_initializer=kernel_initializer)(model)
    model = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(model)

    model = tf.keras.models.Model(input_layer, model)
    model.summary()
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=1e-2),
        loss=yolo_loss)

    return model
