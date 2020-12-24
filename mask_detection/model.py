import tensorflow as tf

from losses import yolo_loss
from mask_detection.vars import target_width, target_height


def yolo_model(kernel_initializer: str = "he_normal"):
    model = tf.keras.models.Sequential([
        # (224, 224, 3) -> (224, 224, 8)
        tf.keras.layers.Conv2D(
            filters=8,
            kernel_size=3,
            padding="same",
            kernel_initializer=kernel_initializer,
            use_bias=False,
            input_shape=(target_height, target_width, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=.1),
        tf.keras.layers.DepthwiseConv2D(
            kernel_size=3,
            padding="same",
            kernel_initializer=kernel_initializer,
            use_bias=False,
            input_shape=(target_height, target_width, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=.1),
        # (224, 224, 8) -> (112, 112, 8)
        tf.keras.layers.MaxPool2D(),
        # (112, 112, 8) -> (112, 112, 16)
        tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=3,
            padding="same",
            kernel_initializer=kernel_initializer,
            use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=.1),
        tf.keras.layers.DepthwiseConv2D(
            kernel_size=3,
            padding="same",
            kernel_initializer=kernel_initializer,
            use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=.1),
        # (112, 112, 16) -> (56, 56, 16)
        tf.keras.layers.MaxPool2D(),
        # (56, 56, 16) -> (56, 56, 32)
        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            padding="same",
            kernel_initializer=kernel_initializer,
            use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=.1),
        tf.keras.layers.DepthwiseConv2D(
            kernel_size=3,
            padding="same",
            kernel_initializer=kernel_initializer,
            use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=.1),
        # (28, 28, 32) -> (28, 28, 32)
        tf.keras.layers.MaxPool2D(),
        # (28, 28, 32) -> (28, 28, 64)
        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            padding="same",
            kernel_initializer=kernel_initializer,
            use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=.1),
        tf.keras.layers.DepthwiseConv2D(
            kernel_size=3,
            padding="same",
            kernel_initializer=kernel_initializer,
            use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=.1),
        # (28, 28, 64) -> (14, 14, 64)
        tf.keras.layers.MaxPool2D(),
        # (14, 14, 64) -> (14, 14, 8)
        tf.keras.layers.Conv2D(
            filters=8,
            kernel_size=1,
            kernel_initializer=kernel_initializer),
        tf.keras.layers.Activation(tf.keras.activations.sigmoid)
    ])
    model.summary()
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=1e-2),
        loss=yolo_loss)

    return model
