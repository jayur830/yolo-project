import tensorflow as tf

from yolo.maplestory_loon_v2.common import factor


def model(
        kernel_initializer="he_normal",
        relu_alpha=.01,
        learning_rate=.005):
    input_layer = tf.keras.layers.Input(shape=(128, 512, 3))

    # (128, 512, 3) -> (128, 512, 32)
    model = tf.keras.layers.Conv2D(
        filters=8,
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer,
        use_bias=False)(input_layer)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.LeakyReLU(alpha=relu_alpha)(model)
    # (128, 512, 32) -> (64, 256, 32)
    model = tf.keras.layers.MaxPool2D()(model)

    # (64, 256, 32) -> (64, 256, 32)
    model = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer)(model)
    model = tf.keras.layers.LeakyReLU(alpha=relu_alpha)(model)
    # (64, 256, 32) -> (32, 128, 32)
    model = tf.keras.layers.MaxPool2D()(model)

    # (32, 128, 32) -> (32, 128, 32)
    model = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer)(model)
    model = tf.keras.layers.LeakyReLU(alpha=relu_alpha)(model)
    # (32, 128, 32) -> (16, 64, 32)
    model = tf.keras.layers.MaxPool2D()(model)

    # (16, 64, 32) -> (16, 64, 32)
    model = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer)(model)
    model = tf.keras.layers.LeakyReLU(alpha=relu_alpha)(model)
    # (16, 64, 32) -> (8, 32, 32)
    model = tf.keras.layers.MaxPool2D()(model)

    # (8, 32, 32) -> (8, 32, 32)
    model = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer)(model)
    model = tf.keras.layers.LeakyReLU(alpha=relu_alpha)(model)
    # (8, 32, 32) -> (4, 16, 32)
    model = tf.keras.layers.MaxPool2D()(model)

    # (4, 16, 32) -> (4, 16, 10)
    model = tf.keras.layers.Conv2D(
        filters=10,
        kernel_size=1,
        kernel_initializer=kernel_initializer)(model)
    model = tf.keras.layers.Activation(tf.keras.activations.linear)(model)

    model = tf.keras.models.Model(input_layer, model)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=.9,
            nesterov=True),
        loss=tf.losses.mean_squared_error)
    model.summary()

    return model
