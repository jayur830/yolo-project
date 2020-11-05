import tensorflow as tf


def model(
        kernel_initializer="he_normal",
        relu_alpha=0.01,
        learning_rate=0.005,
        momentum=0.9):
    input_layer = tf.keras.layers.Input(shape=(224, 224, 3))

    # (224, 224, 3) -> (224, 224, 8)
    model = tf.keras.layers.Conv2D(
        filters=8,
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer,
        use_bias=False)(input_layer)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.LeakyReLU(alpha=relu_alpha)(model)
    # (224, 224, 8) -> (112, 112, 8)
    model = tf.keras.layers.MaxPool2D()(model)

    # (112, 112, 8) -> (112, 112, 16)
    model = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer)(model)
    model = tf.keras.layers.LeakyReLU(alpha=relu_alpha)(model)
    # (112, 112, 16) -> (56, 56, 16)
    model = tf.keras.layers.MaxPool2D()(model)

    # (56, 56, 16) -> (56, 56, 32)
    model = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer)(model)
    model = tf.keras.layers.LeakyReLU(alpha=relu_alpha)(model)
    # (56, 56, 32) -> (28, 28, 32)
    model = tf.keras.layers.MaxPool2D()(model)

    # (28, 28, 32) -> (28, 28, 64)
    model = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer)(model)
    model = tf.keras.layers.LeakyReLU(alpha=relu_alpha)(model)
    # (28, 28, 64) -> (14, 14, 64)
    model = tf.keras.layers.MaxPool2D()(model)

    # (14, 14, 64) -> (14, 14, 128)
    model = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer)(model)
    model = tf.keras.layers.LeakyReLU(alpha=relu_alpha)(model)
    # (14, 14, 128) -> (7, 7, 128)
    model = tf.keras.layers.MaxPool2D()(model)

    # (7, 7, 128) -> (7, 7, 5)
    loc_regressor = tf.keras.layers.Conv2D(
        filters=3,
        kernel_size=1,
        kernel_initializer=kernel_initializer)(model)
    loc_regressor = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(loc_regressor)

    # (7, 7, 128) -> (7, 7, 5)
    size_regressor = tf.keras.layers.Conv2D(
        filters=2,
        kernel_size=1,
        kernel_initializer=kernel_initializer)(model)
    size_regressor = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(size_regressor)

    model = tf.keras.models.Model(input_layer, [loc_regressor, size_regressor])
    model.compile(
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=momentum,
            nesterov=True),
        loss=[
            tf.losses.binary_crossentropy,
            tf.losses.mean_absolute_error,
            tf.losses.binary_crossentropy
        ])
    model.summary()

    return model
