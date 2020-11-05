import tensorflow as tf


def model(
        kernel_initializer="he_normal",
        relu_alpha=.01,
        learning_rate=.005):
    input_layer = tf.keras.layers.Input(shape=(128, 512, 3))

    # (128, 512, 3) -> (128, 512, 32)
    model = tf.keras.layers.Conv2D(
        filters=32,
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
        filters=32,
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
        filters=32,
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer)(model)
    model = tf.keras.layers.LeakyReLU(alpha=relu_alpha)(model)
    # (16, 64, 32) -> (8, 32, 32)
    model = tf.keras.layers.MaxPool2D()(model)

    # (8, 32, 32) -> (8, 32, 32)
    model = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer)(model)
    model = tf.keras.layers.LeakyReLU(alpha=relu_alpha)(model)
    # (8, 32, 32) -> (4, 16, 32)
    model = tf.keras.layers.MaxPool2D()(model)

    # Location Regression
    loc_regressor = tf.keras.layers.Conv2D(
        filters=3,
        kernel_size=1,
        padding="same",
        kernel_initializer=kernel_initializer)(model)
    loc_regressor = tf.keras.layers.Activation(tf.keras.activations.sigmoid, name="loc_regression")(loc_regressor)

    # Size Regression
    size_regressor = tf.keras.layers.Conv2D(
        filters=2,
        kernel_size=1,
        padding="same",
        kernel_initializer=kernel_initializer)(model)
    size_regressor = tf.keras.layers.Activation(tf.keras.activations.linear, name="size_regression")(size_regressor)

    # Classification
    classifier = tf.keras.layers.Conv2D(
        filters=5,
        kernel_size=1,
        padding="same",
        kernel_initializer=kernel_initializer)(model)
    classifier = tf.keras.layers.Softmax(name="classification")(classifier)

    model = tf.keras.models.Model(input_layer, [loc_regressor, size_regressor, classifier])
    model.compile(
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=.9,
            nesterov=True),
        loss=[
            tf.losses.mean_absolute_error,
            tf.losses.mean_absolute_error,
            tf.losses.categorical_crossentropy
        ])
    model.summary()

    return model
