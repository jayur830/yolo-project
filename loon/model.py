import tensorflow as tf

from yolo.loon.custom_loss import FPWeightedError


def mean_absolute_log_error( y_true, y_pred):
    from tensorflow.python.framework.ops import convert_to_tensor_v2
    y_pred = convert_to_tensor_v2(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    mae = tf.math.abs(tf.math.subtract(y_pred, y_true))
    sub = tf.math.subtract(tf.constant(1.0 + 1e-7), mae)
    log = -tf.math.log(sub)
    return tf.keras.backend.mean(log, axis=-1)

def model(
        kernel_initializer="he_normal",
        learning_rate=.001):
    input_layer = tf.keras.layers.Input(shape=(128, 512, 3))

    # (128, 512, 3) -> (128, 512, 8)
    model = tf.keras.layers.Conv2D(
        filters=8,
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer,
        use_bias=False)(input_layer)
    model = tf.keras.layers.ReLU()(model)
    model = tf.keras.layers.BatchNormalization()(model)
    # (128, 512, 8) -> (64, 256, 8)
    model = tf.keras.layers.MaxPool2D()(model)

    # (64, 256, 8) -> (64, 256, 16)
    model = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer)(model)
    model = tf.keras.layers.ReLU()(model)
    model = tf.keras.layers.BatchNormalization()(model)
    # (64, 256, 16) -> (32, 128, 16)
    model = tf.keras.layers.MaxPool2D()(model)

    # (32, 128, 16) -> (32, 128, 32)
    model = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer)(model)
    model = tf.keras.layers.ReLU()(model)
    model = tf.keras.layers.BatchNormalization()(model)
    # (32, 128, 32) -> (16, 64, 32)
    model = tf.keras.layers.MaxPool2D()(model)

    # (16, 64, 32) -> (16, 64, 9)
    model = tf.keras.layers.Conv2D(
        filters=9,
        kernel_size=1,
        kernel_initializer=kernel_initializer)(model)
    model = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(model)

    model = tf.keras.models.Model(input_layer, model)
    model.compile(
        # optimizer=tf.keras.optimizers.Nadam(learning_rate=learning_rate),
        # optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=.9,
            nesterov=False),
        loss=tf.losses.binary_crossentropy)
        # loss=FPWeightedError())
        # loss=tf.keras.losses.MeanAbsoluteError())
        # loss=MeanAbsoluteLogError())
        # loss=tf.keras.losses.BinaryCrossentropy())
        # loss=tf.keras.losses.binary_crossentropy)
    model.summary()

    return model
