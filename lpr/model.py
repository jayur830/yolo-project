import tensorflow as tf

from losses import yolo_loss, yolo_mean_absolute_log_error, yolo_mean_squared_log_error, YoloLoss
from lpr.common import target_width, target_height


def yolo_model(
        num_classes: int,
        kernel_initializer: str = "he_normal",
        learning_rate: float = 1e-3,
        bn_momentum: float = .9,
        lrelu_alpha: float = .1):
    input_layer = tf.keras.layers.Input(shape=(target_height, target_width, 3))

    # (368, 640, 3) -> (184, 320, 8)
    model = tf.keras.layers.Conv2D(
        filters=8,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer=kernel_initializer,
        use_bias=False)(input_layer)
    model = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(model)
    model = tf.keras.layers.LeakyReLU(alpha=lrelu_alpha)(model)
    model = tf.keras.layers.Dropout(rate=5e-2)(model)
    # (184, 320, 8) -> (92, 160, 16)
    model = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer=kernel_initializer,
        use_bias=False)(model)
    model = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(model)
    model = tf.keras.layers.LeakyReLU(alpha=lrelu_alpha)(model)
    # (92, 160, 16) -> (46, 80, 32)
    model = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer=kernel_initializer,
        use_bias=False)(model)
    model = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(model)
    model = tf.keras.layers.LeakyReLU(alpha=lrelu_alpha)(model)
    # (46, 80, 32) -> (23, 40, 64)
    model = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer=kernel_initializer,
        use_bias=False)(model)
    model = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(model)
    model = tf.keras.layers.LeakyReLU(alpha=lrelu_alpha)(model)
    # (23, 40, 64) -> (23, 40, 5)
    model = tf.keras.layers.Conv2D(
        filters=5 + num_classes,
        kernel_size=1,
        kernel_initializer=kernel_initializer)(model)
    """
    x, y: sigmoid
    w, h: exp
    confidence, classes: sigmoid
    """
    model = tf.keras.layers.Lambda(lambda x: tf.concat([tf.sigmoid(x[:, :, :, :2]), tf.exp(x[:, :, :, 2:4]), tf.sigmoid(x[:, :, :, 4:])], axis=-1))(model)

    model = tf.keras.models.Model(input_layer, model)

    model.summary()
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        loss=yolo_loss)

    return model
