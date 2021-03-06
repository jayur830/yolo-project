import tensorflow as tf

from losses import YOLOLoss
from output_layer import YOLOOutput


def yolo_model(
        anchors: [[float]],
        num_classes: int,
        kernel_initializer: str = "he_normal",
        learning_rate: float = 1e-3,
        bn_momentum: float = .9,
        lrelu_alpha: float = .1):
    # (416, 416, 3)
    input_layer = tf.keras.layers.Input(shape=(416, 416, 3))

    # (416, 416, 3) -> (208, 208, 8)
    model = tf.keras.layers.Conv2D(
        filters=8,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer=kernel_initializer,
        use_bias=False)(input_layer)
    model = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(model)
    model = tf.keras.layers.LeakyReLU(alpha=lrelu_alpha)(model)
    # (208, 208, 8) -> (104, 104, 16)
    model = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer=kernel_initializer,
        use_bias=False)(model)
    model = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(model)
    model = tf.keras.layers.LeakyReLU(alpha=lrelu_alpha)(model)
    # (104, 104, 16) -> (52, 52, 32)
    model = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer=kernel_initializer,
        use_bias=False)(model)
    model = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(model)
    model = tf.keras.layers.LeakyReLU(alpha=lrelu_alpha)(model)
    # (52, 52, 32) -> (26, 26, 64)
    model = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer=kernel_initializer,
        use_bias=False)(model)
    model = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(model)
    model = tf.keras.layers.LeakyReLU(alpha=lrelu_alpha)(model)
    # (26, 26, 64) -> (13, 13, 128)
    model = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer=kernel_initializer,
        use_bias=False)(model)
    model = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(model)
    model = tf.keras.layers.LeakyReLU(alpha=lrelu_alpha)(model)
    # (13, 13, 128) -> (13, 13, 5 + num_classes)
    model = tf.keras.layers.Conv2D(
        filters=5 * len(anchors) + num_classes,
        kernel_size=1,
        kernel_initializer=kernel_initializer)(model)
    model = YOLOOutput(len(anchors))(model)

    model = tf.keras.models.Model(input_layer, model)

    model.summary()
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        loss=YOLOLoss(anchors))

    return model
