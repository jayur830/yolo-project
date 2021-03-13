import tensorflow as tf

from losses import YOLOLoss
from output_layer import YOLOOutput


def yolo_model(
        anchors: [[float]],
        num_classes: int,
        kernel_initializer: str = "he_normal",
        learning_rate: float = 1e-2,
        bn_momentum: float = .9,
        lrelu_alpha: float = .1):
    # (256, 512, 3)
    input_layer = tf.keras.layers.Input(shape=(256, 512, 3))

    # (256, 512, 3) -> (128, 256, 8)
    model = tf.keras.layers.Conv2D(
        filters=8,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer=kernel_initializer,
        use_bias=False)(input_layer)
    model = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(model)
    model = tf.keras.layers.LeakyReLU(alpha=lrelu_alpha)(model)
    # (128, 256, 8) -> (64, 128, 16)
    model = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer=kernel_initializer,
        use_bias=False)(model)
    model = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(model)
    model = tf.keras.layers.LeakyReLU(alpha=lrelu_alpha)(model)
    # (64, 128, 16) -> (32, 64, 32)
    model = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer=kernel_initializer,
        use_bias=False)(model)
    model = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(model)
    model = tf.keras.layers.LeakyReLU(alpha=lrelu_alpha)(model)
    # (32, 64, 32) -> (16, 32, 64)
    model = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer=kernel_initializer,
        use_bias=False)(model)
    model = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(model)
    model = tf.keras.layers.LeakyReLU(alpha=lrelu_alpha)(model)
    # (16, 32, 64) -> (8, 16, 128)
    model = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer=kernel_initializer,
        use_bias=False)(model)
    model = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(model)
    model = tf.keras.layers.LeakyReLU(alpha=lrelu_alpha)(model)
    # (8, 16, 128) -> (8, 16, 5 + num_classes)
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
