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
    input_layer = tf.keras.layers.Input(shape=(224, 224, 3))

    # (224, 224, 3) -> (224, 224, 8)
    model = tf.keras.layers.Conv2D(
        filters=8,
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer,
        use_bias=False)(input_layer)
    model = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(model)
    model = tf.keras.layers.LeakyReLU(alpha=lrelu_alpha)(model)
    model = tf.keras.layers.DepthwiseConv2D(
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer,
        use_bias=False)(model)
    model = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(model)
    model = tf.keras.layers.LeakyReLU(alpha=lrelu_alpha)(model)
    # (224, 224, 8) -> (112, 112, 8)
    model = tf.keras.layers.MaxPool2D()(model)
    # (112, 112, 8) -> (112, 112, 16)
    model = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer,
        use_bias=False),
    model = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(model)
    model = tf.keras.layers.LeakyReLU(alpha=lrelu_alpha)(model)
    model = tf.keras.layers.DepthwiseConv2D(
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer,
        use_bias=False)(model)
    model = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(model)
    model = tf.keras.layers.LeakyReLU(alpha=lrelu_alpha)(model)
    # (112, 112, 16) -> (56, 56, 16)
    model = tf.keras.layers.MaxPool2D()(model)
    # (56, 56, 16) -> (56, 56, 32)
    model = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer,
        use_bias=False)(model)
    model = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(model)
    model = tf.keras.layers.LeakyReLU(alpha=lrelu_alpha)(model)
    model = tf.keras.layers.DepthwiseConv2D(
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer,
        use_bias=False)(model)
    model = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(model)
    model = tf.keras.layers.LeakyReLU(alpha=lrelu_alpha)(model)
    # (28, 28, 32) -> (28, 28, 32)
    model = tf.keras.layers.MaxPool2D()(model)
    # (28, 28, 32) -> (28, 28, 64)
    model = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer,
        use_bias=False)(model)
    model = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(model)
    model = tf.keras.layers.LeakyReLU(alpha=.1)(model)
    model = tf.keras.layers.DepthwiseConv2D(
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer,
        use_bias=False)(model)
    model = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(model)
    model = tf.keras.layers.LeakyReLU(alpha=.1)(model)
    # (28, 28, 64) -> (14, 14, 64)
    model = tf.keras.layers.MaxPool2D()(model)
    # (14, 14, 64) -> (14, 14, 8)
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
