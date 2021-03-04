import tensorflow as tf

from losses import yolo_loss
from traffic_signs.common import target_width, target_height


def yolo_model(
        num_classes: int,
        kernel_initializer: str = "he_normal",
        learning_rate: float = 1e-3,
        bn_momentum: float = .9,
        lrelu_alpha: float = .1):
    # (256, 512, 3)
    input_layer = tf.keras.layers.Input(shape=(target_height, target_width, 3))

    # (256, 512, 3) -> (128, 256, 8)
    model = tf.keras.layers.SeparableConv2D(
        filters=8,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer=kernel_initializer,
        use_bias=False)(input_layer)
    model = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(model)
    model = tf.keras.layers.LeakyReLU(alpha=lrelu_alpha)(model)
    # (128, 256, 8) -> (64, 128, 16)
    model = tf.keras.layers.SeparableConv2D(
        filters=16,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer=kernel_initializer,
        use_bias=False)(model)
    model = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(model)
    model = tf.keras.layers.LeakyReLU(alpha=lrelu_alpha)(model)
    # (64, 128, 16) -> (32, 64, 32)
    model = tf.keras.layers.SeparableConv2D(
        filters=32,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer=kernel_initializer,
        use_bias=False)(model)
    model = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(model)
    model = tf.keras.layers.LeakyReLU(alpha=lrelu_alpha)(model)
    # (32, 64, 32) -> (16, 32, 64)
    model = tf.keras.layers.SeparableConv2D(
        filters=64,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer=kernel_initializer,
        use_bias=False)(model)
    model = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(model)
    model = tf.keras.layers.LeakyReLU(alpha=lrelu_alpha)(model)
    # (16, 32, 64) -> (8, 16, 128)
    model = tf.keras.layers.SeparableConv2D(
        filters=128,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer=kernel_initializer,
        use_bias=False)(model)
    model = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(model)
    model = tf.keras.layers.LeakyReLU(alpha=lrelu_alpha)(model)
    # (8, 16, 128) -> (8, 16, 5 + num_classes)
    model = tf.keras.layers.SeparableConv2D(
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
