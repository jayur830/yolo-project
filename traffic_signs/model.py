import tensorflow as tf

from losses import yolo_loss
from traffic_signs.common import path, target_width, target_height, grid_width_ratio, grid_height_ratio


import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'


def traffic_sign_model(
        kernel_initializer="he_normal",
        learning_rate=1e-3):
    with open(f"{path}/classes.names") as reader:
        classes = [class_name[:-1] for class_name in reader.readlines()]

    # (256, 512, 3)
    input_layer = tf.keras.layers.Input(shape=(target_height, target_width, 3))
    # (256, 512, 3) -> (128, 256, 8)
    x = tf.keras.layers.SeparableConv2D(
        filters=8,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer=kernel_initializer,
        use_bias=False)(input_layer)
    x = tf.keras.layers.BatchNormalization(momentum=.9)(x)
    x = tf.keras.layers.LeakyReLU(alpha=.1)(x)
    # (128, 256, 8) -> (64, 128, 16)
    x = tf.keras.layers.SeparableConv2D(
        filters=16,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer=kernel_initializer,
        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(momentum=.9)(x)
    x = tf.keras.layers.LeakyReLU(alpha=.1)(x)
    # (64, 128, 16) -> (32, 64, 32)
    x = tf.keras.layers.SeparableConv2D(
        filters=32,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer=kernel_initializer,
        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(momentum=.9)(x)
    x = tf.keras.layers.LeakyReLU(alpha=.1)(x)
    # (32, 64, 32) -> (16, 32, 64)
    x = tf.keras.layers.SeparableConv2D(
        filters=64,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer=kernel_initializer,
        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(momentum=.9)(x)
    x = tf.keras.layers.LeakyReLU(alpha=.1)(x)
    # (16, 32, 64) -> (8, 16, 128)
    x = tf.keras.layers.SeparableConv2D(
        filters=128,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer=kernel_initializer,
        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(momentum=.9)(x)
    x = tf.keras.layers.LeakyReLU(alpha=.1)(x)
    # (8, 16, 128) -> (8, 16, 5 + num_classes)
    x = tf.keras.layers.SeparableConv2D(
        filters=5 + len(classes),
        kernel_size=1,
        kernel_initializer=kernel_initializer)(x)
    x = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(x)

    _model = tf.keras.models.Model(input_layer, x)
    _model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        loss=yolo_loss,
        metrics=[tf.metrics.Recall()])
    _model.summary()

    return _model
