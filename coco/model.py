import tensorflow as tf

from losses import yolo_loss
from coco.layer_wrapper import LayerWrapper
from coco.common import target_width, target_height

import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def yolo_model(
        num_classes,
        kernel_initializer="he_normal",
        learning_rate=1e-3):
    # (416, 416, 3)
    input_layer = tf.keras.layers.Input(shape=(target_height, target_width, 3))
    # (416, 416, 3) -> (208, 208, 8)
    model = tf.keras.layers.SeparableConv2D(
        filters=8,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer=kernel_initializer,
        use_bias=False)(input_layer)
    model = LayerWrapper.bn_lrelu(model)
    # (208, 208, 8) -> (104, 104, 16)
    model = tf.keras.layers.SeparableConv2D(
        filters=16,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer=kernel_initializer,
        use_bias=False)(model)
    model = LayerWrapper.bn_lrelu(model)
    # (104, 104, 16) -> (52, 52, 32)
    model = tf.keras.layers.SeparableConv2D(
        filters=32,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer=kernel_initializer,
        use_bias=False)(model)
    model = LayerWrapper.bn_lrelu(model)
    # (52, 52, 32) -> (26, 26, 64)
    model = tf.keras.layers.SeparableConv2D(
        filters=64,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer=kernel_initializer,
        use_bias=False)(model)
    model = LayerWrapper.bn_lrelu(model)
    # (26, 26, 64) -> (13, 13, 128)
    model = tf.keras.layers.SeparableConv2D(
        filters=128,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer=kernel_initializer,
        use_bias=False)(model)
    model = LayerWrapper.bn_lrelu(model)
    # (13, 13, 128) -> (13, 13, 5 + num_classes)
    model = tf.keras.layers.SeparableConv2D(
        filters=5 + num_classes,
        kernel_size=1,
        kernel_initializer=kernel_initializer)(model)
    model = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(model)

    model = tf.keras.models.Model(input_layer, model)
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        loss=yolo_loss,
        metrics=[tf.metrics.Recall()])
    model.summary()

    return model
