import tensorflow as tf

from loon_yolov3.common import target_width, target_height, grid_width_ratio, grid_height_ratio, anchor_width, anchor_height
from losses import yolo_loss
from loon_yolov3.yolo_output_layer import YoloOutputLayer


def yolo_model(kernel_initializer: str = "he_normal"):
    input_layer = tf.keras.layers.Input(shape=(target_height, target_width, 3))

    # (128, 512, 3) -> (128, 512, 8)
    model = tf.keras.layers.Conv2D(
        filters=8,
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer,
        use_bias=False)(input_layer)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.LeakyReLU(alpha=1e-2)(model)
    model = tf.keras.layers.DepthwiseConv2D(
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer,
        use_bias=False)(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.LeakyReLU(alpha=1e-2)(model)
    # (128, 512, 8) -> (64, 256, 8)
    model = tf.keras.layers.MaxPool2D()(model)
    # (64, 256, 8) -> (64, 256, 16)
    model = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer,
        use_bias=False)(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.LeakyReLU(alpha=1e-2)(model)
    model = tf.keras.layers.DepthwiseConv2D(
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer,
        use_bias=False)(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.LeakyReLU(alpha=1e-2)(model)
    # (64, 256, 16) -> (32, 128, 16)
    model = tf.keras.layers.MaxPool2D()(model)
    # (32, 128, 16) -> (32, 128, 32)
    model = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer,
        use_bias=False)(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.LeakyReLU(alpha=1e-2)(model)
    model = tf.keras.layers.DepthwiseConv2D(
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer,
        use_bias=False)(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.LeakyReLU(alpha=1e-2)(model)
    # (32, 128, 32) -> (16, 64, 32)
    model = tf.keras.layers.MaxPool2D()(model)
    # (16, 64, 32) -> (16, 64, 9)
    model = tf.keras.layers.Conv2D(
        filters=9,
        kernel_size=1,
        kernel_initializer=kernel_initializer)(model)
    model = tf.keras.layers.Lambda(lambda x: tf.concat([x[:, :, :, :4], tf.sigmoid(x[:, :, :, 4:])], axis=-1))(model)

    model = tf.keras.models.Model(input_layer, model)

    model.summary()
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=1e-2),
        loss=yolo_loss)

    return model


if __name__ == '__main__':
    tf.keras.utils.plot_model(yolo_model(), show_shapes=True)