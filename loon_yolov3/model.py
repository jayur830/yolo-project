import tensorflow as tf

from losses import yolo_loss, yolo_mean_squared_log_error, YoloLoss
from loon.common import target_width, target_height


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

    # (32, 128, 16) -> (32, 128, 32)
    model = tf.keras.layers.Conv2D(
        filters=64,
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
    model = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(model)

    model = tf.keras.models.Model(input_layer, model)

    # model_input = tf.keras.layers.Input(shape=(target_height, target_width, 3))
    #
    # x = tf.keras.layers.Conv2D(filters=8, kernel_size=3, kernel_initializer='he_uniform', padding='same')(model_input)
    # x = tf.keras.layers.ReLU()(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.MaxPool2D()(x)
    #
    # x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, kernel_initializer='he_uniform', padding='same')(x)
    # x = tf.keras.layers.ReLU()(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.MaxPool2D()(x)
    #
    # x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, kernel_initializer='he_uniform', padding='same')(x)
    # x = tf.keras.layers.ReLU()(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Conv2D(filters=16, kernel_size=1, kernel_initializer='he_uniform', padding='same')(x)
    # x = tf.keras.layers.ReLU()(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, kernel_initializer='he_uniform', padding='same')(x)
    # x = tf.keras.layers.ReLU()(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.MaxPool2D()(x)
    #
    # x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, kernel_initializer='he_uniform', padding='same')(x)
    # x = tf.keras.layers.ReLU()(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Conv2D(filters=32, kernel_size=1, kernel_initializer='he_uniform', padding='same')(x)
    # x = tf.keras.layers.ReLU()(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, kernel_initializer='he_uniform', padding='same')(x)
    # x = tf.keras.layers.ReLU()(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.MaxPool2D()(x)
    #
    # x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, kernel_initializer='he_uniform', padding='same')(x)
    # x = tf.keras.layers.ReLU()(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Conv2D(filters=64, kernel_size=1, kernel_initializer='he_uniform', padding='same')(x)
    # x = tf.keras.layers.ReLU()(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, kernel_initializer='he_uniform', padding='same')(x)
    # x = tf.keras.layers.ReLU()(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    #
    # x = tf.keras.layers.Conv2D(filters=9, kernel_size=1, activation='sigmoid')(x)
    # model = tf.keras.models.Model(model_input, x)


    model.summary()
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=1e-2),
        loss=yolo_loss)

    return model


if __name__ == '__main__':
    tf.keras.utils.plot_model(yolo_model(), show_shapes=True)