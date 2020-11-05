import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    x = np.random.randn(1, 15)

    dense_layer = tf.keras.layers.Dense(units=10, kernel_initializer="he_uniform", use_bias=False, input_shape=(15,))

    bn_model = tf.keras.models.Sequential([
        dense_layer,
        tf.keras.layers.BatchNormalization()
    ])
    model1 = tf.keras.models.Sequential([
        dense_layer,
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=.01)
    ])
    model2 = tf.keras.models.Sequential([
        dense_layer,
        tf.keras.layers.LeakyReLU(alpha=.01),
        tf.keras.layers.BatchNormalization()
    ])
    model2.summary()
    print(f"Input:\n{x}")
    print(f"BN:\n{bn_model.predict(x)}")
    print(f"BN -> ReLU:\n{model1.predict(x)}")
    print(f"ReLU -> BN:\n{model2.predict(x)}")
    print(bn_model.layers[1].trainable_weights)
