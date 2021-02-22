import tensorflow as tf

from wildlife.common import path
from wildlife.dataset import load_data
from wildlife.model import model

batch_size = 2
epochs = 100

if __name__ == '__main__':
    with open(f"{path}/classes.names", "r") as reader:
        classes = [line[:-1] for line in reader.readlines()[:-1]]

    (train_x, train_y), (test_x, test_y) = load_data()
    model = model(len(classes))

    def on_batch_end(_, logs):
        pass

    model.fit(
        x=train_x,
        y=train_y,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[tf.keras.callbacks.LambdaCallback(on_batch_end=on_batch_end)])
