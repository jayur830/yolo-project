import tensorflow as tf
import os

from utils import on_batch_end_callback
from wildlife.common import path, target_width, target_height, grid_width_ratio, grid_height_ratio
from wildlife.dataset import load_data
from wildlife.model import model

batch_size = 2
epochs = 100
step = 0
step_interval = 50

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

if __name__ == '__main__':
    with open(f"{path}/classes.txt", "r") as reader:
        classes = [line[:-1] for line in reader.readlines()]

    x, y = load_data(len(classes))
    model = model(len(classes))

    def on_batch_end(_, logs):
        global step, step_interval
        if step >= x.shape[0]:
            step = 0
        elif step % step_interval == 0:
            on_batch_end_callback(model, x, target_width, target_height, grid_width_ratio, grid_height_ratio, step)
        step += batch_size

    model.fit(
        x=x,
        y=y,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[tf.keras.callbacks.LambdaCallback(on_batch_end=on_batch_end)])
