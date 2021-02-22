import tensorflow as tf
import numpy as np
import cv2

from mask_detection.dataset import load_data
from mask_detection.model import yolo_model
from utils import gpu_init, high_confidence_vector, convert_yolo_to_abs, on_batch_end_callback
from mask_detection.common import target_width, target_height, grid_width_ratio, grid_height_ratio

step = 0
step_interval = 50
batch_size = 2


if __name__ == '__main__':
    gpu_init()

    x, y = load_data()
    model = yolo_model()

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
        epochs=100,
        batch_size=batch_size,
        validation_split=.2,
        callbacks=[tf.keras.callbacks.LambdaCallback(on_batch_end=on_batch_end)])
