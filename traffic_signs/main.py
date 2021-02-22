import tensorflow as tf
import numpy as np
import cv2

from time import time

from utils import high_confidence_vector, convert_yolo_to_abs
from traffic_signs.common import path, target_width, target_height, grid_width_ratio, grid_height_ratio
from traffic_signs.dataset import load_data
from traffic_signs.model import traffic_sign_model

batch_size = 2
epochs = 100
step = 0
step_interval = 2
prev_time = time()

if __name__ == '__main__':
    with open(f"{path}/classes.names", "r") as reader:
        classes = [line[:-1] for line in reader.readlines()]

    (train_x, train_y), (test_x, test_y) = load_data()

    print(train_x.shape, train_y.shape)
    print(test_x.shape, test_y.shape)

    model = traffic_sign_model()

    @tf.function
    def predict(_model, _x):
        return _model(_x)

    def on_batch_end(_, logs):
        global step, prev_time
        cur_time = time()
        if cur_time - prev_time < 0.5:
            return
        prev_time = cur_time
        if step >= train_x.shape[0]:
            step = 0
        # elif step % step_interval == 0:
        print(f'\n\n{step}\n\n')
        img = train_x[step]
        output = np.asarray(model(img.reshape((1,) + img.shape)))
        vectors = high_confidence_vector(output)
        for vector in vectors:
            x1, y1, x2, y2 = convert_yolo_to_abs(target_width, target_height, grid_width_ratio, grid_height_ratio, vector[:6])
            img = cv2.rectangle(
                img=img,
                pt1=(round(x1), round(y1)),
                pt2=(round(x2), round(y2)),
                color=(0, 0, 255),
                thickness=2)
            img = cv2.putText(
                img=img,
                text=classes[vector[6:].argmax()],
                org=(round(x1), round(y1) - 5),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=.5,
                color=(0, 0, 255),
                thickness=2)
        cv2.imshow("test", img)
        cv2.waitKey(1)
        step += 1

    def on_epoch_end(_, logs):
        global step
        # step = 0

    model.fit(
        x=train_x,
        y=train_y,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.LambdaCallback(
                on_batch_end=on_batch_end,
                on_epoch_end=on_epoch_end)
        ])
