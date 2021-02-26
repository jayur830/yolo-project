import tensorflow as tf
import cv2
import os
import math

from loon_yolov3.dataset import load_data
from loon_yolov3.model import yolo_model
from utils import high_confidence_vector, convert_yolo_to_abs
from loon_yolov3.common import \
    target_width, \
    target_height, \
    grid_width_ratio, \
    grid_height_ratio, \
    anchor_width, \
    anchor_height, \
    category_index

step = 0
step_interval = 20
batch_size = 2

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_data()

    model = yolo_model()

    def on_batch_end(_1, logs):
        global step, step_interval
        if step >= x_train.shape[0]:
            step = 0
        elif step % step_interval == 0:
            img = x_train[step].copy()
            x = cv2.resize(
                src=img,
                dsize=(target_width, target_height),
                interpolation=cv2.INTER_AREA)
            x = x.reshape((1,) + x.shape)
            output = model.predict(x)
            vectors = high_confidence_vector(output[0])
            for vector in vectors:
                x1, y1, x2, y2 = \
                    (vector[2] - vector[4] * .5) * target_width / grid_width_ratio, \
                    (vector[3] - vector[5] * .5) * target_height / grid_height_ratio, \
                    (vector[2] + vector[4] * .5) * target_width / grid_width_ratio, \
                    (vector[3] + vector[5] * .5) * target_height / grid_height_ratio
                img = cv2.rectangle(
                    img=img,
                    pt1=(round(x1), round(y1)),
                    pt2=(round(x2), round(y2)),
                    color=(0, 0, 255),
                    thickness=2)
                img = cv2.putText(
                    img=img,
                    text=category_index[vector[-1]],
                    org=(round(x1), round(y1)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=.5,
                    color=(0, 0, 255),
                    thickness=2)
            cv2.imshow("test", img)
            cv2.waitKey(1)
        step += batch_size

    model.fit(
        x=x_train,
        y=y_train,
        epochs=200,
        batch_size=batch_size,
        validation_split=.2,
        callbacks=[
            tf.keras.callbacks.LambdaCallback(on_batch_end=on_batch_end)
        ])

    model.save(filepath="model.h5")
    model = tf.keras.models.load_model(filepath="model.h5", compile=False)

    for i in range(x_test.shape[0]):
        img = x_test[i].copy()
        output = model.predict(x_test[i].reshape((1,) + x_test[i].shape))
        vectors = high_confidence_vector(output[0])
        for vector in vectors:
            x1, y1, x2, y2 = \
                (vector[2] - vector[4] * .5) * target_width / grid_width_ratio, \
                (vector[3] - vector[5] * .5) * target_height / grid_height_ratio, \
                (vector[2] + vector[4] * .5) * target_width / grid_width_ratio, \
                (vector[3] + vector[5] * .5) * target_height / grid_height_ratio
            img = cv2.rectangle(
                img=img,
                pt1=(round(x1), round(y1)),
                pt2=(round(x2), round(y2)),
                color=(0, 0, 255),
                thickness=2)
            img = cv2.putText(
                img=img,
                text=category_index[vector[-1]],
                org=(round(x1), round(y1) - 5),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=.5,
                color=(0, 0, 255),
                thickness=2)
        print("=" * 40)
        cv2.imshow("test", img)
        cv2.waitKey()
