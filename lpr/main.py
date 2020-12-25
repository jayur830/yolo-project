import tensorflow as tf
import cv2

from lpr.dataset import load_data
from lpr.model import yolo_model
from utils import gpu_init, high_confidence_vector, convert_yolo_to_abs
from lpr.vars import target_width, target_height, grid_width_ratio, grid_height_ratio

step = 0
step_interval = 10
batch_size = 2


if __name__ == '__main__':
    # gpu_init()

    (x_train, y_train), (x_test, y_test) = load_data()
    model = yolo_model()

    def imshow(_1, logs):
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
                x1, y1, x2, y2 = convert_yolo_to_abs(target_width, target_height, grid_width_ratio, grid_height_ratio, vector)
                img = cv2.rectangle(
                    img=img,
                    pt1=(round(x1), round(y1)),
                    pt2=(round(x2), round(y2)),
                    color=(0, 0, 255),
                    thickness=2)
                img = cv2.putText(
                    img=img,
                    text="License plate",
                    org=(round(x1), round(y1) - 5),
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
        epochs=100,
        batch_size=batch_size,
        validation_split=.2,
        callbacks=[tf.keras.callbacks.LambdaCallback(on_batch_end=imshow)])

    for i in range(x_test.shape[0]):
        img = x_test[i].copy()
        output = model.predict(x_test[i].reshape((1,) + x_test[i].shape))
        vectors = high_confidence_vector(output[0])
        for vector in vectors:
            x1, y1, x2, y2 = convert_yolo_to_abs(target_width, target_height, grid_width_ratio, grid_height_ratio, vector)
            img = cv2.rectangle(
                img=img,
                pt1=(round(x1), round(y1)),
                pt2=(round(x2), round(y2)),
                color=(0, 0, 255),
                thickness=2)
            img = cv2.putText(
                img=img,
                text="License plate",
                org=(round(x1), round(y1) - 5),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=.5,
                color=(0, 0, 255),
                thickness=2)
        cv2.imshow("test", img)
        cv2.waitKey()
