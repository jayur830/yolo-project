import tensorflow as tf
import cv2

from mask_detection.dataset import load_data
from mask_detection.model import yolo_model
from utils import gpu_init, high_confidence_vector, convert_yolo_to_abs
from mask_detection.vars import target_width, target_height, grid_width_ratio, grid_height_ratio

step = 0
step_interval = 50
batch_size = 2


if __name__ == '__main__':
    gpu_init()

    x_data, y_data = load_data()
    model = yolo_model()

    def imshow(_1, logs):
        global step, step_interval
        if step >= x_data.shape[0]:
            step = 0
        elif step % step_interval == 0:
            img = x_data[step].copy()
            x = cv2.resize(
                src=img,
                dsize=(target_width, target_height),
                interpolation=cv2.INTER_AREA)
            x = x.reshape((1,) + x.shape)
            output = model.predict(x)
            vectors = high_confidence_vector(output[0])
            for vector in vectors:
                x1, y1, x2, y2 = convert_yolo_to_abs(target_width, target_height, grid_width_ratio, grid_height_ratio, vector[:-1])
                img = cv2.rectangle(
                    img=img,
                    pt1=(int(x1), int(y1)),
                    pt2=(int(x2), int(y2)),
                    color=(0, 0, 255),
                    thickness=2)
            cv2.imshow("test", img)
            cv2.waitKey(1)
        step += batch_size

    model.fit(
        x=x_data,
        y=y_data,
        epochs=100,
        batch_size=batch_size,
        validation_split=.2,
        callbacks=[tf.keras.callbacks.LambdaCallback(on_batch_end=imshow)]
    )
