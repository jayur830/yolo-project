import tensorflow as tf
import cv2
import os

from dataset_utils import load_data
from utils import on_batch_end, high_confidence_vector
from loon.common import \
    target_width, \
    target_height, \
    grid_width_ratio, \
    grid_height_ratio, \
    anchor_width, \
    anchor_height
from loon.model import yolo_model

batch_size = 2
epochs = 200

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test), classes = load_data(
        paths=["D:/Dataset/loon_rpn_split"],
        # paths = ["E:/Dataset/image/loon_rpn_split"],
        target_width=target_width,
        target_height=target_height,
        grid_width_ratio=grid_width_ratio,
        grid_height_ratio=grid_height_ratio,
        anchor_width=anchor_width,
        anchor_height=anchor_height)
    model = yolo_model(len(classes))

    model.fit(
        x=x_train,
        y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=.2,
        callbacks=[tf.keras.callbacks.LambdaCallback(on_batch_end=on_batch_end(
            model=model,
            x_data=x_train,
            classes=classes,
            batch_size=batch_size,
            target_width=target_width,
            target_height=target_height,
            grid_width_ratio=grid_width_ratio,
            grid_height_ratio=grid_height_ratio,
            anchor_width=anchor_width,
            anchor_height=anchor_height,
            step_interval=20))])

    model.save(filepath="model.h5")
    model = tf.keras.models.load_model(filepath="model.h5", compile=False)

    for i in range(x_test.shape[0]):
        img = x_test[i].copy()
        output = model.predict(x_test[i].reshape((1,) + x_test[i].shape))
        vectors = high_confidence_vector(output[0])
        for vector in vectors:
            print(vector)
            c_x, c_y, t_x, t_y, t_w, t_h, class_index = vector
            b_x = (t_x + c_x) * target_width / grid_width_ratio
            b_y = (t_y + c_y) * target_height / grid_height_ratio
            b_w = t_w * target_width * anchor_width / grid_width_ratio
            b_h = t_h * target_height * anchor_height / grid_height_ratio
            x1, y1, x2, y2 = int(b_x - b_w * .5), int(b_y - b_h * .5), int(b_x + b_w * .5), int(b_y + b_h * .5)

            img = cv2.rectangle(
                img=img,
                pt1=(round(x1), round(y1)),
                pt2=(round(x2), round(y2)),
                color=(0, 0, 255),
                thickness=2)
            img = cv2.putText(
                img=img,
                text=classes[class_index],
                org=(round(x1), round(y1) - 5),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=.5,
                color=(0, 0, 255),
                thickness=2)
        print("=" * 40)
        cv2.imshow("test", img)
        cv2.waitKey()
