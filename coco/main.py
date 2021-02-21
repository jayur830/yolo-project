import tensorflow as tf
import numpy as np
import cv2

from coco.data_generator import YOLODataGenerator
from coco.model import yolo_model
from utils import high_confidence_vector, convert_yolo_to_abs

target_width, target_height = 416, 416
grid_width_ratio, grid_height_ratio = 13, 13
batch_size = 2
epochs = 100
step = 0

if __name__ == '__main__':
    with open("E:/Dataset/image/coco/classes.txt", "r") as reader:
        classes = [line[:-1] for line in reader.readlines()]
    data_gen = YOLODataGenerator(
        x_paths="E:/Dataset/image/coco/train2017/*.jpg",
        y_paths="E:/Dataset/image/coco/train2017/*.txt",
        num_classes=len(classes),
        target_size=(target_width, target_height),
        grid_ratio=(grid_width_ratio, grid_height_ratio),
        batch_size=batch_size)

    yolo = yolo_model(num_classes=len(classes))

    @tf.function
    def predict(x):
        return yolo(x)

    def on_batch_end(_, logs):
        global batch_size, step
        img = data_gen[step][0][0].reshape((1,) + data_gen[step][0][0].shape)
        output = np.asarray(predict(img))
        vectors = high_confidence_vector(output[0])
        for vector in vectors:
            grid_x, grid_y, x, y, w, h, c = vector
            x1, y1, x2, y2 = convert_yolo_to_abs(target_width, target_height, grid_width_ratio, grid_height_ratio, [grid_x, grid_y, x, y, w, h])
            img = cv2.rectangle(
                img=img,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=(0, 0, 255),
                thickness=2)
            img = cv2.putText(
                img=img,
                text=classes[c],
                org=(round(x1), round(y1) - 5),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=.5,
                color=(0, 0, 255),
                thickness=2)
        cv2.imshow("train", img.reshape(img.shape[1:]))
        cv2.waitKey(1)
        step += batch_size

    def on_epoch_end(_, logs):
        global step
        step = 0

    yolo.fit(
        x=data_gen,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.LambdaCallback(
                on_batch_end=on_batch_end,
                on_epoch_end=on_epoch_end)
        ])

    yolo.save(filepath="coco_yolo.h5")
