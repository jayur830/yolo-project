import tensorflow as tf
import numpy as np
import cv2

from coco.data_generator import YOLODataGenerator
from coco.model import yolo_model
from utils import high_confidence_vector

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
        target_size=(416, 416),
        grid_ratio=(13, 13),
        batch_size=2)

    yolo = yolo_model(num_classes=len(classes))

    # print(yolo(data_gen[0]))

    # @tf.function
    # def predict(model, x):
    #     return model(x)
    #
    # def imshow(_, logs):
    #     img = data_gen[step][0][0]
    #     output = np.asarray(predict(yolo, img))
    #     vectors = high_confidence_vector(output[0])
    #     print(vectors)

    yolo.fit(
        x=data_gen,
        batch_size=batch_size,
        epochs=epochs)
        # callbacks=[tf.keras.callbacks.LambdaCallback(on_batch_end=imshow)])
