import cv2
import numpy as np

from yolo.loon.common import grid_width, grid_height
from yolo.yolo_utils import convert_abs_to_yolo
from glob import glob


def load_data():
    data_x, data_y = [], []
    target_width, target_height = 512, 128

    img_list = glob(f"./loon_rpn_split/*.jpg")
    img_list = [filename.split("\\")[-1] for filename in img_list]
    img_list = [filename[:filename.index(".jpg")] for filename in img_list]

    for filename in img_list:
        img = cv2.imread(f"./loon_rpn_split/{filename}.jpg")
        img = cv2.resize(
            src=img,
            dsize=(target_width, target_height))
        data_x.append(img)
        with open(f"./loon_rpn_split/{filename}.txt", "r") as reader:
            labels = reader.readlines()
            labels = [label.split(" ") for label in labels]
            labels = [(int(label[0]), int(float(label[1]) * img.shape[1]), int(float(label[2]) * img.shape[0]), int(float(label[3]) * img.shape[1]), int(float(label[4]) * img.shape[0])) for label in labels]

        label_tensor = np.zeros(shape=(grid_height, grid_width, 9), dtype=np.float32)
        for label in labels:
            x1, y1, x2, y2 = \
                target_width * (label[1] - int(label[3] / 2)) / img.shape[1], \
                target_height * (label[2] - int(label[4] / 2)) / img.shape[0], \
                target_width * (label[1] + int(label[3] / 2)) / img.shape[1], \
                target_height * (label[2] + int(label[4] / 2)) / img.shape[0]
            grid_x, grid_y, x, y, w, h = convert_abs_to_yolo(target_width, target_height, grid_width, grid_height, [x1, y1, x2 - x1, y2 - y1])

            # x1, y1, x2, y2 = convert_yolo_to_abs(target_width, target_height, grid_width, grid_height, [grid_x, grid_y, x, y, w, h])

            # print(x, y, w, h)
            # input('a')
            label_tensor[grid_y, grid_x, 0] = 1.
            label_tensor[grid_y, grid_x, 1] = x
            label_tensor[grid_y, grid_x, 2] = y
            label_tensor[grid_y, grid_x, 3] = w
            label_tensor[grid_y, grid_x, 4] = h
            label_tensor[grid_y, grid_x, 5 + int(label[0])] = 1.

            # img = cv2.rectangle(
            #     img=img,
            #     pt1=(int(x1), int(y1)),
            #     pt2=(int(x2), int(y2)),
            #     color=(0, 0, 255),
            #     thickness=2)

        data_y.append(label_tensor)

        # cv2.imshow("test", img)
        # cv2.waitKey()

    return np.asarray(data_x), np.asarray(data_y)
