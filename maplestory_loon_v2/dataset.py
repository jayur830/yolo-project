from glob import glob

import cv2
import numpy as np

from yolo.maplestory_loon_v2.yolo_converter import convert_abs_to_yolo
from yolo.maplestory_loon_v2.common import factor, grid_width, grid_height


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

        label_tensor = np.zeros(shape=(grid_height, grid_width, 10))
        label_tensor[:, :, 9] = 1.
        for label in labels:
            x1, y1, x2, y2 = \
                target_width * (label[1] - int(label[3] / 2)) / img.shape[1], \
                target_height * (label[2] - int(label[4] / 2)) / img.shape[0], \
                target_width * (label[1] + int(label[3] / 2)) / img.shape[1], \
                target_height * (label[2] + int(label[4] / 2)) / img.shape[0]
            grid_x, grid_y, x, y, w, h = convert_abs_to_yolo(target_width, target_height, grid_width, grid_height, [x1, y1, x2 - x1, y2 - y1])

            label_tensor[grid_y, grid_x, 0] = x
            label_tensor[grid_y, grid_x, 1] = y
            label_tensor[grid_y, grid_x, 2] = w
            label_tensor[grid_y, grid_x, 3] = h
            label_tensor[grid_y, grid_x, 4] = 1.
            label_tensor[grid_y, grid_x, 9] = 0
            label_tensor[grid_y, grid_x, 5 + int(label[0])] = 1.
        data_y.append(label_tensor * factor)

    return np.array(data_x), np.array(data_y)
