from glob import glob

import cv2
import numpy as np

from yolo.yolo_converter import convert_abs_to_yolo


def load_data():
    data_x, data_y_loc, data_y_size, data_y_cls = [], [], [], []
    target_width, target_height = 512, 128

    img_list = glob(f"./maplestory_loon/loon_rpn_split/*.jpg")
    img_list = [filename.split("\\")[-1] for filename in img_list]
    img_list = [filename[:filename.index(".jpg")] for filename in img_list]

    for filename in img_list:
        img = cv2.imread(f"./maplestory_loon/loon_rpn_split/{filename}.jpg")
        img = cv2.resize(
            src=img,
            dsize=(target_width, target_height))
        data_x.append(img)
        with open(f"./maplestory_loon/loon_rpn_split/{filename}.txt", "r") as reader:
            labels = reader.readlines()
            labels = [label.split(" ") for label in labels]
            labels = [(int(label[0]), int(float(label[1]) * img.shape[1]), int(float(label[2]) * img.shape[0]), int(float(label[3]) * img.shape[1]), int(float(label[4]) * img.shape[0])) for label in labels]

        grid_width, grid_height = 16, 4

        loc_tensor = np.zeros(shape=(grid_height, grid_width, 3))
        size_tensor = np.zeros(shape=(grid_height, grid_width, 2))
        cls_tensor = np.zeros(shape=(grid_height, grid_width, 5))
        cls_tensor[:, :, 4] = 1.
        for label in labels:
            x1, y1, x2, y2 = \
                target_width * (label[1] - int(label[3] / 2)) / img.shape[1], \
                target_height * (label[2] - int(label[4] / 2)) / img.shape[0], \
                target_width * (label[1] + int(label[3] / 2)) / img.shape[1], \
                target_height * (label[2] + int(label[4] / 2)) / img.shape[0]
            grid_x, grid_y, x, y, w, h = convert_abs_to_yolo(target_width, target_height, grid_width, grid_height, [x1, y1, x2 - x1, y2 - y1])

            loc_tensor[grid_y, grid_x, 0] = x
            loc_tensor[grid_y, grid_x, 1] = y
            loc_tensor[grid_y, grid_x, 2] = 1.
            size_tensor[grid_y, grid_x, 0] = w
            size_tensor[grid_y, grid_x, 1] = h
            cls_tensor[grid_y, grid_x, 4] = 0
            cls_tensor[grid_y, grid_x, int(label[0])] = 1.
        data_y_loc.append(loc_tensor)
        data_y_size.append(size_tensor)
        data_y_cls.append(cls_tensor)

    return np.array(data_x), np.array(data_y_loc), np.array(data_y_size), np.array(data_y_cls)
