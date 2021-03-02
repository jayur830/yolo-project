import cv2
import numpy as np

from glob import glob
from tqdm import tqdm
from loon_yolov3.common import target_width, target_height, grid_width_ratio, grid_height_ratio, anchor_width, anchor_height, category


def load_data():
    path = "D:/Dataset/loon_rpn_split"
    # path = "E:/Dataset/image/loon_rpn_split"

    x_data, y_data = [], []

    img_list = glob(f"{path}/*.jpg")

    for filename in tqdm(img_list):
        filename = filename[filename.find("\\") + 1:filename.find(".jpg")]

        img = cv2.resize(
            src=cv2.imread(f"{path}/{filename}.jpg"),
            dsize=(target_width, target_height),
            interpolation=cv2.INTER_AREA)
        x_data.append(img)

        label_tensor = np.zeros(shape=(grid_height_ratio, grid_width_ratio, 5 + len(category)))

        with open(f"{path}/{filename}.txt", "r") as reader:
            lines = reader.readlines()
            for line in lines:
                c, x, y, w, h = line[:-1].split(" ")
                c, x, y, w, h = int(c), float(x), float(y), float(w), float(h)
                grid_x, grid_y = int(x * grid_width_ratio), int(y * grid_height_ratio)
                x, y = x * grid_width_ratio - grid_x, y * grid_height_ratio - grid_y

                label_tensor[grid_y, grid_x, 0] = x
                label_tensor[grid_y, grid_x, 1] = y
                label_tensor[grid_y, grid_x, 2] = w / anchor_width
                label_tensor[grid_y, grid_x, 3] = h / anchor_height

                label_tensor[grid_y, grid_x, 4] = 1.
                label_tensor[grid_y, grid_x, 5 + int(c)] = 1.

        y_data.append(label_tensor)

    x_data, y_data = np.asarray(x_data), np.asarray(y_data)
    indexes = np.arange(x_data.shape[0])
    np.random.shuffle(indexes)
    x_data, y_data = x_data[indexes], y_data[indexes]

    return (x_data[:int(x_data.shape[0] * .8)], y_data[:int(y_data.shape[0] * .8)]), (x_data[int(x_data.shape[0] * .8):], y_data[int(y_data.shape[0] * .8):])