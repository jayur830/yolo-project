import cv2
import numpy as np

from glob import glob
from tqdm import tqdm
from utils import convert_abs_to_yolo
from loon.vars import target_width, target_height, grid_width_ratio, grid_height_ratio, category


def load_data():
    x_data, y_data = [], []

    img_list = glob("./loon_rpn_split/*.jpg")

    for filename in tqdm(img_list):
        filename = filename[filename.find("\\") + 1:filename.find(".jpg")]

        img = cv2.resize(
            src=cv2.imread(f"./loon_rpn_split/{filename}.jpg"),
            dsize=(target_width, target_height),
            interpolation=cv2.INTER_AREA)
        x_data.append(img)

        label_tensor = np.zeros(shape=(grid_height_ratio, grid_width_ratio, 5 + len(category)))

        with open(f"./loon_rpn_split/{filename}.txt", "r") as reader:
            lines = reader.readlines()
            for line in lines:
                c, x, y, w, h = line[:-1].split(" ")

                x1 = (float(x) - float(w) * .5) * target_width
                y1 = (float(y) - float(h) * .5) * target_height
                x2 = (float(x) + float(w) * .5) * target_width
                y2 = (float(y) + float(h) * .5) * target_height

                grid_x, grid_y, x, y, w, h = convert_abs_to_yolo(target_width, target_height, grid_width_ratio, grid_height_ratio, [(x1 + x2) * .5, (y1 + y2) * .5, x2 - x1, y2 - y1])

                label_tensor[grid_y, grid_x, 0] = x
                label_tensor[grid_y, grid_x, 1] = y
                label_tensor[grid_y, grid_x, 2] = w
                label_tensor[grid_y, grid_x, 3] = h
                label_tensor[grid_y, grid_x, 4] = 1.
                label_tensor[grid_y, grid_x, 5 + int(c)] = 1.

        y_data.append(label_tensor)

    return np.asarray(x_data), np.asarray(y_data)