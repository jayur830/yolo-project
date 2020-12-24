import cv2
import numpy as np

from glob import glob
from tqdm import tqdm
from utils import convert_abs_to_yolo
from lpr.vars import target_width, target_height, grid_width_ratio, grid_height_ratio


def load_data():
    x_data, y_data = [], []

    categories = ["lane_day_ag_5", "lane_day_etc_1"]

    for category in categories:
        file_list = glob(f"./{category}/*.jpg")

        for filename in tqdm(file_list):
            filename = filename[filename.find("\\") + 1:filename.find(".jpg")]

            img = cv2.imread(f"./{category}/{filename}.jpg")
            label_tensor = np.zeros(shape=(grid_height_ratio, grid_width_ratio, 5))

            with open(f"./{category}/{filename}.txt", "r") as reader:
                lines = reader.readlines()
                if len(lines) == 0:
                    continue

                for line in lines:
                    _, x, y, w, h = line[:-1].split(" ")
                    x, y, w, h = float(x) * target_width, float(y) * target_height, float(w) * target_width, float(h) * target_height
                    x1, y1, x2, y2 = x - w * .5, y - h * .5, x + w * .5, y + h * .5
                    grid_x, grid_y, x, y, w, h = convert_abs_to_yolo(target_width, target_height, grid_width_ratio, grid_height_ratio, [(x1 + x2) * .5, (y1 + y2) * .5, x2 - x1, y2 - y1])

                    label_tensor[grid_y, grid_x, 0] = x
                    label_tensor[grid_y, grid_x, 1] = y
                    label_tensor[grid_y, grid_x, 2] = w
                    label_tensor[grid_y, grid_x, 3] = h
                    label_tensor[grid_y, grid_x, 4] = 1.
            x_data.append(img)
            y_data.append(label_tensor)

    x_data, y_data = np.asarray(x_data), np.asarray(y_data)
    indexes = np.arange(x_data.shape[0])
    np.random.shuffle(indexes)
    x_data, y_data = x_data[indexes], y_data[indexes]

    return (x_data[:int(x_data.shape[0] * .8)], y_data[:int(y_data.shape[0] * .8)]), (x_data[int(x_data.shape[0] * .8):], y_data[int(y_data.shape[0] * .8):])
