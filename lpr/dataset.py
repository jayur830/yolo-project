import cv2
import numpy as np

from glob import glob
from tqdm import tqdm
from utils import convert_abs_to_yolo
from lpr.common import target_width, target_height, grid_width_ratio, grid_height_ratio
from concurrent.futures import ThreadPoolExecutor


def load_data(validation_split: float = .2):
    paths = [
        "D:/Dataset/license_plates/lane_day_ag_1",
        # "D:/Dataset/license_plates/lane_day_ag_5",
        # "D:/Dataset/license_plates/lane_day_etc_1"
    ]

    x_data, y_data = [], []

    def load(_filename, _path, num_classes):
        _filename = _filename[_filename.find("\\") + 1:_filename.find(".jpg")]

        img = cv2.resize(
            src=cv2.imread(f"{_path}/{_filename}.jpg"),
            dsize=(target_width, target_height),
            interpolation=cv2.INTER_AREA)
        x_data.append(img)

        label_tensor = np.zeros(shape=(grid_height_ratio, grid_width_ratio, 5 + num_classes))

        with open(f"{_path}/{_filename}.txt", "r") as reader:
            lines = reader.readlines()
            for line in lines:
                c, x, y, w, h = line[:-1].split(" ")
                c, x, y, w, h = int(c), float(x), float(y), float(w), float(h)
                grid_x, grid_y = int(x * grid_width_ratio), int(y * grid_height_ratio)
                x, y = x * grid_width_ratio - grid_x, y * grid_height_ratio - grid_y

                label_tensor[grid_y, grid_x, 0] = x
                label_tensor[grid_y, grid_x, 1] = y
                label_tensor[grid_y, grid_x, 2] = w
                label_tensor[grid_y, grid_x, 3] = h
                label_tensor[grid_y, grid_x, 4] = 1.
        y_data.append(label_tensor)

    executor = ThreadPoolExecutor(16)

    for path in paths:
        img_list = glob(f"{path}/*.jpg")
        futures = []
        for filename in tqdm(img_list):
            futures.append(executor.submit(load, filename, path))
        for future in tqdm(futures):
            future.result()

    x_data, y_data = np.asarray(x_data), np.asarray(y_data)
    indexes = np.arange(x_data.shape[0])
    np.random.shuffle(indexes)
    x_data, y_data = x_data[indexes], y_data[indexes]

    return (x_data[:int(x_data.shape[0] * (1 - validation_split))], y_data[:int(y_data.shape[0] * (1 - validation_split))]), \
           (x_data[int(x_data.shape[0] * (1 - validation_split)):], y_data[int(y_data.shape[0] * (1 - validation_split)):])
