import numpy as np
import cv2

from glob import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from utils import convert_abs_to_yolo
from wildlife.common import path, target_width, target_height, grid_width_ratio, grid_height_ratio


def load_data(num_classes: int):
    x_data, y_data = [], []
    img_paths = glob(f"{path}/*/*.jpg")

    executor = ThreadPoolExecutor(16)

    def load(_filepath, _x_data, _y_data):
        img = cv2.resize(
            src=cv2.imread(_filepath),
            dsize=(target_width, target_height),
            interpolation=cv2.INTER_AREA)
        _x_data.append(img)

        with open(_filepath.replace(".jpg", ".txt"), "r") as reader:
            labels = [line.split(" ") for line in reader.readlines()]

        label_tensor = np.zeros(shape=(grid_height_ratio, grid_width_ratio, 5 + num_classes), dtype="float32")
        for label in labels:
            class_index, x, y, w, h = label
            class_index, x, y, w, h = \
                int(class_index), \
                target_width * float(x), \
                target_height * float(y), \
                target_width * float(w), \
                target_height * float(h)
            grid_x, grid_y, x, y, w, h = convert_abs_to_yolo(
                target_width,
                target_height,
                grid_width_ratio,
                grid_height_ratio,
                [x - w / 2, y - h / 2, x + w / 2, y + h / 2])
            label_tensor[grid_y, grid_x, 0] = x
            label_tensor[grid_y, grid_x, 1] = y
            label_tensor[grid_y, grid_x, 2] = w
            label_tensor[grid_y, grid_x, 3] = h
            label_tensor[grid_y, grid_x, 4] = 1.
            label_tensor[grid_y, grid_x, 5 + class_index] = 1.
        _y_data.append(label_tensor)

    futures = []
    for filepath in img_paths:
        futures.append(executor.submit(load, filepath, x_data, y_data))
    for future in tqdm(futures):
        future.result()

    return np.asarray(x_data), np.asarray(y_data)
