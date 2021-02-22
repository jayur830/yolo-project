import numpy as np
import cv2

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from utils import convert_abs_to_yolo
from traffic_signs.common import path
from traffic_signs.vars import target_width, target_height, grid_width_ratio, grid_height_ratio


def load_data():
    with open(f"{path}/classes.names", "r") as reader:
        classes = [class_name[:-1] for class_name in reader.readlines()]

    executor = ThreadPoolExecutor(16)

    def load(_img_path, data_x, data_y):
        img = cv2.resize(
            src=cv2.imread(img_path),
            dsize=(target_width, target_height),
            interpolation=cv2.INTER_AREA)
        data_x.append(img)

        with open(_img_path[:_img_path.index(".jpg")] + ".txt", "r") as reader:
            labels = [line[:-1].split(" ") for line in reader.readlines()]

        label_tensor = np.zeros(shape=(grid_height_ratio, grid_width_ratio, 5 + len(classes)))
        for label in labels:
            class_num, x, y, w, h = label
            class_num, x, y, w, h = \
                int(class_num), \
                target_width * (float(x) - float(w) / 2), \
                target_height * (float(y) - float(h) / 2), \
                target_width * (float(x) + float(w) / 2), \
                target_height * (float(y) + float(h) / 2)
            grid_x, grid_y, x, y, w, h = convert_abs_to_yolo(target_width, target_height, grid_width_ratio, grid_height_ratio, [x, y, w, h])
            label_tensor[grid_y, grid_x, 0] = x
            label_tensor[grid_y, grid_x, 1] = y
            label_tensor[grid_y, grid_x, 2] = w
            label_tensor[grid_y, grid_x, 3] = h
            label_tensor[grid_y, grid_x, 4] = 1.
            label_tensor[grid_y, grid_x, 5 + class_num] = 1.
        data_y.append(label_tensor)

    with open(f"{path}/train.txt", "r") as reader:
        train_paths = [filepath[:-1].format(path) for filepath in reader.readlines()[:-1]]

    train_x, train_y, futures = [], [], []

    for img_path in tqdm(train_paths):
        futures.append(executor.submit(load, img_path, train_x, train_y))
    for future in tqdm(futures):
        future.result()

    with open(f"{path}/test.txt", "r") as reader:
        test_paths = [filepath[:-1].format(path) for filepath in reader.readlines()[:-1]]

    test_x, test_y, futures = [], [], []

    for img_path in tqdm(test_paths):
        futures.append(executor.submit(load, img_path, test_x, test_y))
    for future in tqdm(futures):
        future.result()

    return (np.asarray(train_x), np.asarray(train_y)), (np.asarray(test_x), np.asarray(test_y))
