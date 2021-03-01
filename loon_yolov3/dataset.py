import math
import cv2
import numpy as np

from glob import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from utils import convert_abs_to_yolo
from loon_yolov3.common import \
    target_width, \
    target_height, \
    grid_width_ratio, \
    grid_height_ratio, \
    anchor_width, \
    anchor_height, \
    category


def load_data():
    x_data, y_data = [], []
    # img_list = glob("D:/Dataset/loon_rpn_split/*.jpg")
    img_list = glob("../loon/loon_rpn_split/*.jpg")
    executor = ThreadPoolExecutor(16)

    def load(filename, _x_data, _y_data):
        filename = filename[filename.find("\\") + 1:filename.find(".jpg")]

        img = cv2.resize(
            src=cv2.imread(f"../loon/loon_rpn_split/{filename}.jpg"),
            dsize=(target_width, target_height),
            interpolation=cv2.INTER_AREA) / 255.
        _x_data.append(img)

        label_tensor = np.zeros(shape=(grid_height_ratio, grid_width_ratio, 5 + len(category)))

        with open(f"../loon/loon_rpn_split/{filename}.txt", "r") as reader:
            lines = reader.readlines()
            for line in lines:
                c, x, y, w, h = line[:-1].split(" ")
                c, x, y, w, h = int(c), float(x), float(y), float(w), float(h)

                grid_x, grid_y, x, y, w, h = convert_abs_to_yolo(
                    target_width,
                    target_height,
                    grid_width_ratio,
                    grid_height_ratio,
                    [
                        x * target_width,
                        y * target_height,
                        w * target_width,
                        h * target_height
                    ])

                label_tensor[grid_y, grid_x, 0] = x
                label_tensor[grid_y, grid_x, 1] = y
                # label_tensor[grid_y, grid_x, 2] = w
                # label_tensor[grid_y, grid_x, 3] = h

                # label_tensor[grid_y, grid_x, 0] = math.log(x / (1 - x) + 1e-5)
                # label_tensor[grid_y, grid_x, 1] = math.log(y / (1 - y) + 1e-5)
                label_tensor[grid_y, grid_x, 2] = math.log(w / anchor_width + 1e-5)
                label_tensor[grid_y, grid_x, 3] = math.log(h / anchor_height + 1e-5)

                label_tensor[grid_y, grid_x, 4] = 1.
                label_tensor[grid_y, grid_x, 5 + c] = 1.
        _y_data.append(label_tensor)

    futures = []
    for filename in tqdm(img_list):
        futures.append(executor.submit(load, filename, x_data, y_data))
    for future in tqdm(futures):
        future.result()

    x_data, y_data = np.asarray(x_data), np.asarray(y_data)
    indexes = np.arange(x_data.shape[0])
    np.random.shuffle(indexes)
    x_data, y_data = x_data[indexes], y_data[indexes]

    return (x_data[:int(x_data.shape[0] * .8)], y_data[:int(y_data.shape[0] * .8)]), (x_data[int(x_data.shape[0] * .8):], y_data[int(y_data.shape[0] * .8):])
