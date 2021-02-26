import cv2
import numpy as np

from glob import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from utils import convert_abs_to_yolo
from loon_yolov3.common import target_width, target_height, grid_width_ratio, grid_height_ratio, category


def load_data():
    x_data, y_data = [], []
    img_list = glob("D:/Dataset/loon_rpn_split/*.jpg")
    executor = ThreadPoolExecutor(16)

    def load(filename, _x_data, _y_data):
        filename = filename[filename.find("\\") + 1:filename.find(".jpg")]

        img = cv2.resize(
            src=cv2.imread(f"D:/Dataset/loon_rpn_split/{filename}.jpg"),
            dsize=(target_width, target_height),
            interpolation=cv2.INTER_AREA) / 255.
        _x_data.append(img)

        label_tensor = np.zeros(shape=(grid_height_ratio, grid_width_ratio, 5 + len(category)))

        with open(f"D:/Dataset/loon_rpn_split/{filename}.txt", "r") as reader:
            lines = reader.readlines()
            for line in lines:
                c, x, y, w, h = line[:-1].split(" ")
                c, x, y, w, h = int(c), float(x), float(y), float(w), float(h)

                x1 = (x - w * .5) * target_width
                y1 = (y - h * .5) * target_height
                x2 = (x + w * .5) * target_width
                y2 = (y + h * .5) * target_height

                grid_x, grid_y, x, y, w, h = convert_abs_to_yolo(
                    target_width,
                    target_height,
                    grid_width_ratio,
                    grid_height_ratio,
                    [(x1 + x2) * .5, (y1 + y2) * .5, x2 - x1, y2 - y1])

                label_tensor[grid_y, grid_x, 0] = x + grid_x
                label_tensor[grid_y, grid_x, 1] = y + grid_y
                label_tensor[grid_y, grid_x, 2] = w * grid_width_ratio
                label_tensor[grid_y, grid_x, 3] = h * grid_height_ratio
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
