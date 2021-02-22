import cv2
import numpy as np
import os

from xml.etree.ElementTree import parse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from utils import convert_abs_to_yolo
from mask_detection.common import target_width, target_height, grid_width_ratio, grid_height_ratio, category


def load_data():
    x_data, y_data = [], []

    filenames = os.listdir("./data/images")
    filenames = [filename[:filename.find(".png")] for filename in filenames]

    executor = ThreadPoolExecutor(16)

    def load(filename, _x_data, _y_data):
        _x_data.append(cv2.resize(
            src=cv2.imread(f"./data/images/{filename}.png"),
            dsize=(target_width, target_height),
            interpolation=cv2.INTER_AREA))

        tree = parse(f"./data/annotations/{filename}.xml")
        root = tree.getroot()

        size = root.find("size")
        height, width = int(size.find("height").text), int(size.find("width").text)

        label_tensor = np.zeros((grid_height_ratio, grid_width_ratio, 5 + len(category)))

        objects = root.findall("object")
        for obj in objects:
            c = category[obj.find("name").text]
            bbox = obj.find("bndbox")
            x1, y1, x2, y2 = \
                int(bbox.find("xmin").text), \
                int(bbox.find("ymin").text), \
                int(bbox.find("xmax").text), \
                int(bbox.find("ymax").text)

            x1 *= target_width / width
            x2 *= target_width / width
            y1 *= target_height / height
            y2 *= target_height / height

            grid_x, grid_y, x, y, w, h = convert_abs_to_yolo(
                target_width,
                target_height,
                grid_width_ratio,
                grid_height_ratio,
                [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
            label_tensor[grid_y, grid_x, 0] = x
            label_tensor[grid_y, grid_x, 1] = y
            label_tensor[grid_y, grid_x, 2] = w
            label_tensor[grid_y, grid_x, 3] = h
            label_tensor[grid_y, grid_x, 4] = 1.
            label_tensor[grid_y, grid_x, 5 + c] = 1.

        _y_data.append(label_tensor)

    futures = []
    for filename in tqdm(filenames):
        futures.append(executor.submit(load, filename, x_data, y_data))
    for future in futures:
        future.result()

    return np.asarray(x_data), np.asarray(y_data)
