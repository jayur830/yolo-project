import cv2
import numpy as np

from yolo.yolo_converter import convert_abs_to_yolo


def load_data():
    data_x, data_y_loc, data_y_size = [], [], []
    with open("indian_eu_br_us.csv", "r") as reader:
        records = reader.readlines()[1:]
        records = [record[:-1] for record in records]

    grid_width, grid_height = 7, 7

    for record in records:
        filename, img_width, img_height, x1, y1, x2, y2 = record.split(",")
        img_width, img_height, x1, y1, x2, y2 = \
            int(img_width), \
            int(img_height), \
            int(float(x1) * float(img_width)), \
            int(float(y1) * float(img_height)), \
            int(float(x2) * float(img_width)), \
            int(float(y2) * float(img_height))
        img = cv2.imread(f"./indian_eu_br_us_plates_compressed/{filename}.jpeg")
        data_x.append(cv2.resize(img, dsize=(224, 224)))
        loc_tensor = np.zeros(shape=(grid_height, grid_width, 3))
        size_tensor = np.zeros(shape=(grid_height, grid_width, 2))
        grid_x, grid_y, x, y, w, h = convert_abs_to_yolo(img_width, img_height, grid_width, grid_height, [x1, y1, x2 - x1, y2 - y1])
        loc_tensor[grid_y, grid_x, 0] = x
        loc_tensor[grid_y, grid_x, 1] = y
        loc_tensor[grid_y, grid_x, 2] = 1.
        size_tensor[grid_y, grid_x, 0] = w
        size_tensor[grid_y, grid_x, 1] = h
        data_y_loc.append(loc_tensor)
        data_y_size.append(size_tensor)
    return np.asarray(data_x), np.asarray(data_y_loc), np.array(data_y_size)
