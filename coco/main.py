# import tensorflow as tf
import os
import cv2

from glob import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from utils import convert_abs_to_yolo, convert_yolo_to_abs

if __name__ == '__main__':
    img_list = glob("E:/Dataset/image/coco/train2017/*.jpg")

    with open("E:/Dataset/image/coco/classes.txt", "r") as reader:
        class_list = [filename[:-1] for filename in reader.readlines()]

    executor = ThreadPoolExecutor(max_workers=16)

    def load(_filename):
        img = cv2.imread(_filename)
        path = _filename[:_filename.index(".jpg")]

        if os.path.exists(path + ".txt"):
            with open(path + ".txt", "r") as reader:
                labels = [label_filename[:-1] for label_filename in reader.readlines()]

            # print(img.shape)
            for label in labels:
                class_num, x, y, w, h = label.split(" ")
                x = int(float(x) * img.shape[1])
                y = int(float(y) * img.shape[0])
                w = int(float(w) * img.shape[1])
                h = int(float(h) * img.shape[0])

                grid_x, grid_y, x, y, w, h = convert_abs_to_yolo(img.shape[1], img.shape[0], 52, 52, [x, y, w, h])
                # x1, y1, x2, y2 = convert_yolo_to_abs(img.shape[1], img.shape[0], 52, 52, [grid_x, grid_y, x, y, w, h])

                img = cv2.rectangle(
                    img=img,
                    pt1=(x1, y1),
                    pt2=(x2, y2),
                    color=(0, 0, 255),
                    thickness=2)

        cv2.imshow("test", img)
        cv2.waitKey()

    futures = []
    for filename in tqdm(img_list):
        futures.append(executor.submit(load, filename))

    for future in tqdm(futures):
        future.result()
