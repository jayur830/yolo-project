import tensorflow as tf
import numpy as np
import cv2


def gpu_init():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


"""
# bbox: [x, y, w, h]
# yolo_loc: [grid_x, grid_y, x, y, w, h]
"""
def convert_abs_to_yolo(
        img_width: int,
        img_height: int,
        grid_width_ratio: int,
        grid_height_ratio: int,
        bbox: list):
    grid_cell_width, grid_cell_height = \
        img_width / grid_width_ratio, \
        img_height / grid_height_ratio

    x, y, w, h = bbox
    return [
        int(x / grid_cell_width),
        int(y / grid_cell_height),
        (x % grid_cell_width) / grid_cell_width,
        (y % grid_cell_height) / grid_cell_height,
        float(w / img_width),
        float(h / img_height)
    ]


def convert_yolo_to_abs(
        img_width: int,
        img_height: int,
        grid_width_ratio: int,
        grid_height_ratio: int,
        yolo_loc: list):
    grid_cell_width, grid_cell_height = \
        img_width / grid_width_ratio, \
        img_height / grid_height_ratio

    grid_x, grid_y, x, y, w, h = yolo_loc
    return [
        int(int(grid_cell_width * (grid_x + x)) - int(w * img_width) / 2),
        int(int(grid_cell_height * (grid_y + y)) - int(h * img_height) / 2),
        int(int(grid_cell_width * (grid_x + x)) + int(w * img_width) / 2),
        int(int(grid_cell_height * (grid_y + y)) + int(h * img_height) / 2)
    ]


def high_confidence_vector(yolo_tensor, threshold: float = .5):
    if len(yolo_tensor.shape) != 3:
        return []
    vectors = []
    for h in range(yolo_tensor.shape[0]):
        for w in range(yolo_tensor.shape[1]):
            if yolo_tensor[h, w, 4] >= threshold:
                vector = [w, h] + yolo_tensor[h, w, :4].tolist()
                if yolo_tensor.shape[-1] > 5:
                    vector += [yolo_tensor[h, w, 5:].argmax()]
                vectors.append(vector)
    return vectors


@tf.function
def predict(model, x):
    return model(x)


def iou_tensor(a, b):
    intersection = tf.maximum(tf.minimum(a[:, :, :, 2], b[:, :, :, 2]) - tf.maximum(a[:, :, :, 0], b[:, :, :, 0]), 0) * tf.maximum(tf.minimum(a[:, :, :, 3], b[:, :, :, 3]) - tf.maximum(a[:, :, :, 1], b[:, :, :, 1]), 0)
    union = a[:, :, :, 2] * a[:, :, :, 3] + b[:, :, :, 2] * b[:, :, :, 3] - intersection
    return intersection / union


def on_batch_end_callback(model, x, target_width, target_height, grid_width_ratio, grid_height_ratio, step):
    img = x[step].copy()
    x = cv2.resize(
        src=img,
        dsize=(target_width, target_height),
        interpolation=cv2.INTER_AREA)
    x = x.reshape((1,) + x.shape)
    output = np.asarray(predict(model, x))
    vectors = high_confidence_vector(output[0])
    for vector in vectors:
        x1, y1, x2, y2 = convert_yolo_to_abs(target_width, target_height, grid_width_ratio, grid_height_ratio,
                                             vector[:-1])
        img = cv2.rectangle(
            img=img,
            pt1=(int(x1), int(y1)),
            pt2=(int(x2), int(y2)),
            color=(0, 0, 255),
            thickness=2)
    cv2.imshow("test", img)
    cv2.waitKey(1)


if __name__ == '__main__':
    # [x1, y1, x2, y2]
    a = [1, 1, 2, 2]
    b = [5, 3, 6, 4]

    print(iou(a, b))
