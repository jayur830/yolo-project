import tensorflow as tf
import numpy as np
import cv2
import os

from tqdm import tqdm
from glob import glob
from concurrent.futures import ThreadPoolExecutor


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


def load_data(
        paths: [str],
        target_width: int,
        target_height: int,
        grid_width_ratio: int,
        grid_height_ratio: int,
        anchor_width: float,
        anchor_height: float,
        shuffle: bool = True,
        validation_split: float = .2):
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
                label_tensor[grid_y, grid_x, 2] = w / (anchor_width / grid_width_ratio)
                label_tensor[grid_y, grid_x, 3] = h / (anchor_height / grid_height_ratio)
                label_tensor[grid_y, grid_x, 4] = 1.
                label_tensor[grid_y, grid_x, 5 + int(c)] = 1.
        y_data.append(label_tensor)

    executor = ThreadPoolExecutor(16)
    classes = []

    for path in paths:
        img_list = glob(f"{path}/*.jpg")
        with open(f"{path}/classes.txt", "r") as reader:
            _classes = [label[:-1] for label in reader.readlines()]

        futures = []
        for filename in tqdm(img_list):
            futures.append(executor.submit(load, filename, path, len(_classes)))
        for future in tqdm(futures):
            future.result()
        classes += _classes

    x_data, y_data = np.asarray(x_data), np.asarray(y_data)

    if shuffle:
        indexes = np.arange(x_data.shape[0])
        np.random.shuffle(indexes)
        x_data, y_data = x_data[indexes], y_data[indexes]
        return (x_data[:int(x_data.shape[0] * (1 - validation_split))], y_data[:int(y_data.shape[0] * (1 - validation_split))]), \
               (x_data[int(x_data.shape[0] * (1 - validation_split)):], y_data[int(y_data.shape[0] * (1 - validation_split)):]), classes
    else:
        return x_data, y_data


@tf.function
def predict(model, x):
    return model(x)


step = 0

def on_batch_end(
        model,
        x_data,
        classes,
        batch_size,
        target_width,
        target_height,
        grid_width_ratio,
        grid_height_ratio,
        anchor_width,
        anchor_height,
        step_interval):
    def _on_batch_end(_1, logs):
        global step
        if step >= x_data.shape[0]:
            step = 0
        elif step % step_interval == 0:
            img = x_data[step].copy()
            x = cv2.resize(
                src=img,
                dsize=(target_width, target_height),
                interpolation=cv2.INTER_AREA)
            x = x.reshape((1,) + x.shape)
            output = np.asarray(predict(model, x))
            vectors = high_confidence_vector(output[0])
            for vector in vectors:
                c_x, c_y, t_x, t_y, t_w, t_h, class_index = vector
                b_x = (t_x + c_x) * target_width / grid_width_ratio
                b_y = (t_y + c_y) * target_height / grid_height_ratio
                b_w = t_w * target_width * anchor_width / grid_width_ratio
                b_h = t_h * target_height * anchor_height / grid_height_ratio
                x1, y1, x2, y2 = int(b_x - b_w * .5), int(b_y - b_h * .5), int(b_x + b_w * .5), int(b_y + b_h * .5)

                img = cv2.rectangle(
                    img=img,
                    pt1=(round(x1), round(y1)),
                    pt2=(round(x2), round(y2)),
                    color=(0, 0, 255),
                    thickness=2)
                img = cv2.putText(
                    img=img,
                    text=classes[class_index],
                    org=(round(x1), round(y1) - 5),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=.5,
                    color=(0, 0, 255),
                    thickness=2)
            cv2.imshow("test", img)
            cv2.waitKey(1)
        step += batch_size
    return _on_batch_end


def test(
        model,
        x_data,
        classes,
        target_width,
        target_height,
        grid_width_ratio,
        grid_height_ratio,
        anchor_width,
        anchor_height):
    for i in range(x_data.shape[0]):
        img = x_data[i].copy()
        output = np.asarray(predict(model, x_data[i].reshape((1,) + x_data[i].shape)))
        vectors = high_confidence_vector(output[0])
        for vector in vectors:
            print(vector)
            c_x, c_y, t_x, t_y, t_w, t_h, class_index = vector
            b_x = (t_x + c_x) * target_width / grid_width_ratio
            b_y = (t_y + c_y) * target_height / grid_height_ratio
            b_w = t_w * target_width * anchor_width / grid_width_ratio
            b_h = t_h * target_height * anchor_height / grid_height_ratio
            x1, y1, x2, y2 = int(b_x - b_w * .5), int(b_y - b_h * .5), int(b_x + b_w * .5), int(b_y + b_h * .5)

            img = cv2.rectangle(
                img=img,
                pt1=(round(x1), round(y1)),
                pt2=(round(x2), round(y2)),
                color=(0, 0, 255),
                thickness=2)
            img = cv2.putText(
                img=img,
                text=classes[class_index],
                org=(round(x1), round(y1) - 5),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=.5,
                color=(0, 0, 255),
                thickness=2)
        print("=" * 40)
        cv2.imshow("test", img)
        cv2.waitKey()


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


def iou_tensor(a, b):
    intersection = tf.maximum(tf.minimum(a[:, :, :, 2], b[:, :, :, 2]) - tf.maximum(a[:, :, :, 0], b[:, :, :, 0]), 0) * tf.maximum(tf.minimum(a[:, :, :, 3], b[:, :, :, 3]) - tf.maximum(a[:, :, :, 1], b[:, :, :, 1]), 0)
    union = a[:, :, :, 2] * a[:, :, :, 3] + b[:, :, :, 2] * b[:, :, :, 3] - intersection
    return intersection / union


def train(
        paths,
        model_function,
        target_width: int,
        target_height: int,
        grid_width_ratio: int,
        grid_height_ratio: int,
        anchor_width: float,
        anchor_height: float,
        epochs: int,
        batch_size: int):
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    (x_train, y_train), (x_test, y_test), classes = load_data(
        paths=paths,
        target_width=target_width,
        target_height=target_height,
        grid_width_ratio=grid_width_ratio,
        grid_height_ratio=grid_height_ratio,
        anchor_width=anchor_width,
        anchor_height=anchor_height)
    model = model_function(len(classes))

    model.fit(
        x=x_train,
        y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=.2,
        callbacks=[tf.keras.callbacks.LambdaCallback(on_batch_end=on_batch_end(
            model=model,
            x_data=x_train,
            classes=classes,
            batch_size=batch_size,
            target_width=target_width,
            target_height=target_height,
            grid_width_ratio=grid_width_ratio,
            grid_height_ratio=grid_height_ratio,
            anchor_width=anchor_width,
            anchor_height=anchor_height,
            step_interval=20))])

    model.save(filepath="model.h5")
    model = tf.keras.models.load_model(filepath="model.h5", compile=False)

    test(
        model=model,
        x_data=x_test,
        classes=classes,
        target_width=target_width,
        target_height=target_height,
        grid_width_ratio=grid_width_ratio,
        grid_height_ratio=grid_height_ratio,
        anchor_width=anchor_width,
        anchor_height=anchor_height)
