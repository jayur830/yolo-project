import tensorflow as tf


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
        int(grid_cell_width * (grid_x + x)) - int(w * img_width) / 2,
        int(grid_cell_height * (grid_y + y)) - int(h * img_height) / 2,
        int(grid_cell_width * (grid_x + x)) + int(w * img_width) / 2,
        int(grid_cell_height * (grid_y + y)) + int(h * img_height) / 2
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
