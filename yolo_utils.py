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
        int(grid_cell_width * (grid_x + x)),
        int(grid_cell_height * (grid_y + y)),
        int(grid_cell_width * (grid_x + x)) + int(w * img_width),
        int(grid_cell_height * (grid_y + y)) + int(h * img_height)
    ]


def high_confidence_vector(tensor, threshold=0.5):
    data = []
    for n in range(tensor.shape[0]):
        for h in range(tensor.shape[1]):
            for w in range(tensor.shape[2]):
                if tensor[n, h, w, 4] >= threshold:
                    data.append([w, h] + tensor[n, h, w, :4].tolist())
    return data
