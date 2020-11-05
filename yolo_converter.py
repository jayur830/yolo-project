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
        float(w),
        float(h)
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
        int(grid_cell_width * (grid_x + x)) + int(w),
        int(grid_cell_height * (grid_y + y)) + int(h)
    ]
