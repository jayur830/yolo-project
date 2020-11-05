

def abs_to_yolo(img_width, img_height, grid_width, grid_height, x1, y1, x2, y2):
    grid_cell_width, grid_cell_height = img_width / grid_width, img_height / grid_height
    x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    grid_x, grid_y = int(x / grid_cell_width), int(y / grid_cell_height)
    x, y = (x / grid_cell_width) - grid_x, (y / grid_cell_height) - grid_y
    return grid_x, grid_y, x, y, float(w), float(h)


def yolo_to_abs(img_width, img_height, grid_width, grid_height, grid_x, grid_y, x, y, w, h):
    grid_cell_width, grid_cell_height = img_width / grid_width, img_height / grid_height
    x, y = grid_cell_width * (grid_x + x), grid_cell_height * (grid_y + y)
    x1, y1, x2, y2 = int(x - (w / 2)), int(y - (h / 2)), int(x + (w / 2)), int(y + (h / 2))
    return x1, y1, x2, y2


def high_confidence_vector(tensor, threshold=0.5):
    data = []
    for n in range(tensor.shape[0]):
        for h in range(tensor.shape[1]):
            for w in range(tensor.shape[2]):
                if tensor[n, h, w, 4] >= threshold:
                    data.append([w, h] + tensor[n, h, w, :4].tolist())
    return data
