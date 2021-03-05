from utils import train
from lpr.model import yolo_model

if __name__ == '__main__':
    train(
        paths=["D:/Dataset/license_plates/crime_day_etc_1"],
        # paths=["E:/Dataset/image/license_plates/crime_day_etc_1"],
        model_function=yolo_model,
        target_width=640,
        target_height=368,
        grid_width_ratio=40,
        grid_height_ratio=23,
        anchors=[2.4, .7],
        epochs=100,
        batch_size=2)
