from utils import train
from traffic_signs.model import yolo_model

if __name__ == '__main__':
    train(
        paths=["D:/Dataset/traffic_signs_yolo"],
        # paths=["E:/Dataset/image/traffic_signs_yolo"],
        model_function=yolo_model,
        target_width=512,
        target_height=256,
        grid_width_ratio=16,
        grid_height_ratio=8,
        anchors=[[3, 2]],
        epochs=100,
        batch_size=2)
