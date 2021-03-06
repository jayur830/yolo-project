from utils import train
from loon.model import yolo_model

if __name__ == '__main__':
    train(
        # paths=["D:/Dataset/loon_rpn_split"],
        paths=["E:/Dataset/image/loon_rpn_split"],
        model_function=yolo_model,
        target_width=512,
        target_height=128,
        grid_width_ratio=64,
        grid_height_ratio=16,
        anchors=[[5, 4], [4, 5]],
        epochs=100,
        batch_size=2)
