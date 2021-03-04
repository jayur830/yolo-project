from utils import train
from wildlife.model import yolo_model

if __name__ == '__main__':
    train(
        paths=["D:/Dataset/wildlife_yolo"],
        # paths=["E:/Dataset/image/wildlife_yolo"],
        model_function=yolo_model,
        target_width=416,
        target_height=416,
        grid_width_ratio=13,
        grid_height_ratio=13,
        anchor_width=5,
        anchor_height=5,
        epochs=100,
        batch_size=2)
