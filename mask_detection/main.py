from utils import train
from mask_detection.model import yolo_model

if __name__ == '__main__':
    train(
        paths=["D:/Dataset/mask_detection"],
        # paths=["E:/Dataset/image/mask_detection"],
        model_function=yolo_model,
        target_width=224,
        target_height=224,
        grid_width_ratio=14,
        grid_height_ratio=14,
        anchor_width=3,
        anchor_height=3,
        epochs=100,
        batch_size=2)
