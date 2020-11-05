import cv2
import numpy as np

from yolo.maplestory_loon_v2.dataset import load_data
from yolo.maplestory_loon_v2.model import model
from yolo.maplestory_loon_v2.yolo_converter import convert_yolo_to_abs
from yolo.maplestory_loon_v2.yolo_util import high_confidence_vector
from yolo.maplestory_loon_v2.common import factor, grid_width, grid_height

if __name__ == '__main__':
    label_text = ["up", "down", "left", "right"]

    data_x, data_y = load_data()
    indexes = np.random.shuffle(np.arange(0, data_x.shape[0]))
    data_x, data_y = data_x[indexes][0], data_y[indexes][0]
    train_x, train_y = data_x[:int(data_x.shape[0] * 0.8)], data_y[:int(data_y.shape[0] * 0.8)]
    test_x, test_y = data_x[int(data_x.shape[0] * 0.8):], data_y[int(data_y.shape[0] * 0.8):]

    model = model()

    print(f"src_x shape: {data_x.shape}")
    print(f"src_y_loc shape: {data_y.shape}")
    print(f"train_x shape: {train_x.shape}")
    print(f"train_y_loc shape: {train_y.shape}")
    print(f"test_x shape: {test_x.shape}")
    print(f"test_y_loc shape: {test_y.shape}")
    print(f"model input shape: {model.input_shape}")
    print(f"model output shape: {model.output_shape}")

    model.fit(
        x=train_x,
        y=train_y,
        batch_size=2,
        epochs=100,
        validation_split=0.2)

    # model.save(filepath="loon_model.h5")
    #
    # model = tf.keras.models.load_model(filepath="loon_model.h5")

    test_set = test_x

    for i in range(test_set.shape[0]):
        output = model.predict(test_set[i].reshape((1,) + test_set[i].shape))
        vectors = high_confidence_vector(output)
        img = test_set[i]
        for j in range(len(vectors)):
            grid_x, grid_y, x, y, w, h = vectors[j]
            x, y, w, h = x / factor, y / factor, w / factor, h / factor
            x1, y1, x2, y2 = convert_yolo_to_abs(512, 128, grid_width, grid_height, [grid_x, grid_y, x, y, w, h])
            img = cv2.rectangle(
                img=img,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=(0, 0, 255),
                thickness=2)
            label_index = int(output[0, grid_y, grid_x, 5:].argmax())
            if label_index != 4:
                img = cv2.putText(
                    img=img,
                    text=label_text[label_index],
                    org=(x1, y1 - 6),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 0, 255),
                    thickness=2)
        cv2.imshow("test", img)
        cv2.waitKey()
