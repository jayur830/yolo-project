import cv2
import numpy as np

# from yolo.dataset import load_data
from yolo.maplestory_loon.dataset import load_data
# from yolo.model import model
from yolo.maplestory_loon.model import model
from yolo.yolo_converter import convert_yolo_to_abs
from yolo.yolo_util import high_confidence_vector

if __name__ == '__main__':
    label_text = ["up", "down", "left", "right"]

    data_x, data_y_loc, data_y_size, data_y_cls = load_data()
    indexes = np.random.shuffle(np.arange(0, data_x.shape[0]))
    data_x, data_y_loc, data_y_size, data_y_cls = \
        data_x[indexes][0], \
        data_y_loc[indexes][0], \
        data_y_size[indexes][0], \
        data_y_cls[indexes][0]
    train_x, train_y_loc, train_y_size, train_y_cls = \
        data_x[:int(data_x.shape[0] * 0.8)], \
        data_y_loc[:int(data_y_loc.shape[0] * 0.8)], \
        data_y_size[:int(data_y_size.shape[0] * 0.8)], \
        data_y_cls[:int(data_y_cls.shape[0] * 0.8)]
    test_x, test_y_loc, test_y_size, test_y_cls = \
        data_x[int(data_x.shape[0] * 0.8):], \
        data_y_loc[int(data_y_loc.shape[0] * 0.8):], \
        data_y_size[int(data_y_size.shape[0] * 0.8):], \
        data_y_cls[int(data_y_cls.shape[0] * 0.8):]

    model = model()

    print(f"src_x shape: {data_x.shape}")
    print(f"src_y_loc shape: {data_y_loc.shape}")
    print(f"src_y_size shape: {data_y_size.shape}")
    print(f"src_y_cls shape: {data_y_cls.shape}")
    print(f"train_x shape: {train_x.shape}")
    print(f"train_y_loc shape: {train_y_loc.shape}")
    print(f"train_y_size shape: {train_y_size.shape}")
    print(f"train_y_cls shape: {train_y_cls.shape}")
    print(f"test_x shape: {test_x.shape}")
    print(f"test_y_loc shape: {test_y_loc.shape}")
    print(f"test_y_size shape: {test_y_size.shape}")
    print(f"test_y_cls shape: {test_y_cls.shape}")
    print(f"model input shape: {model.input_shape}")
    print(f"model output shape: {model.output_shape}")

    model.fit(
        x=train_x,
        y=(train_y_loc, train_y_size, train_y_cls),
        batch_size=2,
        epochs=100)

    # model.save(filepath="loon_model.h5")
    #
    # model = tf.keras.models.load_model(filepath="loon_model.h5")

    test_set = test_x

    for i in range(test_set.shape[0]):
        loc_regression_output, size_regression_output, classification_output = model.predict(test_set[i].reshape((1,) + test_set[i].shape))
        vectors = high_confidence_vector(loc_regression_output, size_regression_output)
        img = test_set[i]
        for j in range(len(vectors)):
            grid_x, grid_y, x, y, w, h = vectors[j]
            x1, y1, x2, y2 = convert_yolo_to_abs(512, 128, 16, 4, [grid_x, grid_y, x, y, w, h])
            img = cv2.rectangle(
                img=img,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=(0, 0, 255),
                thickness=2)
            label_index = int(classification_output[0, grid_y, grid_x, :].argmax())
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
