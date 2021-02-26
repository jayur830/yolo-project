# YOLOv3 Implementation using Tensorflow

We implemented the YOLO architecture using the Tensorflow framework.

## Architecture
## Preprocessing

Of the training datasets x and y, x is an image and y is a label in the following format.

    [class] [x] [y] [w] [h]

[class] is the class number for classification. x, y, w, h are coordinate information of a specific object in the image. x and y denote the center of the object, w and h denote the width and height of the object, and these 4 values ​​are normalized to 0~1 based on the total size of the image pixel.

YOLO divides the image into a grid of S by S. In actual training, we multiply these four values ​​of the label by the grid size S so that the neural network can regress a larger number.

## Loss Function

The loss function for training the model is:

![](images/loss.png)

## Requirements

- Tensorflow 2.x
- OpenCV-Python
- Numpy

## Training

- 

## Loss And Recall

## Evaluation

## Conclusion

There are still many issues with the v3 model we made. In the future, we will do anything to improve accuracy and improve speed.