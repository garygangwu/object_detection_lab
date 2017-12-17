# Object Detection

Leverage existing tensor flow pre-trained models to detect objects in images and videos

## Instruction

* Download the pretrained models from the [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).
* Make sure the update the model file path in [object_detection.py] (https://github.com/garygangwu/object_detection_lab/blob/master/object_detection.py)

## Usage

`python image|video file_to_process`

## Examples of results

### SSD (Single Shot MultiBox Detecto)
|Original Image|Result|
|:---:|:---:|
|<img src="examples/person.jpg"/>|<img src="output/person.png"/>|

### R-FCN
|Original Image|Result|
|:---:|:---:|
|<img src="examples/dog.jpg"/>|<img src="output/dog.png"/>|

### Faster R-CNN
|Original Image|Result|
|:---:|:---:|
|<img src="examples/giraffe.jpg"/>|<img src="output/giraffe.png"/>|
|<img src="examples/horses.jpg"/>|<img src="output/horses.png"/>|
