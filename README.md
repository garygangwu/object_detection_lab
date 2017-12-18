# Object Detection

Leverage existing tensor flow pre-trained models to detect objects in images and videos

|<a href="http://www.youtube.com/watch?feature=player_embedded&v=q2LmRzr0GHk" target="_blank"><img src="http://img.youtube.com/vi/q2LmRzr0GHk/0.jpg" width="240" height="180" border="10" /></a>|
|---|
|[Object detection under various weather conditions during driving](https://youtu.be/q2LmRzr0GHk)|

Another example to detect objects in an [indoor office space](https://vimeo.com/247742070)|

## Instruction

* Download the pretrained models from the [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).
* Make sure the update the model file path in [object_detection.py](https://github.com/garygangwu/object_detection_lab/blob/master/object_detection.py)

## Usage

Run `python object_detection.py --help` for available options

example: 
* `python object_detection.py --type image --image_file examples/dog.jpg`
* `python object_detection.py --type video --video_file examples/driving.mp4`

## Examples of results

### SSD (Single Shot MultiBox Detection) based on MobileNet
|Original Image|Result|
|:---:|:---:|
|<img src="examples/person.jpg"/>|<img src="output/person.png"/>|

### R-FCN based on ResNet-101
|Original Image|Result|
|:---:|:---:|
|<img src="examples/dog.jpg"/>|<img src="output/dog.png"/>|

### Faster R-CNN based on Inception-ResNet-v2
|Original Image|Result|
|:---:|:---:|
|<img src="examples/giraffe.jpg"/>|<img src="output/giraffe.png"/>|
|<img src="examples/horses.jpg"/>|<img src="output/horses.png"/>|
